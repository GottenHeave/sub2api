package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAIRealtimeCallsAcceptEndpointPrefix = "/v1/realtime/calls/"
	openAIRealtimeCallsAcceptEndpointSuffix = "/accept"
	openAIRealtimeRESTDefaultBaseURL        = "https://api.openai.com"
)

type OpenAIRealtimeRESTModelRef struct {
	Path  string
	Value string
}

type OpenAIRealtimeRESTRequest struct {
	CallID            string
	Body              []byte
	Model             string
	Endpoint          string
	Action            string
	ScheduleModelPath string
	ModelRefs         []OpenAIRealtimeRESTModelRef
}

type OpenAIRealtimeCallsAcceptRequest = OpenAIRealtimeRESTRequest

func (r *OpenAIRealtimeRESTRequest) StickySessionSeed() string {
	if r == nil {
		return ""
	}
	return strings.Join([]string{
		"openai-realtime-rest",
		strings.TrimSpace(r.Endpoint),
		strings.TrimSpace(r.CallID),
		strings.TrimSpace(r.Model),
	}, "|")
}

func ParseOpenAIRealtimeRESTRequest(c *gin.Context, body []byte) (*OpenAIRealtimeRESTRequest, error) {
	if c == nil || c.Request == nil || c.Request.URL == nil {
		return nil, fmt.Errorf("missing request context")
	}
	endpoint, callID, action, err := normalizeOpenAIRealtimeRESTEndpoint(c.Request.URL.Path)
	if err != nil {
		return nil, err
	}

	parsed := &OpenAIRealtimeRESTRequest{
		CallID:   callID,
		Body:     body,
		Endpoint: endpoint,
		Action:   action,
	}

	if endpoint == OpenAIRealtimeCallsAcceptUpstreamEndpoint(callID) {
		if len(body) == 0 {
			return nil, fmt.Errorf("request body is empty")
		}
		if !gjson.ValidBytes(body) {
			return nil, fmt.Errorf("failed to parse request body")
		}
		if sessionType := strings.TrimSpace(gjson.GetBytes(body, "type").String()); sessionType != "realtime" {
			return nil, fmt.Errorf("type must be realtime")
		}
		model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
		if model == "" {
			return nil, fmt.Errorf("model is required")
		}
		parsed.Model = model
		parsed.ScheduleModelPath = "model"
		parsed.ModelRefs = []OpenAIRealtimeRESTModelRef{{Path: "model", Value: model}}
		return parsed, nil
	}

	if len(body) == 0 || !gjson.ValidBytes(body) {
		return parsed, nil
	}
	parsed.ModelRefs = extractOpenAIRealtimeRESTModelRefs(endpoint, body)
	if len(parsed.ModelRefs) > 0 {
		parsed.Model = parsed.ModelRefs[0].Value
		parsed.ScheduleModelPath = parsed.ModelRefs[0].Path
	}
	return parsed, nil
}

func ParseOpenAIRealtimeCallsAcceptRequest(c *gin.Context, body []byte) (*OpenAIRealtimeCallsAcceptRequest, error) {
	return ParseOpenAIRealtimeRESTRequest(c, body)
}

func normalizeOpenAIRealtimeRESTEndpoint(path string) (endpoint string, callID string, action string, err error) {
	trimmed := strings.TrimRight(strings.TrimSpace(path), "/")
	if trimmed == "" {
		return "", "", "", fmt.Errorf("realtime endpoint is required")
	}
	if idx := strings.LastIndex(trimmed, "/realtime/"); idx >= 0 {
		trimmed = "/v1" + trimmed[idx:]
	}

	switch trimmed {
	case "/v1/realtime/client_secrets",
		"/v1/realtime/translations/client_secrets",
		"/v1/realtime/calls",
		"/v1/realtime/sessions",
		"/v1/realtime/transcription_sessions":
		return trimmed, "", "", nil
	}

	const callsPrefix = "/v1/realtime/calls/"
	if !strings.HasPrefix(trimmed, callsPrefix) {
		return "", "", "", fmt.Errorf("unsupported realtime endpoint")
	}
	rest := strings.TrimPrefix(trimmed, callsPrefix)
	parts := strings.Split(rest, "/")
	if len(parts) != 2 {
		return "", "", "", fmt.Errorf("unsupported realtime call endpoint")
	}
	callID = strings.TrimSpace(parts[0])
	action = strings.TrimSpace(parts[1])
	if callID == "" {
		return "", "", "", fmt.Errorf("call_id is required")
	}
	switch action {
	case "accept", "hangup", "refer", "reject":
		return trimmed, callID, action, nil
	default:
		return "", "", "", fmt.Errorf("unsupported realtime call action")
	}
}

func extractOpenAIRealtimeRESTModelRefs(endpoint string, body []byte) []OpenAIRealtimeRESTModelRef {
	var paths []string
	switch endpoint {
	case "/v1/realtime/client_secrets":
		paths = []string{"session.model", "session.audio.input.transcription.model"}
	case "/v1/realtime/translations/client_secrets":
		paths = []string{"session.model", "session.audio.input.transcription.model"}
	case "/v1/realtime/calls":
		paths = []string{"model", "session.model"}
	case "/v1/realtime/sessions":
		paths = []string{"model"}
	case "/v1/realtime/transcription_sessions":
		paths = []string{"input_audio_transcription.model"}
	default:
		return nil
	}

	refs := make([]OpenAIRealtimeRESTModelRef, 0, len(paths))
	for _, path := range paths {
		value := strings.TrimSpace(gjson.GetBytes(body, path).String())
		if value == "" {
			continue
		}
		refs = append(refs, OpenAIRealtimeRESTModelRef{Path: path, Value: value})
	}
	return refs
}

func OpenAIRealtimeCallsAcceptUpstreamEndpoint(callID string) string {
	callID = strings.Trim(strings.TrimSpace(callID), "/")
	if callID == "" {
		return openAIRealtimeCallsAcceptEndpointPrefix + openAIRealtimeCallsAcceptEndpointSuffix[1:]
	}
	return openAIRealtimeCallsAcceptEndpointPrefix + callID + openAIRealtimeCallsAcceptEndpointSuffix
}

func (s *OpenAIGatewayService) ForwardRealtimeREST(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIRealtimeRESTRequest,
	channelMappedModels map[string]string,
) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed realtime REST request is required")
	}
	if account == nil {
		return nil, fmt.Errorf("account is required")
	}
	if account.Platform != PlatformOpenAI || (account.Type != AccountTypeAPIKey && account.Type != AccountTypeOAuth) {
		return nil, fmt.Errorf("realtime REST endpoint requires an OpenAI API key or OAuth account")
	}

	startTime := time.Now()
	requestModel, upstreamModel, forwardBody := buildOpenAIRealtimeRESTForwardBody(parsed, account, channelMappedModels)

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}
	upstreamReq, err := s.buildOpenAIRealtimeRESTRequest(ctx, c, account, parsed.Endpoint, forwardBody, token)
	if err != nil {
		return nil, err
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			UpstreamURL:        safeUpstreamURL(upstreamReq.URL.String()),
			Kind:               "request_error",
			Message:            safeErr,
		})
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
		upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				UpstreamURL:        safeUpstreamURL(upstreamReq.URL.String()),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			s.handleFailoverSideEffects(ctx, resp, account)
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && isPoolModeRetryableStatus(resp.StatusCode),
			}
		}
		return s.handleErrorResponse(ctx, resp, c, account, forwardBody)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if contentType == "" {
		contentType = "application/json"
	}
	c.Data(resp.StatusCode, contentType, body)

	return &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Model:           requestModel,
		UpstreamModel:   upstreamModel,
		Stream:          false,
		ResponseHeaders: resp.Header.Clone(),
		Duration:        time.Since(startTime),
	}, nil
}

func (s *OpenAIGatewayService) ForwardRealtimeCallsAccept(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIRealtimeCallsAcceptRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed != nil {
		if parsed.Endpoint == "" {
			parsed.Endpoint = OpenAIRealtimeCallsAcceptUpstreamEndpoint(parsed.CallID)
		}
		if parsed.ScheduleModelPath == "" && parsed.Model != "" {
			parsed.ScheduleModelPath = "model"
		}
		if len(parsed.ModelRefs) == 0 && parsed.Model != "" {
			parsed.ModelRefs = []OpenAIRealtimeRESTModelRef{{Path: "model", Value: parsed.Model}}
		}
	}
	mapped := map[string]string{}
	if strings.TrimSpace(channelMappedModel) != "" {
		mapped["model"] = strings.TrimSpace(channelMappedModel)
	}
	return s.ForwardRealtimeREST(ctx, c, account, parsed, mapped)
}

func buildOpenAIRealtimeRESTForwardBody(
	parsed *OpenAIRealtimeRESTRequest,
	account *Account,
	channelMappedModels map[string]string,
) (requestModel string, upstreamModel string, body []byte) {
	if parsed == nil {
		return "", "", nil
	}
	body = parsed.Body
	requestModel = strings.TrimSpace(parsed.Model)
	upstreamModel = resolveOpenAIRealtimeRESTModel(account, requestModel, channelMappedModels[parsed.ScheduleModelPath])
	if requestModel == "" {
		upstreamModel = ""
	}

	for _, ref := range parsed.ModelRefs {
		original := strings.TrimSpace(ref.Value)
		if original == "" || strings.TrimSpace(ref.Path) == "" {
			continue
		}
		nextModel := resolveOpenAIRealtimeRESTModel(account, original, channelMappedModels[ref.Path])
		if nextModel == "" || nextModel == original {
			continue
		}
		updated, err := sjson.SetBytes(body, ref.Path, nextModel)
		if err != nil {
			continue
		}
		body = updated
	}
	if upstreamModel == "" && len(parsed.ModelRefs) > 0 {
		upstreamModel = strings.TrimSpace(parsed.ModelRefs[0].Value)
	}
	return requestModel, upstreamModel, body
}

func resolveOpenAIRealtimeRESTModel(account *Account, requestedModel string, channelMappedModel string) string {
	model := strings.TrimSpace(requestedModel)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		model = mapped
	}
	if account != nil {
		if mapped := strings.TrimSpace(account.GetMappedModel(model)); mapped != "" {
			model = mapped
		}
	}
	return model
}

func (s *OpenAIGatewayService) buildOpenAIRealtimeRESTRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	endpoint string,
	body []byte,
	token string,
) (*http.Request, error) {
	targetURL := buildOpenAIRealtimeRESTURL("", endpoint)
	if baseURL := account.GetOpenAIBaseURL(); baseURL != "" {
		validatedURL, err := s.validateUpstreamBaseURL(baseURL)
		if err != nil {
			return nil, err
		}
		targetURL = buildOpenAIRealtimeRESTURL(validatedURL, endpoint)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if c != nil && c.Request != nil {
		for key, values := range c.Request.Header {
			lowerKey := strings.ToLower(key)
			if !openaiPassthroughAllowedHeaders[lowerKey] {
				continue
			}
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
	}
	req.Header.Del("Authorization")
	req.Header.Del("X-Api-Key")
	req.Header.Del("X-Goog-Api-Key")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	if account.Type == AccountTypeOAuth {
		if chatgptAccountID := account.GetChatGPTAccountID(); chatgptAccountID != "" {
			req.Header.Set("chatgpt-account-id", chatgptAccountID)
		}
		isCodexOfficialClient := false
		if c != nil {
			isCodexOfficialClient = openai.IsCodexOfficialClientByHeaders(c.GetHeader("User-Agent"), c.GetHeader("originator"))
		}
		req.Header.Set("originator", resolveOpenAIUpstreamOriginator(c, isCodexOfficialClient))
	}
	if customUA := account.GetOpenAIUserAgent(); customUA != "" {
		req.Header.Set("User-Agent", customUA)
	} else if account.Type == AccountTypeOAuth {
		req.Header.Set("User-Agent", codexCLIUserAgent)
	}
	return req, nil
}

func buildOpenAIRealtimeRESTURL(base string, endpoint string) string {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		endpoint = "/v1/realtime"
	}
	if strings.TrimSpace(base) == "" {
		return buildOpenAIEndpointURL(openAIRealtimeRESTDefaultBaseURL, endpoint)
	}
	return buildOpenAIEndpointURL(base, endpoint)
}

func buildOpenAIRealtimeCallsAcceptURL(base string, callID string) string {
	return buildOpenAIRealtimeRESTURL(base, OpenAIRealtimeCallsAcceptUpstreamEndpoint(callID))
}
