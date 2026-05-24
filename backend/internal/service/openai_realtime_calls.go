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
)

const (
	openAIRealtimeCallsAcceptEndpointPrefix = "/v1/realtime/calls/"
	openAIRealtimeCallsAcceptEndpointSuffix = "/accept"
	openAIRealtimeCallsAcceptDefaultURL     = "https://api.openai.com/v1/realtime/calls"
)

type OpenAIRealtimeCallsAcceptRequest struct {
	CallID string
	Body   []byte
	Model  string
}

func (r *OpenAIRealtimeCallsAcceptRequest) StickySessionSeed() string {
	if r == nil {
		return ""
	}
	return strings.Join([]string{
		"openai-realtime-calls-accept",
		strings.TrimSpace(r.CallID),
		strings.TrimSpace(r.Model),
	}, "|")
}

func ParseOpenAIRealtimeCallsAcceptRequest(c *gin.Context, body []byte) (*OpenAIRealtimeCallsAcceptRequest, error) {
	if c == nil || c.Request == nil || c.Request.URL == nil {
		return nil, fmt.Errorf("missing request context")
	}
	callID := extractOpenAIRealtimeCallsAcceptCallID(c.Request.URL.Path)
	if callID == "" {
		return nil, fmt.Errorf("call_id is required")
	}
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
	return &OpenAIRealtimeCallsAcceptRequest{
		CallID: callID,
		Body:   body,
		Model:  model,
	}, nil
}

func extractOpenAIRealtimeCallsAcceptCallID(path string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(path), "/")
	idx := strings.LastIndex(trimmed, "/realtime/calls/")
	if idx < 0 || !strings.HasSuffix(trimmed, openAIRealtimeCallsAcceptEndpointSuffix) {
		return ""
	}
	start := idx + len("/realtime/calls/")
	end := len(trimmed) - len(openAIRealtimeCallsAcceptEndpointSuffix)
	if start >= end {
		return ""
	}
	return strings.TrimSpace(trimmed[start:end])
}

func OpenAIRealtimeCallsAcceptUpstreamEndpoint(callID string) string {
	callID = strings.Trim(strings.TrimSpace(callID), "/")
	if callID == "" {
		return openAIRealtimeCallsAcceptEndpointPrefix + openAIRealtimeCallsAcceptEndpointSuffix[1:]
	}
	return openAIRealtimeCallsAcceptEndpointPrefix + callID + openAIRealtimeCallsAcceptEndpointSuffix
}

func (s *OpenAIGatewayService) ForwardRealtimeCallsAccept(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIRealtimeCallsAcceptRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed realtime calls accept request is required")
	}
	if account == nil {
		return nil, fmt.Errorf("account is required")
	}
	if account.Platform != PlatformOpenAI || (account.Type != AccountTypeAPIKey && account.Type != AccountTypeOAuth) {
		return nil, fmt.Errorf("realtime calls accept endpoint requires an OpenAI API key or OAuth account")
	}

	startTime := time.Now()
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if strings.TrimSpace(upstreamModel) == "" {
		upstreamModel = requestModel
	}
	forwardBody := parsed.Body
	if upstreamModel != parsed.Model {
		forwardBody = ReplaceModelInBody(parsed.Body, upstreamModel)
	}

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}
	upstreamReq, err := s.buildOpenAIRealtimeCallsAcceptRequest(ctx, c, account, parsed.CallID, forwardBody, token)
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

	usage, _ := extractOpenAIUsageFromJSONBytes(body)
	return &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Usage:           usage,
		Model:           requestModel,
		UpstreamModel:   upstreamModel,
		Stream:          false,
		ResponseHeaders: resp.Header.Clone(),
		Duration:        time.Since(startTime),
	}, nil
}

func (s *OpenAIGatewayService) buildOpenAIRealtimeCallsAcceptRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	callID string,
	body []byte,
	token string,
) (*http.Request, error) {
	targetURL := buildOpenAIRealtimeCallsAcceptURL("", callID)
	if baseURL := account.GetOpenAIBaseURL(); baseURL != "" {
		validatedURL, err := s.validateUpstreamBaseURL(baseURL)
		if err != nil {
			return nil, err
		}
		targetURL = buildOpenAIRealtimeCallsAcceptURL(validatedURL, callID)
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

func buildOpenAIRealtimeCallsAcceptURL(base string, callID string) string {
	endpoint := OpenAIRealtimeCallsAcceptUpstreamEndpoint(callID)
	if strings.TrimSpace(base) == "" {
		return strings.TrimRight(openAIRealtimeCallsAcceptDefaultURL, "/") + "/" + strings.Trim(strings.TrimSpace(callID), "/") + openAIRealtimeCallsAcceptEndpointSuffix
	}
	return buildOpenAIEndpointURL(base, endpoint)
}
