package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
)

const (
	openAIAudioTranscriptionsEndpoint = "/v1/audio/transcriptions"
	openAITranscribeAliasEndpoint     = "/transcribe"
	openAIAudioTranscriptionsURL      = "https://api.openai.com/v1/audio/transcriptions"
	chatgptTranscribeURL              = "https://chatgpt.com/backend-api/transcribe"

	OpenAIAudioTranscriptionsDefaultModel         = "gpt-4o-mini-transcribe"
	openAIAudioTranscriptionsMaxFieldSize         = 1 << 20
	OpenAIAudioTranscriptionsRequiredAccountTypes = AccountTypeAPIKey + "," + AccountTypeOAuth
)

type OpenAIAudioTranscriptionsRequest struct {
	Endpoint        string
	ContentType     string
	Body            []byte
	Model           string
	ExplicitModel   bool
	Language        string
	FileName        string
	FileSizeBytes   int64
	FileContentType string
}

func (r *OpenAIAudioTranscriptionsRequest) IsTranscribeAlias() bool {
	return r != nil && r.Endpoint == openAITranscribeAliasEndpoint
}

func (r *OpenAIAudioTranscriptionsRequest) StickySessionSeed() string {
	if r == nil {
		return ""
	}
	return strings.Join([]string{
		"openai-audio-transcriptions",
		strings.TrimSpace(r.Endpoint),
		strings.TrimSpace(r.Model),
		strings.TrimSpace(r.Language),
		strings.TrimSpace(r.FileName),
	}, "|")
}

func (s *OpenAIGatewayService) ParseOpenAIAudioTranscriptionsRequest(c *gin.Context, body []byte) (*OpenAIAudioTranscriptionsRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	endpoint := normalizeOpenAIAudioTranscriptionsEndpointPath(c.Request.URL.Path)
	if endpoint == "" {
		return nil, fmt.Errorf("unsupported audio transcriptions endpoint")
	}

	contentType := strings.TrimSpace(c.GetHeader("Content-Type"))
	decodedBody := body
	if strings.EqualFold(strings.TrimSpace(c.GetHeader("X-Codex-Base64")), "1") {
		var err error
		decodedBody, err = decodeOpenAICodexBase64MultipartBody(body)
		if err != nil {
			return nil, err
		}
	}

	req := &OpenAIAudioTranscriptionsRequest{
		Endpoint:    endpoint,
		ContentType: contentType,
		Body:        decodedBody,
	}
	if err := parseOpenAIAudioTranscriptionsMultipartRequest(decodedBody, contentType, req); err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.Model) == "" {
		if req.IsTranscribeAlias() {
			req.Model = OpenAIAudioTranscriptionsDefaultModel
		} else {
			return nil, fmt.Errorf("model is required")
		}
	}
	req.Model = strings.TrimSpace(req.Model)
	return req, nil
}

func decodeOpenAICodexBase64MultipartBody(body []byte) ([]byte, error) {
	raw := strings.TrimSpace(string(body))
	if raw == "" {
		return nil, fmt.Errorf("base64 multipart body is empty")
	}
	decoded, err := base64.StdEncoding.DecodeString(raw)
	if err == nil {
		return decoded, nil
	}
	decoded, rawErr := base64.RawStdEncoding.DecodeString(raw)
	if rawErr == nil {
		return decoded, nil
	}
	return nil, fmt.Errorf("invalid base64 multipart body")
}

func parseOpenAIAudioTranscriptionsMultipartRequest(body []byte, contentType string, req *OpenAIAudioTranscriptionsRequest) error {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.EqualFold(mediaType, "multipart/form-data") {
		return fmt.Errorf("audio transcriptions endpoint requires multipart/form-data")
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return fmt.Errorf("multipart boundary is required")
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	hasFile := false
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read multipart body: %w", err)
		}

		name := strings.TrimSpace(part.FormName())
		if name == "" {
			_ = part.Close()
			continue
		}
		if name == "file" {
			hasFile = true
			req.FileName = strings.TrimSpace(part.FileName())
			req.FileContentType = strings.TrimSpace(part.Header.Get("Content-Type"))
			fileSize, copyErr := io.Copy(io.Discard, part)
			_ = part.Close()
			if copyErr != nil {
				return fmt.Errorf("read multipart file: %w", copyErr)
			}
			req.FileSizeBytes = fileSize
			continue
		}

		data, err := io.ReadAll(io.LimitReader(part, openAIAudioTranscriptionsMaxFieldSize))
		_ = part.Close()
		if err != nil {
			return fmt.Errorf("read multipart field %s: %w", name, err)
		}
		value := strings.TrimSpace(string(data))
		switch name {
		case "model":
			req.Model = value
			req.ExplicitModel = value != ""
		case "language":
			req.Language = value
		}
	}
	if !hasFile {
		return fmt.Errorf("file is required")
	}
	return nil
}

func normalizeOpenAIAudioTranscriptionsEndpointPath(path string) string {
	trimmed := strings.TrimSpace(path)
	switch {
	case strings.Contains(trimmed, "/audio/transcriptions"):
		return openAIAudioTranscriptionsEndpoint
	case trimmed == openAITranscribeAliasEndpoint:
		return openAITranscribeAliasEndpoint
	default:
		return ""
	}
}

func (s *OpenAIGatewayService) ForwardAudioTranscriptions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIAudioTranscriptionsRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed audio transcriptions request is required")
	}
	if account == nil {
		return nil, fmt.Errorf("account is required")
	}
	if account.Platform != PlatformOpenAI || (account.Type != AccountTypeAPIKey && account.Type != AccountTypeOAuth) {
		return nil, fmt.Errorf("audio transcriptions endpoint requires an OpenAI API key or OAuth account")
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
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI] Audio transcriptions request routing request_model=%s upstream_model=%s account_type=%s",
		strings.TrimSpace(parsed.Model),
		upstreamModel,
		account.Type,
	)

	forwardBody := parsed.Body
	forwardContentType := parsed.ContentType
	var err error
	if account.Type == AccountTypeAPIKey || parsed.ExplicitModel {
		forwardBody, forwardContentType, err = rewriteOpenAIAudioTranscriptionsMultipartModel(parsed.Body, parsed.ContentType, upstreamModel)
		if err != nil {
			return nil, err
		}
	}
	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}
	upstreamReq, err := s.buildOpenAIAudioTranscriptionsRequest(ctx, c, account, forwardBody, forwardContentType, token)
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
			if s.rateLimitService != nil {
				s.handleFailoverSideEffects(ctx, resp, account)
			}
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

	usage, ok := extractOpenAIUsageFromJSONBytes(body)
	if !ok {
		usage = estimateOpenAIAudioTranscriptionUsage(parsed)
	}
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

func estimateOpenAIAudioTranscriptionUsage(parsed *OpenAIAudioTranscriptionsRequest) OpenAIUsage {
	if parsed == nil || parsed.FileSizeBytes <= 0 {
		return OpenAIUsage{}
	}
	// Fallback for ChatGPT transcribe responses that return text without usage.
	// Estimate duration as 16 kHz 16-bit mono PCM, then use 50 audio tokens/sec.
	const bytesPerSecond = 16000 * 2
	const audioTokensPerSecond = 50
	seconds := math.Ceil(float64(parsed.FileSizeBytes) / float64(bytesPerSecond))
	if seconds < 1 {
		seconds = 1
	}
	return OpenAIUsage{InputAudioTokens: int(seconds * audioTokensPerSecond)}
}

func (s *OpenAIGatewayService) buildOpenAIAudioTranscriptionsRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	contentType string,
	token string,
) (*http.Request, error) {
	targetURL := openAIAudioTranscriptionsURL
	if account.Type == AccountTypeOAuth {
		targetURL = chatgptTranscribeURL
	} else if baseURL := account.GetOpenAIBaseURL(); baseURL != "" {
		validatedURL, err := s.validateUpstreamBaseURL(baseURL)
		if err != nil {
			return nil, err
		}
		targetURL = buildOpenAIAudioTranscriptionsURL(validatedURL)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if c != nil && c.Request != nil {
		for key, values := range c.Request.Header {
			lowerKey := strings.ToLower(key)
			if lowerKey == "x-codex-base64" || !openaiPassthroughAllowedHeaders[lowerKey] {
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
	if account.Type == AccountTypeOAuth {
		req.Host = "chatgpt.com"
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
	if strings.TrimSpace(contentType) != "" {
		req.Header.Set("Content-Type", contentType)
	}
	return req, nil
}

func buildOpenAIAudioTranscriptionsURL(base string) string {
	return buildOpenAIEndpointURL(base, openAIAudioTranscriptionsEndpoint)
}

func rewriteOpenAIAudioTranscriptionsMultipartModel(body []byte, contentType string, model string) ([]byte, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return body, contentType, nil
	}
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, "", fmt.Errorf("parse multipart content-type: %w", err)
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return nil, "", fmt.Errorf("multipart boundary is required")
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	modelWritten := false

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", fmt.Errorf("read multipart body: %w", err)
		}

		formName := strings.TrimSpace(part.FormName())
		partHeader := cloneOpenAIAudioMultipartHeader(part.Header)
		target, err := writer.CreatePart(partHeader)
		if err != nil {
			_ = part.Close()
			return nil, "", fmt.Errorf("create multipart part: %w", err)
		}
		if formName == "model" && part.FileName() == "" {
			if _, err := target.Write([]byte(model)); err != nil {
				_ = part.Close()
				return nil, "", fmt.Errorf("rewrite multipart model: %w", err)
			}
			modelWritten = true
			_ = part.Close()
			continue
		}
		if _, err := io.Copy(target, part); err != nil {
			_ = part.Close()
			return nil, "", fmt.Errorf("copy multipart part: %w", err)
		}
		_ = part.Close()
	}

	if !modelWritten {
		if err := writer.WriteField("model", model); err != nil {
			return nil, "", fmt.Errorf("append multipart model field: %w", err)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("finalize multipart body: %w", err)
	}
	return buffer.Bytes(), writer.FormDataContentType(), nil
}

func cloneOpenAIAudioMultipartHeader(src textproto.MIMEHeader) textproto.MIMEHeader {
	dst := make(textproto.MIMEHeader, len(src))
	for key, values := range src {
		copied := make([]string, len(values))
		copy(copied, values)
		dst[key] = copied
	}
	return dst
}
