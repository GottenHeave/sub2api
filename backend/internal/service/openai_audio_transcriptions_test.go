package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func buildOpenAIAudioTranscriptionMultipart(t *testing.T, fields map[string]string, fileBytes []byte) ([]byte, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = part.Write(fileBytes)
	require.NoError(t, err)
	for key, value := range fields {
		require.NoError(t, writer.WriteField(key, value))
	}
	require.NoError(t, writer.Close())
	return body.Bytes(), writer.FormDataContentType()
}

func newOpenAIAudioTranscriptionTestContext(method, path string, body []byte, contentType string) (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	c.Request = req
	return c, rec
}

func parseMultipartFieldsForTest(t *testing.T, body []byte, contentType string) map[string]string {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(contentType)
	require.NoError(t, err)
	require.Equal(t, "multipart/form-data", mediaType)
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	fields := map[string]string{}
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		data, err := io.ReadAll(part)
		require.NoError(t, err)
		if part.FileName() != "" {
			fields[part.FormName()] = string(data)
		} else {
			fields[part.FormName()] = strings.TrimSpace(string(data))
		}
		require.NoError(t, part.Close())
	}
	return fields
}

func TestOpenAIGatewayServiceParseAudioTranscriptions_StandardMultipartRequiresModel(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"model":           "client-transcribe",
		"language":        "en",
		"prompt":          "domain terms",
		"response_format": "json",
		"temperature":     "0.2",
	}, []byte("fake-audio"))
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)

	parsed, err := (&OpenAIGatewayService{}).ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.NoError(t, err)
	require.Equal(t, openAIAudioTranscriptionsEndpoint, parsed.Endpoint)
	require.Equal(t, "client-transcribe", parsed.Model)
	require.True(t, parsed.ExplicitModel)
	require.Equal(t, "en", parsed.Language)
	require.Equal(t, "sample.wav", parsed.FileName)

	noModelBody, noModelContentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"language": "en",
	}, []byte("fake-audio"))
	c, _ = newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", noModelBody, noModelContentType)
	parsed, err = (&OpenAIGatewayService{}).ParseOpenAIAudioTranscriptionsRequest(c, noModelBody)
	require.Nil(t, parsed)
	require.ErrorContains(t, err, "model is required")
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_APIKeyUsesMappedModelAndCustomBaseURL(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"model":           "client-transcribe",
		"language":        "en",
		"prompt":          "domain terms",
		"response_format": "json",
		"temperature":     "0.2",
	}, []byte("fake-audio"))
	c, rec := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)
	c.Request.Header.Set("Accept", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
			"x-request-id": []string{"rid_audio"},
		},
		Body: io.NopCloser(strings.NewReader(`{"text":"hello"}`)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          7,
		Name:        "api-key",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "sk-test",
			"base_url": "https://audio-upstream.example/v1",
			"model_mapping": map[string]any{
				"client-transcribe": "upstream-transcribe",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}

	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.NoError(t, err)
	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "https://audio-upstream.example/v1/audio/transcriptions", upstream.lastReq.URL.String())
	require.Equal(t, "/v1/audio/transcriptions", upstream.lastReq.URL.Path)
	require.Equal(t, "Bearer sk-test", upstream.lastReq.Header.Get("Authorization"))

	fields := parseMultipartFieldsForTest(t, upstream.lastBody, upstream.lastReq.Header.Get("Content-Type"))
	require.Equal(t, "fake-audio", fields["file"])
	require.Equal(t, "en", fields["language"])
	require.Equal(t, "domain terms", fields["prompt"])
	require.Equal(t, "json", fields["response_format"])
	require.Equal(t, "0.2", fields["temperature"])
	require.Equal(t, "upstream-transcribe", fields["model"])
	require.Equal(t, "client-transcribe", result.Model)
	require.Equal(t, "upstream-transcribe", result.UpstreamModel)
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_UsagePolicy(t *testing.T) {
	tests := []struct {
		name          string
		path          string
		fields        map[string]string
		responseBody  string
		expectedUsage OpenAIUsage
	}{
		{
			name:         "standard endpoint no usage records explicit zero tokens",
			path:         "/v1/audio/transcriptions",
			fields:       map[string]string{"model": "gpt-4o-mini-transcribe"},
			responseBody: `{"text":"hello"}`,
		},
		{
			name:         "transcribe alias no usage records explicit zero tokens",
			path:         "/transcribe",
			fields:       map[string]string{"language": "en"},
			responseBody: `{"text":"hello"}`,
		},
		{
			name:         "standard endpoint top-level usage",
			path:         "/v1/audio/transcriptions",
			fields:       map[string]string{"model": "gpt-4o-mini-transcribe"},
			responseBody: `{"text":"hello","usage":{"input_tokens":11,"output_tokens":3,"input_token_details":{"audio_tokens":7,"cached_tokens":2,"cached_tokens_details":{"audio_tokens":1}},"output_token_details":{"audio_tokens":2}}}`,
			expectedUsage: OpenAIUsage{
				InputTokens:          11,
				OutputTokens:         3,
				CacheReadInputTokens: 2,
				InputAudioTokens:     7,
				OutputAudioTokens:    2,
				CacheReadAudioTokens: 1,
			},
		},
		{
			name:         "standard endpoint response usage",
			path:         "/v1/audio/transcriptions",
			fields:       map[string]string{"model": "gpt-4o-mini-transcribe"},
			responseBody: `{"text":"hello","response":{"usage":{"input_tokens":12,"output_tokens":4,"input_tokens_details":{"audio_tokens":6,"cached_tokens":3,"cached_tokens_details":{"audio_tokens":2}},"output_tokens_details":{"audio_tokens":1}}}}`,
			expectedUsage: OpenAIUsage{
				InputTokens:          12,
				OutputTokens:         4,
				CacheReadInputTokens: 3,
				InputAudioTokens:     6,
				OutputAudioTokens:    1,
				CacheReadAudioTokens: 2,
			},
		},
		{
			name:         "transcribe alias top-level usage",
			path:         "/transcribe",
			fields:       map[string]string{"language": "en"},
			responseBody: `{"text":"hello","usage":{"input_tokens":10,"output_tokens":2,"input_token_details":{"audio_tokens":5,"cached_tokens":1,"cached_tokens_details":{"audio_tokens":1}},"output_token_details":{"audio_tokens":2}}}`,
			expectedUsage: OpenAIUsage{
				InputTokens:          10,
				OutputTokens:         2,
				CacheReadInputTokens: 1,
				InputAudioTokens:     5,
				OutputAudioTokens:    2,
				CacheReadAudioTokens: 1,
			},
		},
		{
			name:         "transcribe alias response usage",
			path:         "/transcribe",
			fields:       map[string]string{"language": "en"},
			responseBody: `{"text":"hello","response":{"usage":{"prompt_tokens":13,"completion_tokens":5,"prompt_tokens_details":{"audio_tokens":8,"cached_tokens":4,"cached_tokens_details":{"audio_tokens":3}},"completion_tokens_details":{"audio_tokens":2}}}}`,
			expectedUsage: OpenAIUsage{
				InputTokens:          13,
				OutputTokens:         5,
				CacheReadInputTokens: 4,
				InputAudioTokens:     8,
				OutputAudioTokens:    2,
				CacheReadAudioTokens: 3,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, contentType := buildOpenAIAudioTranscriptionMultipart(t, tt.fields, []byte("fake-audio"))
			c, rec := newOpenAIAudioTranscriptionTestContext(http.MethodPost, tt.path, body, contentType)

			upstream := &httpUpstreamRecorder{resp: newOpenAIAudioTranscriptionResponse(http.StatusOK, tt.responseBody)}
			svc := &OpenAIGatewayService{
				cfg:          &config.Config{},
				httpUpstream: upstream,
			}

			parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
			require.NoError(t, err)
			result, err := svc.ForwardAudioTranscriptions(context.Background(), c, openAIAudioTranscriptionAPIKeyAccount(7), parsed, "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, http.StatusOK, rec.Code)
			require.Equal(t, tt.expectedUsage, result.Usage)
		})
	}
}

func TestOpenAIGatewayServiceParseAndForwardTranscribeAlias_DefaultModelAndBase64(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"language": "zh",
	}, []byte("codex-audio"))
	plainC, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/transcribe", body, contentType)
	plainParsed, err := (&OpenAIGatewayService{}).ParseOpenAIAudioTranscriptionsRequest(plainC, body)
	require.NoError(t, err)
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, plainParsed.Model)
	require.Equal(t, "zh", plainParsed.Language)

	encoded := []byte(base64.StdEncoding.EncodeToString(body))
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/transcribe", encoded, contentType)
	c.Request.Header.Set("X-Codex-Base64", "1")
	c.Request.Header.Set("User-Agent", "codex-desktop-test")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"text":"你好"}`)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          8,
		Name:        "api-key",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{"api_key": "sk-test"},
		Status:      StatusActive,
		Schedulable: true,
	}

	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, encoded)
	require.NoError(t, err)
	require.Equal(t, openAITranscribeAliasEndpoint, parsed.Endpoint)
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, parsed.Model)
	require.False(t, parsed.ExplicitModel)
	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)

	require.Equal(t, "https://api.openai.com/v1/audio/transcriptions", upstream.lastReq.URL.String())
	require.Empty(t, upstream.lastReq.Header.Get("X-Codex-Base64"))
	fields := parseMultipartFieldsForTest(t, upstream.lastBody, upstream.lastReq.Header.Get("Content-Type"))
	require.Equal(t, "codex-audio", fields["file"])
	require.Equal(t, "zh", fields["language"])
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, fields["model"])
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_OAuthUsesChatGPTTranscribe(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"language": "en",
		"prompt":   "terms",
	}, []byte("codex-audio"))
	encoded := []byte(base64.StdEncoding.EncodeToString(body))
	c, rec := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/transcribe", encoded, contentType)
	c.Request.Header.Set("X-Codex-Base64", "1")
	c.Request.Header.Set("User-Agent", "Codex Desktop")
	c.Request.Header.Set("Authorization", "Bearer client-token")
	c.Request.Header.Set("chatgpt-account-id", "client-account")
	c.Request.Header.Set("originator", "client-originator")
	c.Request.Header.Set("X-Api-Key", "client-api-key")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_oauth_audio"}},
		Body:       io.NopCloser(strings.NewReader(`{"text":"hello"}`)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          9,
		Name:        "oauth",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-account",
		},
		Status:      StatusActive,
		Schedulable: true,
	}

	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, encoded)
	require.NoError(t, err)
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, parsed.Model)
	require.False(t, parsed.ExplicitModel)

	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, chatgptTranscribeURL, upstream.lastReq.URL.String())
	require.Equal(t, "chatgpt.com", upstream.lastReq.Host)
	require.Equal(t, "Bearer oauth-token", upstream.lastReq.Header.Get("Authorization"))
	require.Len(t, upstream.lastReq.Header.Values("Authorization"), 1)
	require.Equal(t, "chatgpt-account", upstream.lastReq.Header.Get("chatgpt-account-id"))
	require.Len(t, upstream.lastReq.Header.Values("chatgpt-account-id"), 1)
	require.Equal(t, "client-originator", upstream.lastReq.Header.Get("originator"))
	require.Len(t, upstream.lastReq.Header.Values("originator"), 1)
	require.Equal(t, codexCLIUserAgent, upstream.lastReq.Header.Get("User-Agent"))
	require.Len(t, upstream.lastReq.Header.Values("User-Agent"), 1)
	require.Empty(t, upstream.lastReq.Header.Get("X-Codex-Base64"))
	require.Empty(t, upstream.lastReq.Header.Get("X-Api-Key"))

	fields := parseMultipartFieldsForTest(t, upstream.lastBody, upstream.lastReq.Header.Get("Content-Type"))
	require.Equal(t, "codex-audio", fields["file"])
	require.Equal(t, "en", fields["language"])
	require.Equal(t, "terms", fields["prompt"])
	require.Empty(t, fields["model"])
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, result.Model)
	require.Equal(t, OpenAIAudioTranscriptionsDefaultModel, result.UpstreamModel)
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_OAuthPreservesExplicitMappedModel(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"model":    "client-transcribe",
		"language": "en",
	}, []byte("codex-audio"))
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"text":"hello"}`)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          11,
		Name:        "oauth",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token": "oauth-token",
			"model_mapping": map[string]any{
				"client-transcribe": "upstream-transcribe",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}

	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.NoError(t, err)
	require.True(t, parsed.ExplicitModel)
	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)

	fields := parseMultipartFieldsForTest(t, upstream.lastBody, upstream.lastReq.Header.Get("Content-Type"))
	require.Equal(t, "upstream-transcribe", fields["model"])
	require.Equal(t, "client-transcribe", result.Model)
	require.Equal(t, "upstream-transcribe", result.UpstreamModel)
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_FailoverStatuses(t *testing.T) {
	tests := []struct {
		name                 string
		status               int
		poolMode             bool
		retryableSameAccount bool
	}{
		{name: "401 retries same pool account", status: http.StatusUnauthorized, poolMode: true, retryableSameAccount: true},
		{name: "403 retries same pool account", status: http.StatusForbidden, poolMode: true, retryableSameAccount: true},
		{name: "429 retries same pool account", status: http.StatusTooManyRequests, poolMode: true, retryableSameAccount: true},
		{name: "500 switches account without same-account retry", status: http.StatusInternalServerError, poolMode: true},
		{name: "503 switches account without same-account retry", status: http.StatusServiceUnavailable, poolMode: true},
		{name: "401 non-pool switches account", status: http.StatusUnauthorized},
		{name: "403 non-pool switches account", status: http.StatusForbidden},
		{name: "429 non-pool switches account", status: http.StatusTooManyRequests},
		{name: "500 non-pool switches account", status: http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
				"model": "gpt-4o-mini-transcribe",
			}, []byte("fake-audio"))
			c, rec := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)
			upstream := &httpUpstreamRecorder{resp: newOpenAIAudioTranscriptionResponse(tt.status, `{"error":{"message":"temporary upstream failure"}}`)}
			svc := &OpenAIGatewayService{
				cfg:          &config.Config{},
				httpUpstream: upstream,
			}
			account := openAIAudioTranscriptionAPIKeyAccount(7)
			if tt.poolMode {
				account.Credentials["pool_mode"] = true
			}

			parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
			require.NoError(t, err)
			result, err := svc.ForwardAudioTranscriptions(context.Background(), c, account, parsed, "")
			require.Nil(t, result)

			var failoverErr *UpstreamFailoverError
			require.ErrorAs(t, err, &failoverErr)
			require.Equal(t, tt.status, failoverErr.StatusCode)
			require.Equal(t, tt.retryableSameAccount, failoverErr.RetryableOnSameAccount)
			require.False(t, rec.Result().Header.Get("Content-Type") == "application/json" && rec.Body.Len() > 0, "failover response should not be written before handler retry logic")
		})
	}
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_RequestErrorIsNotFailover(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"model": "gpt-4o-mini-transcribe",
	}, []byte("fake-audio"))
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: &httpUpstreamRecorder{err: errors.New("dial failed")},
	}

	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.NoError(t, err)
	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, openAIAudioTranscriptionAPIKeyAccount(7), parsed, "")

	require.Nil(t, result)
	require.Error(t, err)
	var failoverErr *UpstreamFailoverError
	require.False(t, errors.As(err, &failoverErr))
}

func TestOpenAIGatewayServiceForwardAudioTranscriptions_RejectsUnsupportedAccountType(t *testing.T) {
	body, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"model": "client-transcribe",
	}, []byte("fake-audio"))
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/v1/audio/transcriptions", body, contentType)
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: &httpUpstreamRecorder{}}
	parsed, err := svc.ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.NoError(t, err)

	result, err := svc.ForwardAudioTranscriptions(context.Background(), c, &Account{
		ID:          10,
		Platform:    PlatformOpenAI,
		Type:        AccountTypeSetupToken,
		Credentials: map[string]any{"access_token": "setup-token"},
	}, parsed, "")

	require.Nil(t, result)
	require.ErrorContains(t, err, "OpenAI API key or OAuth account")
}

func TestOpenAIGatewayServiceParseTranscribeAlias_InvalidBase64(t *testing.T) {
	_, contentType := buildOpenAIAudioTranscriptionMultipart(t, map[string]string{
		"language": "en",
	}, []byte("fake-audio"))
	body := []byte("not base64%")
	c, _ := newOpenAIAudioTranscriptionTestContext(http.MethodPost, "/transcribe", body, contentType)
	c.Request.Header.Set("X-Codex-Base64", "1")

	parsed, err := (&OpenAIGatewayService{}).ParseOpenAIAudioTranscriptionsRequest(c, body)
	require.Nil(t, parsed)
	require.ErrorContains(t, err, "invalid base64 multipart body")
}

func newOpenAIAudioTranscriptionResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
			"x-request-id": []string{"rid_audio"},
		},
		Body: io.NopCloser(strings.NewReader(body)),
	}
}

func openAIAudioTranscriptionAPIKeyAccount(id int64) *Account {
	return &Account{
		ID:          id,
		Name:        "api-key",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
		},
		Status:      StatusActive,
		Schedulable: true,
	}
}
