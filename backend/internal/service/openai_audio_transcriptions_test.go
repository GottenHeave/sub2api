package service

import (
	"bytes"
	"context"
	"encoding/base64"
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
	require.Equal(t, "chatgpt-account", upstream.lastReq.Header.Get("chatgpt-account-id"))
	require.NotEmpty(t, upstream.lastReq.Header.Get("originator"))
	require.Equal(t, codexCLIUserAgent, upstream.lastReq.Header.Get("User-Agent"))
	require.Empty(t, upstream.lastReq.Header.Get("X-Codex-Base64"))

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
