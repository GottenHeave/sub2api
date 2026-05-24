package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestParseOpenAIRealtimeCallsAcceptRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/accept", bytes.NewReader(nil))

	parsed, err := ParseOpenAIRealtimeCallsAcceptRequest(c, []byte(`{"type":"realtime","model":"gpt-realtime"}`))

	require.NoError(t, err)
	require.Equal(t, "call_123", parsed.CallID)
	require.Equal(t, "gpt-realtime", parsed.Model)
}

func TestParseOpenAIRealtimeRESTRequest_ClientSecretsTranscriptionModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/client_secrets", bytes.NewReader(nil))

	parsed, err := ParseOpenAIRealtimeRESTRequest(c, []byte(`{"session":{"type":"transcription","audio":{"input":{"transcription":{"model":"gpt-4o-transcribe"}}}}}`))

	require.NoError(t, err)
	require.Equal(t, "/v1/realtime/client_secrets", parsed.Endpoint)
	require.Equal(t, "gpt-4o-transcribe", parsed.Model)
	require.Equal(t, "session.audio.input.transcription.model", parsed.ScheduleModelPath)
	require.Equal(t, []OpenAIRealtimeRESTModelRef{
		{Path: "session.audio.input.transcription.model", Value: "gpt-4o-transcribe"},
	}, parsed.ModelRefs)
}

func TestParseOpenAIRealtimeRESTRequest_PrefixedPath(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/openai/v1/realtime/calls/call_123/reject", bytes.NewReader(nil))

	parsed, err := ParseOpenAIRealtimeRESTRequest(c, []byte(`{"status_code":603}`))

	require.NoError(t, err)
	require.Equal(t, "/v1/realtime/calls/call_123/reject", parsed.Endpoint)
	require.Equal(t, "call_123", parsed.CallID)
	require.Equal(t, "reject", parsed.Action)
}

func TestOpenAIGatewayService_ForwardRealtimeCallsAccept_APIKeyURLBodyAndHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"type":"realtime","model":"gpt-realtime","instructions":"answer calls"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/accept", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_call"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"id":"call_123"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed := &OpenAIRealtimeCallsAcceptRequest{CallID: "call_123", Body: body, Model: "gpt-realtime"}

	result, err := svc.ForwardRealtimeCallsAccept(context.Background(), c, account, parsed, "")

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, http.MethodPost, upstream.lastReq.Method)
	require.Equal(t, "https://api.openai.com/v1/realtime/calls/call_123/accept", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer sk-test", upstream.lastReq.Header.Get("Authorization"))
	require.JSONEq(t, string(body), string(upstream.lastBody))
	require.JSONEq(t, `{"id":"call_123"}`, rec.Body.String())
}

func TestOpenAIGatewayService_ForwardRealtimeCallsAccept_OAuthURLBodyAndHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"type":"realtime","model":"client-realtime"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/accept", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("originator", "codex_cli_rs")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_call_oauth"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"id":"call_123"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-acc",
			"model_mapping": map[string]any{
				"client-realtime": "gpt-realtime",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed := &OpenAIRealtimeCallsAcceptRequest{CallID: "call_123", Body: body, Model: "client-realtime"}

	result, err := svc.ForwardRealtimeCallsAccept(context.Background(), c, account, parsed, "")

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, "https://api.openai.com/v1/realtime/calls/call_123/accept", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer oauth-token", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "chatgpt-acc", upstream.lastReq.Header.Get("chatgpt-account-id"))
	require.Equal(t, "codex_cli_rs", upstream.lastReq.Header.Get("originator"))
	require.Equal(t, "gpt-realtime", gjson.GetBytes(upstream.lastBody, "model").String())
	require.JSONEq(t, `{"id":"call_123"}`, rec.Body.String())
}

func TestOpenAIGatewayService_ForwardRealtimeREST_ClientSecretMapsTranscriptionModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"session":{"type":"transcription","audio":{"input":{"transcription":{"model":"client-transcribe"}}}}}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/client_secrets", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_secret"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"value":"secret"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
			"model_mapping": map[string]any{
				"gpt-4o-mini-transcribe": "gpt-4o-transcribe",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed, err := ParseOpenAIRealtimeRESTRequest(c, body)
	require.NoError(t, err)

	result, err := svc.ForwardRealtimeREST(
		context.Background(),
		c,
		account,
		parsed,
		map[string]string{"session.audio.input.transcription.model": "gpt-4o-mini-transcribe"},
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://api.openai.com/v1/realtime/client_secrets", upstream.lastReq.URL.String())
	require.Equal(t, "gpt-4o-transcribe", gjson.GetBytes(upstream.lastBody, "session.audio.input.transcription.model").String())
	require.Equal(t, "client-transcribe", result.Model)
	require.Equal(t, "gpt-4o-transcribe", result.UpstreamModel)
}

func TestOpenAIGatewayService_ForwardRealtimeREST_TranslationMapsSessionAndTranscriptionModels(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"session":{"model":"client-translate","audio":{"input":{"transcription":{"model":"client-transcribe"}}}}}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/translations/client_secrets", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_translation"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"value":"secret"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
			"model_mapping": map[string]any{
				"mapped-translate":  "gpt-realtime-translate",
				"mapped-transcribe": "gpt-4o-transcribe",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed, err := ParseOpenAIRealtimeRESTRequest(c, body)
	require.NoError(t, err)

	result, err := svc.ForwardRealtimeREST(
		context.Background(),
		c,
		account,
		parsed,
		map[string]string{
			"session.model": "mapped-translate",
			"session.audio.input.transcription.model": "mapped-transcribe",
		},
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://api.openai.com/v1/realtime/translations/client_secrets", upstream.lastReq.URL.String())
	require.Equal(t, "gpt-realtime-translate", gjson.GetBytes(upstream.lastBody, "session.model").String())
	require.Equal(t, "gpt-4o-transcribe", gjson.GetBytes(upstream.lastBody, "session.audio.input.transcription.model").String())
	require.Equal(t, "client-translate", result.Model)
	require.Equal(t, "gpt-realtime-translate", result.UpstreamModel)
}

func TestOpenAIGatewayService_ForwardRealtimeREST_TranslationCallsMapsModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"model":"client-translate-call","sdp":"v=0"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/translations/calls", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/sdp")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_translation_call"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"id":"call_123"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-acc",
			"model_mapping": map[string]any{
				"mapped-translate-call": "gpt-realtime-translate",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed, err := ParseOpenAIRealtimeRESTRequest(c, body)
	require.NoError(t, err)
	require.Equal(t, "/v1/realtime/translations/calls", parsed.Endpoint)
	require.Equal(t, "client-translate-call", parsed.Model)
	require.Equal(t, "model", parsed.ScheduleModelPath)

	result, err := svc.ForwardRealtimeREST(
		context.Background(),
		c,
		account,
		parsed,
		map[string]string{"model": "mapped-translate-call"},
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://api.openai.com/v1/realtime/translations/calls", upstream.lastReq.URL.String())
	require.Equal(t, "application/sdp", upstream.lastReq.Header.Get("Content-Type"))
	require.Equal(t, "Bearer oauth-token", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "chatgpt-acc", upstream.lastReq.Header.Get("chatgpt-account-id"))
	require.Equal(t, "gpt-realtime-translate", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "client-translate-call", result.Model)
	require.Equal(t, "gpt-realtime-translate", result.UpstreamModel)
}

func TestOpenAIGatewayService_ForwardRealtimeREST_TranslationOAuthHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"session":{"model":"client-translate"}}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/translations/client_secrets", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("originator", "codex_cli_rs")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_translation_oauth"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"value":"secret"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-acc",
			"model_mapping": map[string]any{
				"client-translate": "gpt-realtime-translate",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed, err := ParseOpenAIRealtimeRESTRequest(c, body)
	require.NoError(t, err)

	result, err := svc.ForwardRealtimeREST(context.Background(), c, account, parsed, nil)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://api.openai.com/v1/realtime/translations/client_secrets", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer oauth-token", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "chatgpt-acc", upstream.lastReq.Header.Get("chatgpt-account-id"))
	require.Equal(t, "codex_cli_rs", upstream.lastReq.Header.Get("originator"))
	require.Equal(t, "gpt-realtime-translate", gjson.GetBytes(upstream.lastBody, "session.model").String())
	require.Equal(t, "client-translate", result.Model)
	require.Equal(t, "gpt-realtime-translate", result.UpstreamModel)
}

func TestOpenAIGatewayService_ForwardRealtimeREST_CallControlAllowsNoModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"target_uri":"tel:+15551234567"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/refer", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_refer"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"id":"call_123"}`))),
	}}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          123,
		Name:        "acc",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key": "sk-test",
		},
		Status:      StatusActive,
		Schedulable: true,
	}
	parsed, err := ParseOpenAIRealtimeRESTRequest(c, body)
	require.NoError(t, err)

	result, err := svc.ForwardRealtimeREST(context.Background(), c, account, parsed, nil)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://api.openai.com/v1/realtime/calls/call_123/refer", upstream.lastReq.URL.String())
	require.JSONEq(t, string(body), string(upstream.lastBody))
	require.Empty(t, result.Model)
	require.Empty(t, result.UpstreamModel)
}

func TestOpenAIRealtimeCallsAcceptUpstreamEndpoint(t *testing.T) {
	require.Equal(t, "/v1/realtime/calls/call_123/accept", OpenAIRealtimeCallsAcceptUpstreamEndpoint("call_123"))
	require.Equal(t, "https://example.test/v1/realtime/calls/call_123/accept", buildOpenAIRealtimeCallsAcceptURL("https://example.test/v1", "call_123"))
}
