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

func TestOpenAIRealtimeCallsAcceptUpstreamEndpoint(t *testing.T) {
	require.Equal(t, "/v1/realtime/calls/call_123/accept", OpenAIRealtimeCallsAcceptUpstreamEndpoint("call_123"))
	require.Equal(t, "https://example.test/v1/realtime/calls/call_123/accept", buildOpenAIRealtimeCallsAcceptURL("https://example.test/v1", "call_123"))
}
