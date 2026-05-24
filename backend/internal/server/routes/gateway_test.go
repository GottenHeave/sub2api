package routes

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/handler"
	servermiddleware "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func newGatewayRoutesTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	RegisterGatewayRoutes(
		router,
		&handler.Handlers{
			Gateway:       &handler.GatewayHandler{},
			OpenAIGateway: &handler.OpenAIGatewayHandler{},
		},
		servermiddleware.APIKeyAuthMiddleware(func(c *gin.Context) {
			groupID := int64(1)
			c.Set(string(servermiddleware.ContextKeyAPIKey), &service.APIKey{
				GroupID: &groupID,
				Group:   &service.Group{Platform: service.PlatformOpenAI},
			})
			c.Next()
		}),
		nil,
		nil,
		nil,
		nil,
		&config.Config{},
	)

	return router
}

func newGatewayRoutesTestRouterForPlatform(platform string) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	RegisterGatewayRoutes(
		router,
		&handler.Handlers{
			Gateway:       &handler.GatewayHandler{},
			OpenAIGateway: &handler.OpenAIGatewayHandler{},
		},
		servermiddleware.APIKeyAuthMiddleware(func(c *gin.Context) {
			groupID := int64(1)
			c.Set(string(servermiddleware.ContextKeyAPIKey), &service.APIKey{
				GroupID: &groupID,
				Group:   &service.Group{Platform: platform},
			})
			c.Next()
		}),
		nil,
		nil,
		nil,
		nil,
		&config.Config{},
	)

	return router
}

func TestGatewayRoutesOpenAIResponsesCompactPathIsRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	for _, path := range []string{
		"/v1/responses/compact",
		"/responses/compact",
		"/backend-api/codex/responses",
		"/backend-api/codex/responses/compact",
	} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"gpt-5"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)
		require.NotEqual(t, http.StatusNotFound, w.Code, "path=%s should hit OpenAI responses handler", path)
	}
}

func TestGatewayRoutesOpenAIImagesPathsAreRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	for _, path := range []string{
		"/v1/images/generations",
		"/v1/images/edits",
		"/images/generations",
		"/images/edits",
	} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"gpt-image-2","prompt":"draw a cat"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)
		require.NotEqual(t, http.StatusNotFound, w.Code, "path=%s should hit OpenAI images handler", path)
	}
}

func TestGatewayRoutesOpenAIRealtimePathIsRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	req := httptest.NewRequest(http.MethodGet, "/v1/realtime?model=gpt-realtime", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)
	require.NotEqual(t, http.StatusNotFound, w.Code)
	require.Equal(t, http.StatusUpgradeRequired, w.Code)
}

func TestGatewayRoutesOpenAIRealtimeRejectsNonOpenAIPlatform(t *testing.T) {
	router := newGatewayRoutesTestRouterForPlatform(service.PlatformAnthropic)

	req := httptest.NewRequest(http.MethodGet, "/v1/realtime?model=gpt-realtime", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)
	require.Equal(t, http.StatusNotFound, w.Code)
	require.Contains(t, w.Body.String(), "Realtime API is not supported for this platform")
}

func TestGatewayRoutesOpenAIRealtimeCallsAcceptPathIsRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	req := httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/accept", strings.NewReader(`{"type":"realtime","model":"gpt-realtime"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)
	require.NotEqual(t, http.StatusNotFound, w.Code)
}

func TestGatewayRoutesOpenAIRealtimeRESTPathsAreRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	tests := []struct {
		path string
		body string
	}{
		{"/v1/realtime/client_secrets", `{"session":{"type":"realtime","model":"gpt-realtime"}}`},
		{"/v1/realtime/translations/client_secrets", `{"session":{"model":"gpt-realtime-translate"}}`},
		{"/v1/realtime/calls", `{"model":"gpt-realtime"}`},
		{"/v1/realtime/calls/call_123/hangup", `{}`},
		{"/v1/realtime/calls/call_123/refer", `{"target_uri":"tel:+15551234567"}`},
		{"/v1/realtime/calls/call_123/reject", `{"status_code":603}`},
		{"/v1/realtime/sessions", `{"model":"gpt-realtime"}`},
		{"/v1/realtime/transcription_sessions", `{"input_audio_transcription":{"model":"gpt-4o-transcribe"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)
			require.NotEqual(t, http.StatusNotFound, w.Code, "path=%s should hit OpenAI realtime REST handler", tt.path)
		})
	}
}

func TestGatewayRoutesOpenAIRealtimeCallsAcceptRejectsNonOpenAIPlatform(t *testing.T) {
	router := newGatewayRoutesTestRouterForPlatform(service.PlatformAnthropic)

	req := httptest.NewRequest(http.MethodPost, "/v1/realtime/calls/call_123/accept", strings.NewReader(`{"type":"realtime","model":"gpt-realtime"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)
	require.Equal(t, http.StatusNotFound, w.Code)
	require.Contains(t, w.Body.String(), "Realtime API is not supported for this platform")
}

func TestGatewayRoutesOpenAIRealtimeRESTRejectsNonOpenAIPlatform(t *testing.T) {
	router := newGatewayRoutesTestRouterForPlatform(service.PlatformAnthropic)

	req := httptest.NewRequest(http.MethodPost, "/v1/realtime/client_secrets", strings.NewReader(`{"session":{"model":"gpt-realtime"}}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)
	require.Equal(t, http.StatusNotFound, w.Code)
	require.Contains(t, w.Body.String(), "Realtime API is not supported for this platform")
}

func TestGatewayRoutesOpenAIAudioTranscriptionsPathsAreRegistered(t *testing.T) {
	router := newGatewayRoutesTestRouter()

	for _, path := range []string{
		"/v1/audio/transcriptions",
		"/transcribe",
	} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader("invalid"))
		req.Header.Set("Content-Type", "multipart/form-data; boundary=test")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)
		require.NotEqual(t, http.StatusNotFound, w.Code, "path=%s should hit OpenAI audio transcriptions handler", path)
	}
}
