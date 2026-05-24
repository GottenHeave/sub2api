package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func init() { gin.SetMode(gin.TestMode) }

// ──────────────────────────────────────────────────────────
// NormalizeInboundEndpoint
// ──────────────────────────────────────────────────────────

func TestNormalizeInboundEndpoint(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		// Direct canonical paths.
		{"/v1/messages", EndpointMessages},
		{"/v1/chat/completions", EndpointChatCompletions},
		{"/v1/responses", EndpointResponses},
		{"/v1/realtime", EndpointRealtime},
		{"/v1/realtime/calls/call_123/accept", EndpointRealtimeCallsAccept},
		{"/v1/realtime/client_secrets", EndpointRealtimeREST},
		{"/v1/realtime/translations/client_secrets", EndpointRealtimeREST},
		{"/v1/realtime/calls", EndpointRealtimeREST},
		{"/v1/realtime/translations/calls", EndpointRealtimeREST},
		{"/v1/realtime/calls/call_123/hangup", EndpointRealtimeREST},
		{"/v1/realtime/calls/call_123/refer", EndpointRealtimeREST},
		{"/v1/realtime/calls/call_123/reject", EndpointRealtimeREST},
		{"/v1/realtime/sessions", EndpointRealtimeREST},
		{"/v1/realtime/transcription_sessions", EndpointRealtimeREST},
		{"/v1/audio/transcriptions", EndpointAudioTranscriptions},
		{"/transcribe", EndpointTranscribe},
		{"/v1/images/generations", EndpointImagesGenerations},
		{"/v1/images/edits", EndpointImagesEdits},
		{"/v1beta/models", EndpointGeminiModels},

		// Prefixed paths (antigravity, openai).
		{"/antigravity/v1/messages", EndpointMessages},
		{"/openai/v1/responses", EndpointResponses},
		{"/openai/v1/responses/compact", EndpointResponses},
		{"/openai/v1/realtime/calls/call_123/accept", EndpointRealtimeCallsAccept},
		{"/openai/v1/realtime/client_secrets", EndpointRealtimeREST},
		{"/openai/v1/realtime/calls/call_123/reject", EndpointRealtimeREST},
		{"/openai/v1/audio/transcriptions", EndpointAudioTranscriptions},
		{"/openai/v1/images/generations", EndpointImagesGenerations},
		{"/openai/v1/images/edits", EndpointImagesEdits},
		{"/antigravity/v1beta/models/gemini:generateContent", EndpointGeminiModels},

		// Gin route patterns with wildcards.
		{"/v1beta/models/*modelAction", EndpointGeminiModels},
		{"/v1/responses/*subpath", EndpointResponses},

		// Unknown path is returned as-is.
		{"/v1/embeddings", "/v1/embeddings"},
		{"", ""},
		{"  /v1/messages  ", EndpointMessages},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			require.Equal(t, tt.want, NormalizeInboundEndpoint(tt.path))
		})
	}
}

// ──────────────────────────────────────────────────────────
// DeriveUpstreamEndpoint
// ──────────────────────────────────────────────────────────

func TestDeriveUpstreamEndpoint(t *testing.T) {
	tests := []struct {
		name     string
		inbound  string
		rawPath  string
		platform string
		want     string
	}{
		// Anthropic.
		{"anthropic messages", EndpointMessages, "/v1/messages", service.PlatformAnthropic, EndpointMessages},

		// Gemini.
		{"gemini models", EndpointGeminiModels, "/v1beta/models/gemini:gen", service.PlatformGemini, EndpointGeminiModels},

		// OpenAI — always /v1/responses.
		{"openai responses root", EndpointResponses, "/v1/responses", service.PlatformOpenAI, EndpointResponses},
		{"openai responses compact", EndpointResponses, "/openai/v1/responses/compact", service.PlatformOpenAI, "/v1/responses/compact"},
		{"openai responses nested", EndpointResponses, "/openai/v1/responses/compact/detail", service.PlatformOpenAI, "/v1/responses/compact/detail"},
		{"openai from messages", EndpointMessages, "/v1/messages", service.PlatformOpenAI, EndpointResponses},
		{"openai from completions", EndpointChatCompletions, "/v1/chat/completions", service.PlatformOpenAI, EndpointResponses},
		{"openai realtime", EndpointRealtime, "/v1/realtime", service.PlatformOpenAI, EndpointRealtime},
		{"openai realtime calls accept", EndpointRealtimeCallsAccept, "/v1/realtime/calls/call_123/accept", service.PlatformOpenAI, "/v1/realtime/calls/call_123/accept"},
		{"openai prefixed realtime calls accept", EndpointRealtimeCallsAccept, "/openai/v1/realtime/calls/call_456/accept", service.PlatformOpenAI, "/v1/realtime/calls/call_456/accept"},
		{"openai realtime client secrets", EndpointRealtimeREST, "/v1/realtime/client_secrets", service.PlatformOpenAI, "/v1/realtime/client_secrets"},
		{"openai realtime translation client secrets", EndpointRealtimeREST, "/v1/realtime/translations/client_secrets", service.PlatformOpenAI, "/v1/realtime/translations/client_secrets"},
		{"openai realtime calls create", EndpointRealtimeREST, "/v1/realtime/calls", service.PlatformOpenAI, "/v1/realtime/calls"},
		{"openai realtime translation calls create", EndpointRealtimeREST, "/v1/realtime/translations/calls", service.PlatformOpenAI, "/v1/realtime/translations/calls"},
		{"openai realtime calls hangup", EndpointRealtimeREST, "/v1/realtime/calls/call_123/hangup", service.PlatformOpenAI, "/v1/realtime/calls/call_123/hangup"},
		{"openai realtime calls refer", EndpointRealtimeREST, "/v1/realtime/calls/call_123/refer", service.PlatformOpenAI, "/v1/realtime/calls/call_123/refer"},
		{"openai realtime calls reject", EndpointRealtimeREST, "/v1/realtime/calls/call_123/reject", service.PlatformOpenAI, "/v1/realtime/calls/call_123/reject"},
		{"openai realtime sessions", EndpointRealtimeREST, "/v1/realtime/sessions", service.PlatformOpenAI, "/v1/realtime/sessions"},
		{"openai realtime transcription sessions", EndpointRealtimeREST, "/v1/realtime/transcription_sessions", service.PlatformOpenAI, "/v1/realtime/transcription_sessions"},
		{"openai prefixed realtime reject", EndpointRealtimeREST, "/openai/v1/realtime/calls/call_789/reject", service.PlatformOpenAI, "/v1/realtime/calls/call_789/reject"},
		{"openai prefixed realtime translation calls", EndpointRealtimeREST, "/openai/v1/realtime/translations/calls", service.PlatformOpenAI, "/v1/realtime/translations/calls"},
		{"openai audio transcriptions", EndpointAudioTranscriptions, "/v1/audio/transcriptions", service.PlatformOpenAI, EndpointAudioTranscriptions},
		{"openai transcribe alias", EndpointTranscribe, "/transcribe", service.PlatformOpenAI, EndpointAudioTranscriptions},
		{"openai image generations", EndpointImagesGenerations, "/v1/images/generations", service.PlatformOpenAI, EndpointImagesGenerations},
		{"openai image edits", EndpointImagesEdits, "/openai/v1/images/edits", service.PlatformOpenAI, EndpointImagesEdits},

		// Antigravity — uses inbound to pick Claude vs Gemini upstream.
		{"antigravity claude", EndpointMessages, "/antigravity/v1/messages", service.PlatformAntigravity, EndpointMessages},
		{"antigravity gemini", EndpointGeminiModels, "/antigravity/v1beta/models", service.PlatformAntigravity, EndpointGeminiModels},

		// Unknown platform — passthrough.
		{"unknown platform", "/v1/embeddings", "/v1/embeddings", "unknown", "/v1/embeddings"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, DeriveUpstreamEndpoint(tt.inbound, tt.rawPath, tt.platform))
		})
	}
}

// ──────────────────────────────────────────────────────────
// responsesSubpathSuffix
// ──────────────────────────────────────────────────────────

func TestResponsesSubpathSuffix(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{"/v1/responses", ""},
		{"/v1/responses/", ""},
		{"/v1/responses/compact", "/compact"},
		{"/openai/v1/responses/compact/detail", "/compact/detail"},
		{"/v1/messages", ""},
		{"", ""},
	}
	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			require.Equal(t, tt.want, responsesSubpathSuffix(tt.raw))
		})
	}
}

// ──────────────────────────────────────────────────────────
// InboundEndpointMiddleware + context helpers
// ──────────────────────────────────────────────────────────

func TestInboundEndpointMiddleware(t *testing.T) {
	router := gin.New()
	router.Use(InboundEndpointMiddleware())

	var captured string
	router.POST("/v1/messages", func(c *gin.Context) {
		captured = GetInboundEndpoint(c)
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, EndpointMessages, captured)
}

func TestGetInboundEndpoint_FallbackWithoutMiddleware(t *testing.T) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)

	// Middleware did not run — fallback to normalizing c.Request.URL.Path.
	got := GetInboundEndpoint(c)
	require.Equal(t, EndpointMessages, got)
}

func TestGetUpstreamEndpoint_FullFlow(t *testing.T) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/openai/v1/responses/compact", nil)

	// Simulate middleware.
	c.Set(ctxKeyInboundEndpoint, NormalizeInboundEndpoint(c.Request.URL.Path))

	got := GetUpstreamEndpoint(c, service.PlatformOpenAI)
	require.Equal(t, "/v1/responses/compact", got)
}
