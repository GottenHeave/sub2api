package handler

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type audioTranscriptionHandlerAccountRepo struct {
	service.AccountRepository
	accounts []service.Account
}

func (r *audioTranscriptionHandlerAccountRepo) ListSchedulableByPlatform(_ context.Context, platform string) ([]service.Account, error) {
	out := make([]service.Account, 0, len(r.accounts))
	for _, account := range r.accounts {
		if account.Platform == platform && account.IsSchedulable() {
			out = append(out, account)
		}
	}
	return out, nil
}

func (r *audioTranscriptionHandlerAccountRepo) BatchUpdateLastUsed(context.Context, map[int64]time.Time) error {
	return nil
}

type audioTranscriptionHandlerUsageLogRepo struct {
	service.UsageLogRepository
	logs []service.UsageLog
}

func (r *audioTranscriptionHandlerUsageLogRepo) Create(_ context.Context, log *service.UsageLog) (bool, error) {
	if log != nil {
		r.logs = append(r.logs, *log)
	}
	return true, nil
}

type audioTranscriptionHandlerUpstream struct {
	statuses   []int
	accountIDs []int64
	urls       []string
	bodies     [][]byte
}

func (u *audioTranscriptionHandlerUpstream) Do(req *http.Request, _ string, accountID int64, _ int) (*http.Response, error) {
	u.accountIDs = append(u.accountIDs, accountID)
	if req != nil && req.URL != nil {
		u.urls = append(u.urls, req.URL.String())
	}
	if req != nil && req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		u.bodies = append(u.bodies, body)
		req.Body = io.NopCloser(bytes.NewReader(body))
	}
	status := http.StatusOK
	if idx := len(u.accountIDs) - 1; idx < len(u.statuses) {
		status = u.statuses[idx]
	}
	body := `{"text":"ok"}`
	if status >= 400 {
		body = `{"error":{"message":"temporary upstream failure"}}`
	}
	return &http.Response{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
			"X-Request-Id": []string{"rid-audio-handler"},
		},
		Body:    io.NopCloser(bytes.NewBufferString(body)),
		Request: req,
	}, nil
}

func (u *audioTranscriptionHandlerUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

func TestOpenAIGatewayHandlerAudioTranscriptions_DispatchesOAuthAccount(t *testing.T) {
	upstream := &audioTranscriptionHandlerUpstream{statuses: []int{http.StatusOK}}
	handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{
		newAudioTranscriptionHandlerOAuthAccount(9),
	})
	c, rec := newAudioTranscriptionHandlerContext(t, "/transcribe")

	handler.AudioTranscriptions(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, []int64{9}, upstream.accountIDs)
	require.Equal(t, []string{"https://chatgpt.com/backend-api/transcribe"}, upstream.urls)
	require.JSONEq(t, `{"text":"ok"}`, rec.Body.String())
}

func TestOpenAIGatewayHandlerAudioTranscriptions_FallbackSelectionStillForwardsRequestedModel(t *testing.T) {
	tests := []struct {
		model string
	}{
		{model: "gpt-4o-transcribe"},
		{model: "whisper-1"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			upstream := &audioTranscriptionHandlerUpstream{statuses: []int{http.StatusOK}}
			account := newAudioTranscriptionHandlerAccount(1)
			account.Credentials["model_mapping"] = map[string]any{
				service.OpenAIAudioTranscriptionsDefaultModel: service.OpenAIAudioTranscriptionsDefaultModel,
			}
			handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{account})
			c, rec := newAudioTranscriptionHandlerContextWithModel(t, "/v1/audio/transcriptions", tt.model)

			handler.AudioTranscriptions(c)

			require.Equal(t, http.StatusOK, rec.Code)
			require.Equal(t, []int64{1}, upstream.accountIDs)
			require.Len(t, upstream.bodies, 1)
			require.Contains(t, string(upstream.bodies[0]), tt.model)
			require.NotContains(t, string(upstream.bodies[0]), service.OpenAIAudioTranscriptionsDefaultModel)
		})
	}
}

func TestOpenAIGatewayHandlerAudioTranscriptions_SameAccountRetry(t *testing.T) {
	tests := []struct {
		name   string
		status int
	}{
		{name: "401", status: http.StatusUnauthorized},
		{name: "403", status: http.StatusForbidden},
		{name: "429", status: http.StatusTooManyRequests},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			account := newAudioTranscriptionHandlerAccount(1)
			account.Credentials["pool_mode"] = true
			account.Credentials["pool_mode_retry_count"] = 1
			upstream := &audioTranscriptionHandlerUpstream{statuses: []int{tt.status, http.StatusOK}}
			handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{account})
			c, rec := newAudioTranscriptionHandlerContext(t, "/v1/audio/transcriptions")

			handler.AudioTranscriptions(c)

			require.Equal(t, http.StatusOK, rec.Code)
			require.Equal(t, []int64{1, 1}, upstream.accountIDs)
			require.JSONEq(t, `{"text":"ok"}`, rec.Body.String())
		})
	}
}

func TestOpenAIGatewayHandlerAudioTranscriptions_OAuthServerErrorSameAccountRetry(t *testing.T) {
	upstream := &audioTranscriptionHandlerUpstream{statuses: []int{http.StatusInternalServerError, http.StatusOK}}
	handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{
		newAudioTranscriptionHandlerOAuthAccount(9),
	})
	c, rec := newAudioTranscriptionHandlerContext(t, "/transcribe")

	handler.AudioTranscriptions(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, []int64{9, 9}, upstream.accountIDs)
	require.Equal(t, []string{
		"https://chatgpt.com/backend-api/transcribe",
		"https://chatgpt.com/backend-api/transcribe",
	}, upstream.urls)
	require.JSONEq(t, `{"text":"ok"}`, rec.Body.String())
}

func TestOpenAIGatewayHandlerAudioTranscriptions_AccountSwitch(t *testing.T) {
	tests := []struct {
		name   string
		status int
	}{
		{name: "401", status: http.StatusUnauthorized},
		{name: "403", status: http.StatusForbidden},
		{name: "429", status: http.StatusTooManyRequests},
		{name: "500", status: http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := &audioTranscriptionHandlerUpstream{statuses: []int{tt.status, http.StatusOK}}
			handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{
				newAudioTranscriptionHandlerAccount(1),
				newAudioTranscriptionHandlerAccount(2),
			})
			c, rec := newAudioTranscriptionHandlerContext(t, "/v1/audio/transcriptions")

			handler.AudioTranscriptions(c)

			require.Equal(t, http.StatusOK, rec.Code)
			require.Equal(t, []int64{1, 2}, upstream.accountIDs)
			require.JSONEq(t, `{"text":"ok"}`, rec.Body.String())
		})
	}
}

func TestOpenAIGatewayHandlerAudioTranscriptions_ExhaustedFailure(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		wantStatus int
		wantType   string
	}{
		{name: "401", status: http.StatusUnauthorized, wantStatus: http.StatusBadGateway, wantType: "upstream_error"},
		{name: "403", status: http.StatusForbidden, wantStatus: http.StatusBadGateway, wantType: "upstream_error"},
		{name: "429", status: http.StatusTooManyRequests, wantStatus: http.StatusTooManyRequests, wantType: "rate_limit_error"},
		{name: "500", status: http.StatusInternalServerError, wantStatus: http.StatusBadGateway, wantType: "upstream_error"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := &audioTranscriptionHandlerUpstream{statuses: []int{tt.status}}
			handler := newAudioTranscriptionHandlerForTest(t, upstream, []service.Account{
				newAudioTranscriptionHandlerAccount(1),
			})
			c, rec := newAudioTranscriptionHandlerContext(t, "/v1/audio/transcriptions")

			handler.AudioTranscriptions(c)

			require.Equal(t, tt.wantStatus, rec.Code)
			require.Equal(t, []int64{1}, upstream.accountIDs)
			require.Equal(t, tt.wantType, gjson.GetBytes(rec.Body.Bytes(), "error.type").String())
		})
	}
}

func newAudioTranscriptionHandlerForTest(t *testing.T, upstream service.HTTPUpstream, accounts []service.Account) *OpenAIGatewayHandler {
	t.Helper()
	cfg := &config.Config{RunMode: config.RunModeSimple}
	cfg.Default.RateMultiplier = 1
	cfg.Gateway.Scheduling.LoadBatchEnabled = false

	accountRepo := &audioTranscriptionHandlerAccountRepo{accounts: accounts}
	usageRepo := &audioTranscriptionHandlerUsageLogRepo{}
	concurrencyService := service.NewConcurrencyService(nil)
	billingCacheService := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg)
	billingService := service.NewBillingService(cfg, nil)
	deferredService := service.NewDeferredService(accountRepo, nil, time.Minute)
	gatewayService := service.NewOpenAIGatewayService(
		accountRepo,
		usageRepo,
		nil,
		nil,
		nil,
		nil,
		nil,
		cfg,
		nil,
		concurrencyService,
		billingService,
		nil,
		billingCacheService,
		upstream,
		deferredService,
		nil,
		nil,
		nil,
		nil,
		nil,
	)

	return NewOpenAIGatewayHandler(
		gatewayService,
		concurrencyService,
		billingCacheService,
		&service.APIKeyService{},
		nil,
		nil,
		nil,
		cfg,
	)
}

func newAudioTranscriptionHandlerContext(t *testing.T, path string) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	return newAudioTranscriptionHandlerContextWithModel(t, path, service.OpenAIAudioTranscriptionsDefaultModel)
}

func newAudioTranscriptionHandlerContextWithModel(t *testing.T, path string, model string) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	body, contentType := buildAudioTranscriptionHandlerMultipartWithModel(t, model)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", contentType)

	groupID := int64(10)
	user := &service.User{ID: 42, Status: service.StatusActive, Balance: 100}
	group := &service.Group{
		ID:             groupID,
		Name:           "openai",
		Platform:       service.PlatformOpenAI,
		RateMultiplier: 1,
		Status:         service.StatusActive,
	}
	apiKey := &service.APIKey{
		ID:      77,
		UserID:  user.ID,
		Status:  service.StatusActive,
		User:    user,
		GroupID: &groupID,
		Group:   group,
	}
	c.Set(string(middleware.ContextKeyAPIKey), apiKey)
	c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: user.ID})
	c.Set(string(middleware.ContextKeyUserRole), service.RoleUser)
	return c, rec
}

func buildAudioTranscriptionHandlerMultipartWithModel(t *testing.T, model string) ([]byte, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = part.Write([]byte("fake-audio"))
	require.NoError(t, err)
	require.NoError(t, writer.WriteField("model", model))
	require.NoError(t, writer.Close())
	return body.Bytes(), writer.FormDataContentType()
}

func newAudioTranscriptionHandlerAccount(id int64) service.Account {
	return service.Account{
		ID:          id,
		Name:        "openai-audio",
		Platform:    service.PlatformOpenAI,
		Type:        service.AccountTypeAPIKey,
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 0,
		Priority:    int(id),
		Credentials: map[string]any{"api_key": "sk-test"},
	}
}

func newAudioTranscriptionHandlerOAuthAccount(id int64) service.Account {
	return service.Account{
		ID:          id,
		Name:        "openai-oauth-audio",
		Platform:    service.PlatformOpenAI,
		Type:        service.AccountTypeOAuth,
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 0,
		Priority:    int(id),
		Credentials: map[string]any{
			"access_token": "oauth-test-token",
		},
	}
}
