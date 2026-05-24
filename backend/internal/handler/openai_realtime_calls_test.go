package handler

import (
	"bytes"
	"context"
	"io"
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

type realtimeCallsHandlerAccountRepo struct {
	service.AccountRepository
	accounts []service.Account
}

func (r *realtimeCallsHandlerAccountRepo) ListSchedulableByPlatform(_ context.Context, platform string) ([]service.Account, error) {
	out := make([]service.Account, 0, len(r.accounts))
	for _, account := range r.accounts {
		if account.Platform == platform && account.IsSchedulable() {
			out = append(out, account)
		}
	}
	return out, nil
}

func (r *realtimeCallsHandlerAccountRepo) BatchUpdateLastUsed(context.Context, map[int64]time.Time) error {
	return nil
}

type realtimeCallsHandlerUpstream struct {
	statuses   []int
	accountIDs []int64
	urls       []string
	methods    []string
	bodies     [][]byte
}

func (u *realtimeCallsHandlerUpstream) Do(req *http.Request, _ string, accountID int64, _ int) (*http.Response, error) {
	u.accountIDs = append(u.accountIDs, accountID)
	if req != nil {
		u.methods = append(u.methods, req.Method)
		if req.URL != nil {
			u.urls = append(u.urls, req.URL.String())
		}
		if req.Body != nil {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
			u.bodies = append(u.bodies, body)
			req.Body = io.NopCloser(bytes.NewReader(body))
		}
	}
	status := http.StatusOK
	if idx := len(u.accountIDs) - 1; idx < len(u.statuses) {
		status = u.statuses[idx]
	}
	body := `{"id":"call_123","object":"realtime.call"}`
	if status >= 400 {
		body = `{"error":{"message":"temporary upstream failure"}}`
	}
	return &http.Response{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
			"X-Request-Id": []string{"rid-realtime-call"},
		},
		Body:    io.NopCloser(bytes.NewBufferString(body)),
		Request: req,
	}, nil
}

func (u *realtimeCallsHandlerUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

func TestOpenAIGatewayHandlerRealtimeCallsAccept_DispatchesAccount(t *testing.T) {
	upstream := &realtimeCallsHandlerUpstream{statuses: []int{http.StatusOK}}
	handler := newRealtimeCallsHandlerForTest(t, upstream, []service.Account{
		newRealtimeCallsHandlerAccount(1),
	})
	body := []byte(`{"type":"realtime","model":"gpt-realtime","instructions":"answer calls"}`)
	c, rec := newRealtimeCallsHandlerContext(t, "/v1/realtime/calls/call_123/accept", body)

	handler.RealtimeCallsAccept(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, []int64{1}, upstream.accountIDs)
	require.Equal(t, []string{http.MethodPost}, upstream.methods)
	require.Equal(t, []string{"https://api.openai.com/v1/realtime/calls/call_123/accept"}, upstream.urls)
	require.JSONEq(t, string(body), string(upstream.bodies[0]))
	require.JSONEq(t, `{"id":"call_123","object":"realtime.call"}`, rec.Body.String())
}

func TestOpenAIGatewayHandlerRealtimeCallsAccept_AccountMappingForwardsMappedModel(t *testing.T) {
	upstream := &realtimeCallsHandlerUpstream{statuses: []int{http.StatusOK}}
	account := newRealtimeCallsHandlerAccount(1)
	account.Credentials["model_mapping"] = map[string]any{
		"client-realtime": "gpt-realtime",
	}
	handler := newRealtimeCallsHandlerForTest(t, upstream, []service.Account{account})
	c, rec := newRealtimeCallsHandlerContext(t, "/v1/realtime/calls/call_123/accept", []byte(`{"type":"realtime","model":"client-realtime"}`))

	handler.RealtimeCallsAccept(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Len(t, upstream.bodies, 1)
	require.Equal(t, "gpt-realtime", gjson.GetBytes(upstream.bodies[0], "model").String())
}

func TestOpenAIGatewayHandlerRealtimeCallsAccept_AccountSwitch(t *testing.T) {
	upstream := &realtimeCallsHandlerUpstream{statuses: []int{http.StatusTooManyRequests, http.StatusOK}}
	handler := newRealtimeCallsHandlerForTest(t, upstream, []service.Account{
		newRealtimeCallsHandlerAccount(1),
		newRealtimeCallsHandlerAccount(2),
	})
	c, rec := newRealtimeCallsHandlerContext(t, "/v1/realtime/calls/call_123/accept", []byte(`{"type":"realtime","model":"gpt-realtime"}`))

	handler.RealtimeCallsAccept(c)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, []int64{1, 2}, upstream.accountIDs)
}

func newRealtimeCallsHandlerForTest(t *testing.T, upstream service.HTTPUpstream, accounts []service.Account) *OpenAIGatewayHandler {
	t.Helper()
	cfg := &config.Config{RunMode: config.RunModeSimple}
	cfg.Default.RateMultiplier = 1
	cfg.Gateway.Scheduling.LoadBatchEnabled = false

	accountRepo := &realtimeCallsHandlerAccountRepo{accounts: accounts}
	concurrencyService := service.NewConcurrencyService(nil)
	billingCacheService := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg)
	billingService := service.NewBillingService(cfg, nil)
	deferredService := service.NewDeferredService(accountRepo, nil, time.Minute)
	gatewayService := service.NewOpenAIGatewayService(
		accountRepo,
		nil,
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

func newRealtimeCallsHandlerContext(t *testing.T, path string, body []byte) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

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

func newRealtimeCallsHandlerAccount(id int64) service.Account {
	return service.Account{
		ID:          id,
		Name:        "openai-realtime",
		Platform:    service.PlatformOpenAI,
		Type:        service.AccountTypeAPIKey,
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 0,
		Priority:    int(id),
		Credentials: map[string]any{"api_key": "sk-test"},
	}
}
