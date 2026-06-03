package service

import (
	"context"
	"strings"
	"time"
)

const (
	modelRateLimitsKey                 = "model_rate_limits"
	antigravityGeminiModelRateLimitKey = "antigravity:gemini"
)

type modelRateLimitExtraScopesContextKey struct{}

func WithModelRateLimitExtraScopes(ctx context.Context, scopes ...string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	cleaned := make([]string, 0, len(scopes))
	for _, scope := range scopes {
		scope = strings.TrimSpace(scope)
		if scope != "" {
			cleaned = append(cleaned, scope)
		}
	}
	if len(cleaned) == 0 {
		return ctx
	}
	return context.WithValue(ctx, modelRateLimitExtraScopesContextKey{}, cleaned)
}

func modelRateLimitExtraScopesFromContext(ctx context.Context) []string {
	if ctx == nil {
		return nil
	}
	scopes, _ := ctx.Value(modelRateLimitExtraScopesContextKey{}).([]string)
	return scopes
}

// isRateLimitActiveForKey 检查指定 key 的限流是否生效
func (a *Account) isRateLimitActiveForKey(key string) bool {
	resetAt := a.modelRateLimitResetAt(key)
	return resetAt != nil && time.Now().Before(*resetAt)
}

// getRateLimitRemainingForKey 获取指定 key 的限流剩余时间，0 表示未限流或已过期
func (a *Account) getRateLimitRemainingForKey(key string) time.Duration {
	resetAt := a.modelRateLimitResetAt(key)
	if resetAt == nil {
		return 0
	}
	remaining := time.Until(*resetAt)
	if remaining > 0 {
		return remaining
	}
	return 0
}

func (a *Account) isModelRateLimitedWithContext(ctx context.Context, requestedModel string) bool {
	if a == nil {
		return false
	}

	modelKey := a.GetMappedModel(requestedModel)
	if a.Platform == PlatformAntigravity {
		modelKey = resolveFinalAntigravityModelKey(ctx, a, requestedModel)
		if isAntigravityGeminiModel(modelKey) && a.isRateLimitActiveForKey(antigravityGeminiModelRateLimitKey) {
			return true
		}
	}
	modelKey = strings.TrimSpace(modelKey)
	if modelKey == "" {
		return false
	}
	if a.isRateLimitActiveForKey(modelKey) {
		return true
	}
	for _, scope := range modelRateLimitExtraScopesFromContext(ctx) {
		if a.isRateLimitActiveForKey(scope) {
			return true
		}
	}
	return false
}

// GetModelRateLimitRemainingTime 获取模型限流剩余时间
// 返回 0 表示未限流或已过期
func (a *Account) GetModelRateLimitRemainingTime(requestedModel string) time.Duration {
	return a.GetModelRateLimitRemainingTimeWithContext(context.Background(), requestedModel)
}

func (a *Account) GetModelRateLimitRemainingTimeWithContext(ctx context.Context, requestedModel string) time.Duration {
	if a == nil {
		return 0
	}

	modelKey := a.GetMappedModel(requestedModel)
	if a.Platform == PlatformAntigravity {
		modelKey = resolveFinalAntigravityModelKey(ctx, a, requestedModel)
	}
	modelKey = strings.TrimSpace(modelKey)
	if modelKey == "" {
		return 0
	}
	remaining := a.getRateLimitRemainingForKey(modelKey)
	for _, scope := range modelRateLimitExtraScopesFromContext(ctx) {
		if scopeRemaining := a.getRateLimitRemainingForKey(scope); scopeRemaining > remaining {
			remaining = scopeRemaining
		}
	}
	if a.Platform == PlatformAntigravity && isAntigravityGeminiModel(modelKey) {
		if familyRemaining := a.getRateLimitRemainingForKey(antigravityGeminiModelRateLimitKey); familyRemaining > remaining {
			remaining = familyRemaining
		}
	}
	return remaining
}

func resolveFinalAntigravityModelKey(ctx context.Context, account *Account, requestedModel string) string {
	modelKey := mapAntigravityModel(account, requestedModel)
	if modelKey == "" {
		return ""
	}
	// thinking 会影响 Antigravity 最终模型名（例如 claude-sonnet-4-5 -> claude-sonnet-4-5-thinking）
	if enabled, ok := ThinkingEnabledFromContext(ctx); ok {
		modelKey = applyThinkingModelSuffix(modelKey, enabled)
	}
	return modelKey
}

func isAntigravityGeminiModel(model string) bool {
	return strings.HasPrefix(normalizeAntigravityModelName(model), "gemini-")
}

func antigravityModelRateLimitKeys(model string) []string {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	keys := []string{model}
	if isAntigravityGeminiModel(model) && model != antigravityGeminiModelRateLimitKey {
		keys = append(keys, antigravityGeminiModelRateLimitKey)
	}
	return keys
}

func (a *Account) modelRateLimitResetAt(scope string) *time.Time {
	if a == nil || a.Extra == nil || scope == "" {
		return nil
	}
	rawLimits, ok := a.Extra[modelRateLimitsKey].(map[string]any)
	if !ok {
		return nil
	}
	rawLimit, ok := rawLimits[scope].(map[string]any)
	if !ok {
		return nil
	}
	resetAtRaw, ok := rawLimit["rate_limit_reset_at"].(string)
	if !ok || strings.TrimSpace(resetAtRaw) == "" {
		return nil
	}
	resetAt, err := time.Parse(time.RFC3339, resetAtRaw)
	if err != nil {
		return nil
	}
	return &resetAt
}
