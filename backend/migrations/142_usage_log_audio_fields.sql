-- Persist OpenAI audio usage breakdowns on raw usage logs and dashboard aggregates.

ALTER TABLE usage_logs
    ADD COLUMN IF NOT EXISTS audio_input_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_output_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_creation_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_read_tokens INT NOT NULL DEFAULT 0;

ALTER TABLE usage_dashboard_hourly
    ADD COLUMN IF NOT EXISTS audio_input_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_output_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_creation_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_read_tokens BIGINT NOT NULL DEFAULT 0;

ALTER TABLE usage_dashboard_daily
    ADD COLUMN IF NOT EXISTS audio_input_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_output_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_creation_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS audio_cache_read_tokens BIGINT NOT NULL DEFAULT 0;
