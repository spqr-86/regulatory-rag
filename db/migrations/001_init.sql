-- Event tables for module 05 (issue #15).
--
-- Applied by the postgres image on first start of an empty volume
-- (/docker-entrypoint-initdb.d). Everything is IF NOT EXISTS so that running the
-- file by hand against an existing database is a no-op rather than an error.
--
-- Field list follows the spec ("Что пишем на каждый запрос") and must stay in
-- step with EVENT_FIELDS in src/v7/telemetry.py.

CREATE TABLE IF NOT EXISTS queries (
    query_id          uuid        PRIMARY KEY,
    -- Which batch run wrote the row; NULL for live traffic (issue #18).
    run_id            text,
    ts                timestamptz NOT NULL,
    source            text        NOT NULL CHECK (source IN ('ui', 'api', 'eval', 'mcp')),
    question          text        NOT NULL,
    path              text        NOT NULL CHECK (path IN ('simple', 'complex', 'clarify', 'abstain')),
    answer_len        integer     NOT NULL,
    -- What reached the generation prompt; n_passages_found is what retrieval got.
    n_passages        integer     NOT NULL,
    n_passages_found  integer     NOT NULL,
    latency_ms        integer     NOT NULL,
    prompt_tokens     integer     NOT NULL,
    completion_tokens integer     NOT NULL,
    cost_usd          numeric(12, 6) NOT NULL,
    -- llm_usage as it stands: model, node, tokens per call.
    models            jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- Models with no rate card, so a cheap-looking report cannot lie.
    unpriced_models   text[]      NOT NULL DEFAULT '{}',
    error             text
);

-- The dashboard reads by time, and splits live traffic from eval runs by source.
CREATE INDEX IF NOT EXISTS queries_ts_idx ON queries (ts DESC);
CREATE INDEX IF NOT EXISTS queries_source_ts_idx ON queries (source, ts DESC);
CREATE INDEX IF NOT EXISTS queries_run_id_idx ON queries (run_id) WHERE run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS feedback (
    id       bigserial   PRIMARY KEY,
    query_id uuid        NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    ts       timestamptz NOT NULL DEFAULT now(),
    -- 👍 / 👎, nothing in between.
    verdict  smallint    NOT NULL CHECK (verdict IN (-1, 1)),
    comment  text
);

CREATE INDEX IF NOT EXISTS feedback_query_id_idx ON feedback (query_id);
CREATE INDEX IF NOT EXISTS feedback_ts_idx ON feedback (ts DESC);
