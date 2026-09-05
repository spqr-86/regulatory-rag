-- One vote per answer (issue #20).
--
-- The buttons upsert on query_id: a second press is the user correcting
-- themselves, not a second opinion, so the panel counts answers and not clicks.
-- ON CONFLICT (query_id) needs a unique key to conflict on, which the initial
-- migration does not create — feedback_query_id_idx there is non-unique.
--
-- Applied by the postgres image on first start of an empty volume, in file-name
-- order after 001. Against a database that already has rows, run it by hand:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     < db/migrations/002_feedback_one_vote.sql
-- Duplicate rows from before the index (there should be none) make the CREATE
-- fail rather than pick a survivor — deciding which vote counts is not the
-- migration's call.

CREATE UNIQUE INDEX IF NOT EXISTS feedback_one_per_query_idx ON feedback (query_id);
