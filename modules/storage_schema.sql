PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS project_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL UNIQUE,
    export_name TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    media_type TEXT NOT NULL,
    payload BLOB NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (scope, key)
);
CREATE TABLE IF NOT EXISTS cache_entries (
    scope TEXT NOT NULL,
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    payload TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (scope, namespace, cache_key)
);
CREATE TABLE IF NOT EXISTS log_chunks (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    stream TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS log_stream ON log_chunks(scope, stream, id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    payload TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_scope ON runs(scope, kind, updated_at DESC);
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
INSERT INTO schema_info(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);
