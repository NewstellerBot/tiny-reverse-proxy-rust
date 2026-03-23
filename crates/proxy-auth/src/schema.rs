#[cfg(feature = "store-sqlite")]
pub const SQLITE_CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS auth_projects (
        project_id   TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT,
        active       INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )",
    "CREATE TABLE IF NOT EXISTS auth_principals (
        principal_id TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        active       INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )",
    "CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash   TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        name         TEXT NOT NULL,
        active       INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )",
    "CREATE TABLE IF NOT EXISTS auth_role_bindings (
        binding_id   TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        role         TEXT NOT NULL,
        project_id   TEXT,
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )",
    "CREATE INDEX IF NOT EXISTS idx_auth_tokens_principal ON auth_tokens(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_auth_bindings_principal ON auth_role_bindings(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_auth_bindings_project ON auth_role_bindings(project_id)",
];

#[cfg(feature = "store-postgres")]
pub const POSTGRES_CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS auth_projects (
        project_id   TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   TEXT NOT NULL DEFAULT ((EXTRACT(EPOCH FROM now()))::BIGINT::TEXT)
    )",
    "CREATE TABLE IF NOT EXISTS auth_principals (
        principal_id TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   TEXT NOT NULL DEFAULT ((EXTRACT(EPOCH FROM now()))::BIGINT::TEXT)
    )",
    "CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash   TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        name         TEXT NOT NULL,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   TEXT NOT NULL DEFAULT ((EXTRACT(EPOCH FROM now()))::BIGINT::TEXT)
    )",
    "CREATE TABLE IF NOT EXISTS auth_role_bindings (
        binding_id   TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        role         TEXT NOT NULL,
        project_id   TEXT,
        created_at   TEXT NOT NULL DEFAULT ((EXTRACT(EPOCH FROM now()))::BIGINT::TEXT)
    )",
    "CREATE INDEX IF NOT EXISTS idx_auth_tokens_principal ON auth_tokens(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_auth_bindings_principal ON auth_role_bindings(principal_id)",
    "CREATE INDEX IF NOT EXISTS idx_auth_bindings_project ON auth_role_bindings(project_id)",
];

#[cfg(feature = "store-mysql")]
pub const MYSQL_CREATE_TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS auth_projects (
        project_id   VARCHAR(255) PRIMARY KEY,
        name         VARCHAR(255) NOT NULL,
        description  TEXT,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   VARCHAR(32) NOT NULL DEFAULT (CAST(UNIX_TIMESTAMP() AS CHAR))
    )",
    "CREATE TABLE IF NOT EXISTS auth_principals (
        principal_id VARCHAR(255) PRIMARY KEY,
        name         VARCHAR(255) NOT NULL,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   VARCHAR(32) NOT NULL DEFAULT (CAST(UNIX_TIMESTAMP() AS CHAR))
    )",
    "CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash   VARCHAR(255) PRIMARY KEY,
        principal_id VARCHAR(255) NOT NULL,
        name         VARCHAR(255) NOT NULL,
        active       BOOLEAN NOT NULL DEFAULT TRUE,
        created_at   VARCHAR(32) NOT NULL DEFAULT (CAST(UNIX_TIMESTAMP() AS CHAR)),
        INDEX idx_auth_tokens_principal (principal_id)
    )",
    "CREATE TABLE IF NOT EXISTS auth_role_bindings (
        binding_id   VARCHAR(255) PRIMARY KEY,
        principal_id VARCHAR(255) NOT NULL,
        role         VARCHAR(64) NOT NULL,
        project_id   VARCHAR(255),
        created_at   VARCHAR(32) NOT NULL DEFAULT (CAST(UNIX_TIMESTAMP() AS CHAR)),
        INDEX idx_auth_bindings_principal (principal_id),
        INDEX idx_auth_bindings_project (project_id)
    )",
];
