#[cfg(feature = "store-mysql")]
pub mod mysql;
#[cfg(feature = "store-postgres")]
pub mod postgres;
#[cfg(feature = "store-sqlite")]
pub mod sqlite;

use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct ProjectRecord {
    pub project_id: String,
    pub name: String,
    pub description: Option<String>,
    pub active: bool,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct PrincipalRecord {
    pub principal_id: String,
    pub name: String,
    pub active: bool,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct TokenRecord {
    pub token_hash: String,
    pub principal_id: String,
    pub name: String,
    pub active: bool,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct RoleBindingRecord {
    pub binding_id: String,
    pub principal_id: String,
    pub role: String,
    pub project_id: Option<String>,
    pub created_at: String,
}

#[async_trait]
pub trait AuthStore: Send + Sync {
    async fn get_project(&self, project_id: &str) -> Result<Option<ProjectRecord>, StoreError>;
    async fn get_all_projects(&self) -> Result<Vec<ProjectRecord>, StoreError>;
    async fn upsert_project(&self, record: &ProjectRecord) -> Result<(), StoreError>;
    async fn delete_project(&self, project_id: &str) -> Result<bool, StoreError>;

    async fn get_principal(
        &self,
        principal_id: &str,
    ) -> Result<Option<PrincipalRecord>, StoreError>;
    async fn get_all_principals(&self) -> Result<Vec<PrincipalRecord>, StoreError>;
    async fn upsert_principal(&self, record: &PrincipalRecord) -> Result<(), StoreError>;
    async fn delete_principal(&self, principal_id: &str) -> Result<bool, StoreError>;

    async fn get_token(&self, token_hash: &str) -> Result<Option<TokenRecord>, StoreError>;
    async fn get_all_tokens(
        &self,
        principal_id: Option<&str>,
    ) -> Result<Vec<TokenRecord>, StoreError>;
    async fn upsert_token(&self, record: &TokenRecord) -> Result<(), StoreError>;
    async fn delete_token(&self, token_hash: &str) -> Result<bool, StoreError>;

    async fn get_role_bindings(
        &self,
        principal_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Result<Vec<RoleBindingRecord>, StoreError>;
    async fn upsert_role_binding(&self, record: &RoleBindingRecord) -> Result<(), StoreError>;
    async fn delete_role_binding(&self, binding_id: &str) -> Result<bool, StoreError>;
}

#[derive(Debug)]
pub enum StoreError {
    Db(String),
    Other(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Db(msg) => write!(f, "database error: {}", msg),
            StoreError::Other(msg) => write!(f, "store error: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

pub enum Store {
    #[cfg(feature = "store-sqlite")]
    Sqlite(sqlite::SqliteStore),
    #[cfg(feature = "store-postgres")]
    Postgres(postgres::PostgresStore),
    #[cfg(feature = "store-mysql")]
    Mysql(mysql::MysqlStore),
}

#[async_trait]
impl AuthStore for Store {
    async fn get_project(&self, project_id: &str) -> Result<Option<ProjectRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project(project_id).await,
        }
    }

    async fn get_all_projects(&self) -> Result<Vec<ProjectRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_projects().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_projects().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_projects().await,
        }
    }

    async fn upsert_project(&self, record: &ProjectRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project(record).await,
        }
    }

    async fn delete_project(&self, project_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_project(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_project(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_project(project_id).await,
        }
    }

    async fn get_principal(
        &self,
        principal_id: &str,
    ) -> Result<Option<PrincipalRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_principal(principal_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_principal(principal_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_principal(principal_id).await,
        }
    }

    async fn get_all_principals(&self) -> Result<Vec<PrincipalRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_principals().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_principals().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_principals().await,
        }
    }

    async fn upsert_principal(&self, record: &PrincipalRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_principal(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_principal(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_principal(record).await,
        }
    }

    async fn delete_principal(&self, principal_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_principal(principal_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_principal(principal_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_principal(principal_id).await,
        }
    }

    async fn get_token(&self, token_hash: &str) -> Result<Option<TokenRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_token(token_hash).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_token(token_hash).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_token(token_hash).await,
        }
    }

    async fn get_all_tokens(
        &self,
        principal_id: Option<&str>,
    ) -> Result<Vec<TokenRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_tokens(principal_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_tokens(principal_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_tokens(principal_id).await,
        }
    }

    async fn upsert_token(&self, record: &TokenRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_token(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_token(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_token(record).await,
        }
    }

    async fn delete_token(&self, token_hash: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_token(token_hash).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_token(token_hash).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_token(token_hash).await,
        }
    }

    async fn get_role_bindings(
        &self,
        principal_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Result<Vec<RoleBindingRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_role_bindings(principal_id, project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_role_bindings(principal_id, project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_role_bindings(principal_id, project_id).await,
        }
    }

    async fn upsert_role_binding(&self, record: &RoleBindingRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_role_binding(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_role_binding(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_role_binding(record).await,
        }
    }

    async fn delete_role_binding(&self, binding_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_role_binding(binding_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_role_binding(binding_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_role_binding(binding_id).await,
        }
    }
}

pub async fn connect(url: &str) -> Result<Store, StoreError> {
    if url.starts_with("sqlite:") {
        #[cfg(feature = "store-sqlite")]
        {
            return Ok(Store::Sqlite(sqlite::SqliteStore::connect(url).await?));
        }
        #[cfg(not(feature = "store-sqlite"))]
        {
            return Err(StoreError::Other(
                "SQLite support not compiled in (enable `store-sqlite`)".into(),
            ));
        }
    }
    if url.starts_with("postgres:") || url.starts_with("postgresql:") {
        #[cfg(feature = "store-postgres")]
        {
            return Ok(Store::Postgres(
                postgres::PostgresStore::connect(url).await?,
            ));
        }
        #[cfg(not(feature = "store-postgres"))]
        {
            return Err(StoreError::Other(
                "Postgres support not compiled in (enable `store-postgres`)".into(),
            ));
        }
    }
    if url.starts_with("mysql:") {
        #[cfg(feature = "store-mysql")]
        {
            return Ok(Store::Mysql(mysql::MysqlStore::connect(url).await?));
        }
        #[cfg(not(feature = "store-mysql"))]
        {
            return Err(StoreError::Other(
                "MySQL support not compiled in (enable `store-mysql`)".into(),
            ));
        }
    }
    Err(StoreError::Other(format!(
        "unsupported store URL scheme: {}",
        url
    )))
}
