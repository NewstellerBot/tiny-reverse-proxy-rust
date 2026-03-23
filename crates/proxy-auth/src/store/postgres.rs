use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, QueryBuilder, Row};

use crate::schema::POSTGRES_CREATE_TABLES;

use super::{
    AuthStore, PrincipalRecord, ProjectRecord, RoleBindingRecord, StoreError, TokenRecord,
};

pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(url)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        for sql in POSTGRES_CREATE_TABLES {
            sqlx::query(sql)
                .execute(&pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
        }

        Ok(Self { pool })
    }
}

#[async_trait]
impl AuthStore for PostgresStore {
    async fn get_project(&self, project_id: &str) -> Result<Option<ProjectRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, name, description, active, created_at FROM auth_projects WHERE project_id = $1",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|row| ProjectRecord {
            project_id: row.get::<String, _>(0),
            name: row.get::<String, _>(1),
            description: row.get::<Option<String>, _>(2),
            active: row.get::<bool, _>(3),
            created_at: row.get::<String, _>(4),
        }))
    }

    async fn get_all_projects(&self) -> Result<Vec<ProjectRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, name, description, active, created_at FROM auth_projects ORDER BY project_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| ProjectRecord {
                project_id: row.get::<String, _>(0),
                name: row.get::<String, _>(1),
                description: row.get::<Option<String>, _>(2),
                active: row.get::<bool, _>(3),
                created_at: row.get::<String, _>(4),
            })
            .collect())
    }

    async fn upsert_project(&self, record: &ProjectRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO auth_projects (project_id, name, description, active, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (project_id) DO UPDATE SET
                 name = EXCLUDED.name,
                 description = EXCLUDED.description,
                 active = EXCLUDED.active,
                 created_at = EXCLUDED.created_at",
        )
        .bind(&record.project_id)
        .bind(&record.name)
        .bind(&record.description)
        .bind(record.active)
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project(&self, project_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM auth_projects WHERE project_id = $1")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_principal(
        &self,
        principal_id: &str,
    ) -> Result<Option<PrincipalRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT principal_id, name, active, created_at FROM auth_principals WHERE principal_id = $1",
        )
        .bind(principal_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|row| PrincipalRecord {
            principal_id: row.get::<String, _>(0),
            name: row.get::<String, _>(1),
            active: row.get::<bool, _>(2),
            created_at: row.get::<String, _>(3),
        }))
    }

    async fn get_all_principals(&self) -> Result<Vec<PrincipalRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT principal_id, name, active, created_at FROM auth_principals ORDER BY principal_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| PrincipalRecord {
                principal_id: row.get::<String, _>(0),
                name: row.get::<String, _>(1),
                active: row.get::<bool, _>(2),
                created_at: row.get::<String, _>(3),
            })
            .collect())
    }

    async fn upsert_principal(&self, record: &PrincipalRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO auth_principals (principal_id, name, active, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (principal_id) DO UPDATE SET
                 name = EXCLUDED.name,
                 active = EXCLUDED.active,
                 created_at = EXCLUDED.created_at",
        )
        .bind(&record.principal_id)
        .bind(&record.name)
        .bind(record.active)
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_principal(&self, principal_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM auth_principals WHERE principal_id = $1")
            .bind(principal_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_token(&self, token_hash: &str) -> Result<Option<TokenRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT token_hash, principal_id, name, active, created_at FROM auth_tokens WHERE token_hash = $1",
        )
        .bind(token_hash)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|row| TokenRecord {
            token_hash: row.get::<String, _>(0),
            principal_id: row.get::<String, _>(1),
            name: row.get::<String, _>(2),
            active: row.get::<bool, _>(3),
            created_at: row.get::<String, _>(4),
        }))
    }

    async fn get_all_tokens(
        &self,
        principal_id: Option<&str>,
    ) -> Result<Vec<TokenRecord>, StoreError> {
        let mut query = QueryBuilder::<Postgres>::new(
            "SELECT token_hash, principal_id, name, active, created_at FROM auth_tokens",
        );
        if let Some(principal_id) = principal_id {
            query.push(" WHERE principal_id = ").push_bind(principal_id);
        }
        query.push(" ORDER BY token_hash");

        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| TokenRecord {
                token_hash: row.get::<String, _>(0),
                principal_id: row.get::<String, _>(1),
                name: row.get::<String, _>(2),
                active: row.get::<bool, _>(3),
                created_at: row.get::<String, _>(4),
            })
            .collect())
    }

    async fn upsert_token(&self, record: &TokenRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO auth_tokens (token_hash, principal_id, name, active, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (token_hash) DO UPDATE SET
                 principal_id = EXCLUDED.principal_id,
                 name = EXCLUDED.name,
                 active = EXCLUDED.active,
                 created_at = EXCLUDED.created_at",
        )
        .bind(&record.token_hash)
        .bind(&record.principal_id)
        .bind(&record.name)
        .bind(record.active)
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_token(&self, token_hash: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM auth_tokens WHERE token_hash = $1")
            .bind(token_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_role_bindings(
        &self,
        principal_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Result<Vec<RoleBindingRecord>, StoreError> {
        let mut query = QueryBuilder::<Postgres>::new(
            "SELECT binding_id, principal_id, role, project_id, created_at FROM auth_role_bindings",
        );

        let mut has_where = false;
        if let Some(principal_id) = principal_id {
            query.push(" WHERE principal_id = ").push_bind(principal_id);
            has_where = true;
        }
        if let Some(project_id) = project_id {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("project_id = ").push_bind(project_id);
        }
        query.push(" ORDER BY binding_id");

        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| RoleBindingRecord {
                binding_id: row.get::<String, _>(0),
                principal_id: row.get::<String, _>(1),
                role: row.get::<String, _>(2),
                project_id: row.get::<Option<String>, _>(3),
                created_at: row.get::<String, _>(4),
            })
            .collect())
    }

    async fn upsert_role_binding(&self, record: &RoleBindingRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO auth_role_bindings (binding_id, principal_id, role, project_id, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (binding_id) DO UPDATE SET
                 principal_id = EXCLUDED.principal_id,
                 role = EXCLUDED.role,
                 project_id = EXCLUDED.project_id,
                 created_at = EXCLUDED.created_at",
        )
        .bind(&record.binding_id)
        .bind(&record.principal_id)
        .bind(&record.role)
        .bind(&record.project_id)
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_role_binding(&self, binding_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM auth_role_bindings WHERE binding_id = $1")
            .bind(binding_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }
}
