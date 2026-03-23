use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::store::{
    AuthStore, PrincipalRecord, ProjectRecord, RoleBindingRecord, Store, TokenRecord,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrincipalId(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthSource {
    Bootstrap,
    ControlPlaneToken,
    RuntimeKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    InstanceAdmin,
    ProjectAdmin,
    ProjectOperator,
    ProjectViewer,
    ProjectRuntime,
}

impl Role {
    pub fn all() -> &'static [Self] {
        const ALL_ROLES: [Role; 5] = [
            Role::InstanceAdmin,
            Role::ProjectAdmin,
            Role::ProjectOperator,
            Role::ProjectViewer,
            Role::ProjectRuntime,
        ];
        &ALL_ROLES
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Role::InstanceAdmin => "instance_admin",
            Role::ProjectAdmin => "project_admin",
            Role::ProjectOperator => "project_operator",
            Role::ProjectViewer => "project_viewer",
            Role::ProjectRuntime => "project_runtime",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "instance_admin" => Some(Self::InstanceAdmin),
            "project_admin" => Some(Self::ProjectAdmin),
            "project_operator" => Some(Self::ProjectOperator),
            "project_viewer" => Some(Self::ProjectViewer),
            "project_runtime" => Some(Self::ProjectRuntime),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Role::InstanceAdmin => "Unscoped administrator with full instance and project access.",
            Role::ProjectAdmin => "Project-scoped administrator for policy, keys, usage, and access management.",
            Role::ProjectOperator => "Project operator for day-to-day runtime and policy changes without principal administration.",
            Role::ProjectViewer => "Read-only project observer for status, usage, logs, and policy visibility.",
            Role::ProjectRuntime => "Runtime-only identity allowed to invoke inference for a single project.",
        }
    }

    pub fn permissions(&self) -> Vec<Permission> {
        Permission::all()
            .iter()
            .filter(|permission| role_allows(self, permission))
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Permission {
    ViewStatus,
    ViewProjects,
    ManageProjects,
    ViewPrincipals,
    ManagePrincipals,
    ViewRoleBindings,
    ManageRoleBindings,
    ViewRuntimeKeys,
    ManageRuntimeKeys,
    ViewProjectPolicy,
    ManageProjectPolicy,
    ViewRoutingRules,
    ManageRoutingRules,
    ViewUsage,
    ManageUsage,
    ViewLogs,
    ViewProviders,
    ManageProviders,
    ViewRateLimiter,
    ViewModelCosts,
    ManageModelCosts,
    InvokeInference,
}

impl Permission {
    pub fn all() -> &'static [Self] {
        const ALL_PERMISSIONS: [Permission; 22] = [
            Permission::ViewStatus,
            Permission::ViewProjects,
            Permission::ManageProjects,
            Permission::ViewPrincipals,
            Permission::ManagePrincipals,
            Permission::ViewRoleBindings,
            Permission::ManageRoleBindings,
            Permission::ViewRuntimeKeys,
            Permission::ManageRuntimeKeys,
            Permission::ViewProjectPolicy,
            Permission::ManageProjectPolicy,
            Permission::ViewRoutingRules,
            Permission::ManageRoutingRules,
            Permission::ViewUsage,
            Permission::ManageUsage,
            Permission::ViewLogs,
            Permission::ViewProviders,
            Permission::ManageProviders,
            Permission::ViewRateLimiter,
            Permission::ViewModelCosts,
            Permission::ManageModelCosts,
            Permission::InvokeInference,
        ];
        &ALL_PERMISSIONS
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Permission::ViewStatus => "view_status",
            Permission::ViewProjects => "view_projects",
            Permission::ManageProjects => "manage_projects",
            Permission::ViewPrincipals => "view_principals",
            Permission::ManagePrincipals => "manage_principals",
            Permission::ViewRoleBindings => "view_role_bindings",
            Permission::ManageRoleBindings => "manage_role_bindings",
            Permission::ViewRuntimeKeys => "view_runtime_keys",
            Permission::ManageRuntimeKeys => "manage_runtime_keys",
            Permission::ViewProjectPolicy => "view_project_policy",
            Permission::ManageProjectPolicy => "manage_project_policy",
            Permission::ViewRoutingRules => "view_routing_rules",
            Permission::ManageRoutingRules => "manage_routing_rules",
            Permission::ViewUsage => "view_usage",
            Permission::ManageUsage => "manage_usage",
            Permission::ViewLogs => "view_logs",
            Permission::ViewProviders => "view_providers",
            Permission::ManageProviders => "manage_providers",
            Permission::ViewRateLimiter => "view_rate_limiter",
            Permission::ViewModelCosts => "view_model_costs",
            Permission::ManageModelCosts => "manage_model_costs",
            Permission::InvokeInference => "invoke_inference",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoleAssignment {
    pub role: Role,
    pub project_id: Option<ProjectId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthContext {
    pub principal_id: PrincipalId,
    pub source: AuthSource,
    pub assignments: Vec<RoleAssignment>,
}

impl AuthContext {
    pub fn runtime(project_id: impl Into<String>, runtime_key_id: impl Into<String>) -> Self {
        Self {
            principal_id: PrincipalId(runtime_key_id.into()),
            source: AuthSource::RuntimeKey,
            assignments: vec![RoleAssignment {
                role: Role::ProjectRuntime,
                project_id: Some(ProjectId(project_id.into())),
            }],
        }
    }

    pub fn resolved_project(&self) -> Option<&ProjectId> {
        self.assignments.iter().find_map(|a| a.project_id.as_ref())
    }
}

#[async_trait]
pub trait Authenticator {
    async fn authenticate_bearer(&self, token: &str) -> Option<AuthContext>;
}

pub trait Authorizer {
    fn is_allowed(
        &self,
        ctx: &AuthContext,
        permission: Permission,
        project_id: Option<&ProjectId>,
    ) -> bool;
}

#[derive(Clone)]
pub struct AuthService {
    store: Option<Arc<Store>>,
    bootstrap_token: Option<String>,
    projects: Arc<DashMap<String, ProjectRecord>>,
    principals: Arc<DashMap<String, PrincipalRecord>>,
    tokens: Arc<DashMap<String, TokenRecord>>,
    bindings: Arc<DashMap<String, Vec<RoleBindingRecord>>>,
}

impl AuthService {
    pub fn new(store: Option<Arc<Store>>, bootstrap_token: Option<String>) -> Self {
        Self {
            store,
            bootstrap_token,
            projects: Arc::new(DashMap::new()),
            principals: Arc::new(DashMap::new()),
            tokens: Arc::new(DashMap::new()),
            bindings: Arc::new(DashMap::new()),
        }
    }

    pub async fn load_from_store(&self) -> Result<(), Box<dyn std::error::Error>> {
        let store = match &self.store {
            Some(store) => store,
            None => return Ok(()),
        };

        self.projects.clear();
        self.principals.clear();
        self.tokens.clear();
        self.bindings.clear();

        for record in store.get_all_projects().await? {
            self.projects.insert(record.project_id.clone(), record);
        }
        for record in store.get_all_principals().await? {
            self.principals.insert(record.principal_id.clone(), record);
        }
        for record in store.get_all_tokens(None).await? {
            self.tokens.insert(record.token_hash.clone(), record);
        }
        for binding in store.get_role_bindings(None, None).await? {
            self.bindings
                .entry(binding.principal_id.clone())
                .or_default()
                .push(binding);
        }

        Ok(())
    }

    pub async fn ensure_project(
        &self,
        project_id: &str,
        name: &str,
        description: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.projects.contains_key(project_id) {
            return Ok(());
        }

        let record = ProjectRecord {
            project_id: project_id.to_string(),
            name: name.to_string(),
            description,
            active: true,
            created_at: current_timestamp_string(),
        };
        if let Some(store) = &self.store {
            store.upsert_project(&record).await?;
        }
        self.projects.insert(record.project_id.clone(), record);
        Ok(())
    }

    pub fn list_projects(&self) -> Vec<ProjectRecord> {
        self.projects
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn list_principals(&self) -> Vec<PrincipalRecord> {
        self.principals
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn get_principal(&self, principal_id: &str) -> Option<PrincipalRecord> {
        self.principals
            .get(principal_id)
            .map(|entry| entry.value().clone())
    }

    pub fn list_role_bindings(
        &self,
        principal_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Vec<RoleBindingRecord> {
        let mut all = Vec::new();
        for binding_list in self.bindings.iter() {
            if let Some(expected_principal) = principal_id {
                if binding_list.key() != expected_principal {
                    continue;
                }
            }
            all.extend(
                binding_list
                    .value()
                    .iter()
                    .filter(|binding| {
                        if let Some(expected_project) = project_id {
                            binding.project_id.as_deref() == Some(expected_project)
                        } else {
                            true
                        }
                    })
                    .cloned(),
            );
        }
        all
    }

    pub fn accessible_projects(&self, ctx: &AuthContext) -> Vec<ProjectId> {
        if ctx
            .assignments
            .iter()
            .any(|assignment| assignment.role == Role::InstanceAdmin)
        {
            return self
                .projects
                .iter()
                .map(|entry| ProjectId(entry.key().clone()))
                .collect();
        }

        let mut projects = Vec::new();
        for assignment in &ctx.assignments {
            if let Some(project_id) = &assignment.project_id {
                if !projects
                    .iter()
                    .any(|existing: &ProjectId| existing == project_id)
                {
                    projects.push(project_id.clone());
                }
            }
        }
        projects
    }

    fn assignments_for_principal(&self, principal_id: &str) -> Vec<RoleAssignment> {
        self.bindings
            .get(principal_id)
            .map(|entry| {
                entry
                    .iter()
                    .filter_map(|binding| {
                        let role = Role::parse(&binding.role)?;
                        Some(RoleAssignment {
                            role,
                            project_id: binding.project_id.clone().map(ProjectId),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    pub fn build_auth_context(&self, principal_id: &str) -> Option<AuthContext> {
        let principal = self.principals.get(principal_id)?;
        if !principal.active {
            return None;
        }

        Some(AuthContext {
            principal_id: PrincipalId(principal.principal_id.clone()),
            source: AuthSource::ControlPlaneToken,
            assignments: self.assignments_for_principal(principal_id),
        })
    }

    pub async fn create_project(
        &self,
        project_id: &str,
        name: &str,
        description: Option<String>,
    ) -> Result<ProjectRecord, Box<dyn std::error::Error>> {
        let record = ProjectRecord {
            project_id: project_id.to_string(),
            name: name.to_string(),
            description,
            active: true,
            created_at: current_timestamp_string(),
        };
        if let Some(store) = &self.store {
            store.upsert_project(&record).await?;
        }
        self.projects
            .insert(record.project_id.clone(), record.clone());
        Ok(record)
    }

    pub async fn upsert_project(
        &self,
        record: ProjectRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            store.upsert_project(&record).await?;
        }
        self.projects.insert(record.project_id.clone(), record);
        Ok(())
    }

    pub async fn delete_project(
        &self,
        project_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            let deleted = store.delete_project(project_id).await?;
            self.projects.remove(project_id);
            return Ok(deleted);
        }
        Ok(self.projects.remove(project_id).is_some())
    }

    pub async fn create_principal(
        &self,
        name: &str,
    ) -> Result<PrincipalRecord, Box<dyn std::error::Error>> {
        let record = PrincipalRecord {
            principal_id: generate_id("prn"),
            name: name.to_string(),
            active: true,
            created_at: current_timestamp_string(),
        };
        if let Some(store) = &self.store {
            store.upsert_principal(&record).await?;
        }
        self.principals
            .insert(record.principal_id.clone(), record.clone());
        Ok(record)
    }

    pub async fn upsert_principal(
        &self,
        record: PrincipalRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            store.upsert_principal(&record).await?;
        }
        self.principals.insert(record.principal_id.clone(), record);
        Ok(())
    }

    pub async fn delete_principal(
        &self,
        principal_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            let deleted = store.delete_principal(principal_id).await?;
            self.principals.remove(principal_id);
            self.bindings.remove(principal_id);
            return Ok(deleted);
        }
        self.bindings.remove(principal_id);
        Ok(self.principals.remove(principal_id).is_some())
    }

    pub async fn create_token(
        &self,
        principal_id: &str,
        name: &str,
    ) -> Result<(String, TokenRecord), Box<dyn std::error::Error>> {
        let plaintext = generate_plaintext_token();
        let token_hash = hash_token(&plaintext);
        let record = TokenRecord {
            token_hash: token_hash.clone(),
            principal_id: principal_id.to_string(),
            name: name.to_string(),
            active: true,
            created_at: current_timestamp_string(),
        };
        if let Some(store) = &self.store {
            store.upsert_token(&record).await?;
        }
        self.tokens.insert(token_hash, record.clone());
        Ok((plaintext, record))
    }

    pub fn list_tokens(&self, principal_id: Option<&str>) -> Vec<TokenRecord> {
        self.tokens
            .iter()
            .filter(|entry| {
                principal_id
                    .map(|expected| entry.value().principal_id == expected)
                    .unwrap_or(true)
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn requires_auth(&self) -> bool {
        self.bootstrap_token.is_some() || !self.tokens.is_empty()
    }

    pub fn has_admin_access_path(&self) -> bool {
        if self.bootstrap_token.is_some() {
            return true;
        }

        self.tokens.iter().any(|token_entry| {
            let token = token_entry.value();
            if !token.active {
                return false;
            }

            let principal = match self.principals.get(&token.principal_id) {
                Some(principal) => principal,
                None => return false,
            };
            if !principal.active {
                return false;
            }

            self.bindings
                .get(&token.principal_id)
                .map(|bindings| {
                    bindings.iter().any(|binding| {
                        binding.project_id.is_none() && binding.role == Role::InstanceAdmin.as_str()
                    })
                })
                .unwrap_or(false)
        })
    }

    pub async fn delete_token(&self, token_hash: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            let deleted = store.delete_token(token_hash).await?;
            self.tokens.remove(token_hash);
            return Ok(deleted);
        }
        Ok(self.tokens.remove(token_hash).is_some())
    }

    pub async fn create_role_binding(
        &self,
        principal_id: &str,
        role: Role,
        project_id: Option<String>,
    ) -> Result<RoleBindingRecord, Box<dyn std::error::Error>> {
        let record = RoleBindingRecord {
            binding_id: generate_id("rb"),
            principal_id: principal_id.to_string(),
            role: role.as_str().to_string(),
            project_id,
            created_at: current_timestamp_string(),
        };
        if let Some(store) = &self.store {
            store.upsert_role_binding(&record).await?;
        }
        self.bindings
            .entry(record.principal_id.clone())
            .or_default()
            .push(record.clone());
        Ok(record)
    }

    pub async fn delete_role_binding(
        &self,
        binding_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            let deleted = store.delete_role_binding(binding_id).await?;
            if deleted {
                self.remove_binding(binding_id);
            }
            return Ok(deleted);
        }
        Ok(self.remove_binding(binding_id))
    }

    fn remove_binding(&self, binding_id: &str) -> bool {
        let keys: Vec<String> = self
            .bindings
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            if let Some(mut entry) = self.bindings.get_mut(&key) {
                let before = entry.len();
                entry.retain(|binding| binding.binding_id != binding_id);
                if entry.len() != before {
                    return true;
                }
            }
        }
        false
    }
}

#[async_trait]
impl Authenticator for AuthService {
    async fn authenticate_bearer(&self, token: &str) -> Option<AuthContext> {
        if self.bootstrap_token.as_deref() == Some(token) {
            return Some(AuthContext {
                principal_id: PrincipalId("bootstrap-admin".to_string()),
                source: AuthSource::Bootstrap,
                assignments: vec![RoleAssignment {
                    role: Role::InstanceAdmin,
                    project_id: None,
                }],
            });
        }

        let token_hash = hash_token(token);
        let token_record = self.tokens.get(&token_hash)?;
        if !token_record.active {
            return None;
        }

        let principal = self.principals.get(&token_record.principal_id)?;
        if !principal.active {
            return None;
        }

        let assignments = self.assignments_for_principal(&token_record.principal_id);

        if assignments.is_empty() {
            return None;
        }

        Some(AuthContext {
            principal_id: PrincipalId(principal.principal_id.clone()),
            source: AuthSource::ControlPlaneToken,
            assignments,
        })
    }
}

impl Authorizer for AuthService {
    fn is_allowed(
        &self,
        ctx: &AuthContext,
        permission: Permission,
        project_id: Option<&ProjectId>,
    ) -> bool {
        for assignment in &ctx.assignments {
            if assignment.role == Role::InstanceAdmin {
                return true;
            }

            if let Some(target_project) = project_id {
                if assignment.project_id.as_ref() != Some(target_project) {
                    continue;
                }
            } else if assignment.project_id.is_some()
                && matches!(
                    permission,
                    Permission::ManageProjects
                        | Permission::ManagePrincipals
                        | Permission::ManageRoleBindings
                )
            {
                continue;
            }

            if role_allows(&assignment.role, &permission) {
                return true;
            }
        }
        false
    }
}

fn role_allows(role: &Role, permission: &Permission) -> bool {
    match role {
        Role::InstanceAdmin => true,
        Role::ProjectAdmin => matches!(
            permission,
            Permission::ViewStatus
                | Permission::ViewProjects
                | Permission::ViewPrincipals
                | Permission::ViewRoleBindings
                | Permission::ManagePrincipals
                | Permission::ManageRoleBindings
                | Permission::ViewRuntimeKeys
                | Permission::ManageRuntimeKeys
                | Permission::ViewProjectPolicy
                | Permission::ManageProjectPolicy
                | Permission::ViewRoutingRules
                | Permission::ManageRoutingRules
                | Permission::ViewUsage
                | Permission::ManageUsage
                | Permission::ViewLogs
                | Permission::ViewProviders
                | Permission::ManageProviders
                | Permission::ViewRateLimiter
                | Permission::ViewModelCosts
                | Permission::ManageModelCosts
        ),
        Role::ProjectOperator => matches!(
            permission,
            Permission::ViewStatus
                | Permission::ViewProjects
                | Permission::ViewRuntimeKeys
                | Permission::ManageRuntimeKeys
                | Permission::ViewProjectPolicy
                | Permission::ManageProjectPolicy
                | Permission::ViewRoutingRules
                | Permission::ManageRoutingRules
                | Permission::ViewUsage
                | Permission::ManageUsage
                | Permission::ViewLogs
                | Permission::ViewProviders
                | Permission::ManageProviders
                | Permission::ViewRateLimiter
                | Permission::ViewModelCosts
        ),
        Role::ProjectViewer => matches!(
            permission,
            Permission::ViewStatus
                | Permission::ViewProjects
                | Permission::ViewRuntimeKeys
                | Permission::ViewProjectPolicy
                | Permission::ViewRoutingRules
                | Permission::ViewUsage
                | Permission::ViewLogs
                | Permission::ViewProviders
                | Permission::ViewRateLimiter
                | Permission::ViewModelCosts
        ),
        Role::ProjectRuntime => permission == &Permission::InvokeInference,
    }
}

pub fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

fn generate_id(prefix: &str) -> String {
    let mut rng = rand::thread_rng();
    let mut random_bytes = [0u8; 12];
    rng.fill(&mut random_bytes);
    format!("{}-{}", prefix, hex::encode(random_bytes))
}

fn generate_plaintext_token() -> String {
    let mut rng = rand::thread_rng();
    let mut random_bytes = [0u8; 32];
    rng.fill(&mut random_bytes);
    format!("sk-ctrl-{}", hex::encode(random_bytes))
}

fn current_timestamp_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store;

    #[tokio::test]
    async fn bootstrap_admin_authenticates() {
        let service = AuthService::new(None, Some("bootstrap-secret".to_string()));
        let ctx = service
            .authenticate_bearer("bootstrap-secret")
            .await
            .expect("bootstrap auth");
        assert!(service.is_allowed(&ctx, Permission::ManageProjects, None));
    }

    #[tokio::test]
    async fn sqlite_auth_round_trip() {
        let store = store::connect("sqlite:file:proxy_auth?mode=memory&cache=shared")
            .await
            .expect("store");
        let service = AuthService::new(Some(Arc::new(store)), None);
        service
            .create_project("alpha", "Alpha", Some("project".to_string()))
            .await
            .unwrap();
        let principal = service.create_principal("alice").await.unwrap();
        let (_plaintext, token) = service
            .create_token(&principal.principal_id, "cli")
            .await
            .unwrap();
        service
            .create_role_binding(
                &principal.principal_id,
                Role::ProjectAdmin,
                Some("alpha".to_string()),
            )
            .await
            .unwrap();

        let store = service.store.as_ref().unwrap();
        let reload = AuthService::new(Some(Arc::clone(store)), None);
        reload.load_from_store().await.unwrap();
        let ctx = reload.authenticate_bearer("not-real").await;
        assert!(ctx.is_none());
        let assignments = reload
            .bindings
            .get(&principal.principal_id)
            .expect("bindings");
        assert_eq!(assignments.len(), 1);
        assert!(reload.tokens.contains_key(&token.token_hash));
    }
}
