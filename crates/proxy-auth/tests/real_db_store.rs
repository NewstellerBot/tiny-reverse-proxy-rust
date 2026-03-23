use std::sync::Arc;

use proxy_auth::service::{AuthService, Authenticator, Authorizer, Permission, ProjectId, Role};
use proxy_auth::store;

fn unique_id(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!("{prefix}-{nanos}")
}

async fn run_auth_round_trip(url: &str) {
    let store = Arc::new(store::connect(url).await.expect("connect store"));
    let service = AuthService::new(Some(Arc::clone(&store)), None);

    let project_id = unique_id("project");
    service
        .create_project(&project_id, "Alpha", Some("real db test".to_string()))
        .await
        .expect("create project");

    let principal = service.create_principal("alice").await.expect("principal");
    let (plaintext, token) = service
        .create_token(&principal.principal_id, "cli")
        .await
        .expect("token");
    service
        .create_role_binding(
            &principal.principal_id,
            Role::ProjectAdmin,
            Some(project_id.clone()),
        )
        .await
        .expect("binding");

    let reload = AuthService::new(Some(Arc::clone(&store)), None);
    reload.load_from_store().await.expect("reload");

    let ctx = reload
        .authenticate_bearer(&plaintext)
        .await
        .expect("auth context");
    assert_eq!(ctx.principal_id.0, principal.principal_id);
    assert!(
        reload.is_allowed(
            &ctx,
            Permission::ManageRuntimeKeys,
            Some(&ProjectId(project_id.clone()))
        ),
        "project admin should manage runtime keys in its project"
    );
    assert!(
        !reload.is_allowed(&ctx, Permission::ManageProjects, None),
        "project admin should not have instance-wide project management"
    );

    let bindings = reload.list_role_bindings(Some(&principal.principal_id), Some(&project_id));
    assert_eq!(bindings.len(), 1);
    let tokens = reload.list_tokens(Some(&principal.principal_id));
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].token_hash, token.token_hash);
}

#[tokio::test]
#[ignore = "requires TRP_TEST_POSTGRES_URL"]
async fn postgres_auth_round_trip() {
    let url = std::env::var("TRP_TEST_POSTGRES_URL").expect("TRP_TEST_POSTGRES_URL");
    run_auth_round_trip(&url).await;
}

#[tokio::test]
#[ignore = "requires TRP_TEST_MYSQL_URL"]
async fn mysql_auth_round_trip() {
    let url = std::env::var("TRP_TEST_MYSQL_URL").expect("TRP_TEST_MYSQL_URL");
    run_auth_round_trip(&url).await;
}
