use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use semantic_safety_protocol::SemanticSafetyServiceServer;
use semantic_safety_service::backend::{InferenceBackend, TensorRtBackend};
use semantic_safety_service::metrics::{start_metrics_server, SemanticSafetyServiceMetrics};
use semantic_safety_service::persistence::FileProjectIndexStore;
use semantic_safety_service::service::{SemanticSafetyConfig, SemanticSafetyGrpcService};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "semantic_safety_service=info".into()),
        )
        .init();

    let addr = std::env::var("SEMANTIC_SAFETY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50061".to_string())
        .parse()?;
    let auth_token = std::env::var("SEMANTIC_SAFETY_AUTH_TOKEN").ok();
    let data_dir = PathBuf::from(
        std::env::var("SEMANTIC_SAFETY_DATA_DIR")
            .unwrap_or_else(|_| "./semantic-safety-data".to_string()),
    );
    let metrics_addr = std::env::var("SEMANTIC_SAFETY_METRICS_ADDR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<SocketAddr>())
        .transpose()?;

    let store = Arc::new(FileProjectIndexStore::new(data_dir)?);
    let backend = Arc::new(TensorRtBackend::from_env()?);
    let backend_name = backend.backend_name();
    let metrics = Arc::new(SemanticSafetyServiceMetrics::new());
    let service =
        SemanticSafetyGrpcService::new(SemanticSafetyConfig { auth_token }, store, backend)
            .with_metrics(Arc::clone(&metrics));
    service.load_from_store().await?;

    if let Some(metrics_addr) = metrics_addr {
        let metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            if let Err(error) = start_metrics_server(metrics_addr, metrics).await {
                tracing::error!(error = %error, "semantic safety metrics server failed");
            }
        });
    }

    tracing::info!(
        backend = backend_name,
        "semantic safety service listening on {}",
        addr
    );
    Server::builder()
        .add_service(SemanticSafetyServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
