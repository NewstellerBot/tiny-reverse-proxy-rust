//! OpenTelemetry distributed tracing support (feature-gated).
//!
//! Enabled via `--features opentelemetry`. Uses OTLP exporter configured
//! through standard environment variables:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4317`)
//! - `OTEL_SERVICE_NAME` (default: `tiny-reverse-proxy`)

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::runtime;

/// Newtype wrapper for storing an OTEL-enrichable span in `ctx.extensions`.
#[derive(Clone)]
pub struct OtelSpan(pub tracing::Span);

/// Initialise the OTLP trace exporter and return a `tracing` layer.
///
/// The returned layer is generic over the subscriber `S`, so it can be
/// composed into any layered subscriber stack.
pub fn init_tracer<S>(
) -> tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "tiny-reverse-proxy".to_string());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()
        .expect("failed to build OTLP span exporter");

    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(opentelemetry_sdk::Resource::new(vec![
            opentelemetry::KeyValue::new("service.name", service_name),
        ]))
        .build();

    let tracer = provider.tracer("proxy-core");

    // Store the provider globally so `shutdown_tracer` can flush it.
    opentelemetry::global::set_tracer_provider(provider);

    tracing_opentelemetry::layer().with_tracer(tracer)
}

/// Flush pending spans and shut down the global tracer provider.
pub fn shutdown_tracer() {
    opentelemetry::global::shutdown_tracer_provider();
}
