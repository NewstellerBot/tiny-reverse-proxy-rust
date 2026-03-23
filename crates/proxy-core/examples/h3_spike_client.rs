use std::collections::BTreeMap;
use std::future::poll_fn;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::{Buf, Bytes};
use h3::client::SendRequest;
use h3_quinn::{quinn, OpenStreams};
use hyper::http::{Method, Request, Uri};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls_pki_types::pem::PemObject;

#[derive(Debug, Clone)]
struct Args {
    url: String,
    method: String,
    requests: usize,
    concurrency: usize,
    body_file: Option<String>,
    ca_cert: Option<String>,
    insecure: bool,
    server_name: String,
    timeout_ms: u64,
}

fn usage() -> &'static str {
    "Usage:
  h3_spike_client --url <https://host:port/path> --method <GET|POST> --requests <N> --concurrency <N> [--ca-cert <cert.pem> | --insecure] [--body-file <file>] [--server-name <name>] [--timeout-ms <ms>]"
}

fn parse_args() -> Result<Args, String> {
    let mut url = None;
    let mut method = Some("GET".to_string());
    let mut requests = Some(1usize);
    let mut concurrency = Some(1usize);
    let mut body_file = None;
    let mut ca_cert = None;
    let mut insecure = false;
    let mut server_name = Some("localhost".to_string());
    let mut timeout_ms = Some(5000u64);

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--url" => url = it.next(),
            "--method" => method = it.next(),
            "--requests" => {
                let raw = it.next().ok_or("--requests requires a value")?;
                requests = Some(
                    raw.parse::<usize>()
                        .map_err(|_| format!("invalid --requests: {raw}"))?,
                );
            }
            "--concurrency" => {
                let raw = it.next().ok_or("--concurrency requires a value")?;
                concurrency = Some(
                    raw.parse::<usize>()
                        .map_err(|_| format!("invalid --concurrency: {raw}"))?,
                );
            }
            "--body-file" => body_file = it.next(),
            "--ca-cert" => ca_cert = it.next(),
            "--insecure" => insecure = true,
            "--server-name" => server_name = it.next(),
            "--timeout-ms" => {
                let raw = it.next().ok_or("--timeout-ms requires a value")?;
                timeout_ms = Some(
                    raw.parse::<u64>()
                        .map_err(|_| format!("invalid --timeout-ms: {raw}"))?,
                );
            }
            "--help" | "-h" => return Err(usage().to_string()),
            other => {
                return Err(format!("unknown argument: {other}\n{}", usage()));
            }
        }
    }

    let args = Args {
        url: url.ok_or_else(|| format!("missing --url\n{}", usage()))?,
        method: method.ok_or_else(|| format!("missing --method\n{}", usage()))?,
        requests: requests.unwrap_or(1),
        concurrency: concurrency.unwrap_or(1).max(1),
        body_file,
        ca_cert,
        insecure,
        server_name: server_name.unwrap_or_else(|| "localhost".to_string()),
        timeout_ms: timeout_ms.unwrap_or(5000),
    };

    if args.requests == 0 {
        return Err("--requests must be > 0".to_string());
    }
    if args.timeout_ms == 0 {
        return Err("--timeout-ms must be > 0".to_string());
    }
    if !args.insecure && args.ca_cert.is_none() {
        return Err(format!(
            "missing --ca-cert (or use --insecure)\n{}",
            usage()
        ));
    }

    Ok(args)
}

fn percentile_ms(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

fn parse_target(uri: &Uri) -> Result<SocketAddr, Box<dyn std::error::Error>> {
    let host = uri.host().ok_or("URL host is missing")?;
    let port = uri.port_u16().unwrap_or(443);
    let mut addrs = (host, port).to_socket_addrs()?;
    addrs
        .next()
        .ok_or_else(|| "target host resolved to no addresses".into())
}

fn load_roots_from_pem(path: &str) -> Result<rustls::RootCertStore, Box<dyn std::error::Error>> {
    let mut roots = rustls::RootCertStore::empty();
    for cert in CertificateDer::pem_file_iter(path)? {
        roots.add(cert?)?;
    }
    Ok(roots)
}

#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
    }
}

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

async fn single_request(
    mut send: SendRequest<OpenStreams, Bytes>,
    method: Method,
    uri: Uri,
    body: Bytes,
) -> Result<u16, Box<dyn std::error::Error + Send + Sync>> {
    let req = Request::builder().method(method).uri(uri).body(())?;
    let mut stream = send.send_request(req).await?;

    if !body.is_empty() {
        stream.send_data(body).await?;
    }

    stream.finish().await?;
    let resp = stream.recv_response().await?;
    let status = resp.status().as_u16();

    while let Some(chunk) = stream.recv_data().await? {
        let mut chunk = chunk;
        while chunk.has_remaining() {
            let _ = chunk.chunk();
            let remaining = chunk.remaining();
            chunk.advance(remaining);
        }
    }

    Ok(status)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("argument error: {e}"))?;
    let uri: Uri = args.url.parse()?;
    let method: Method = args.method.parse()?;
    let target_addr = parse_target(&uri)?;

    let tls_builder = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])?;
    let mut tls = if args.insecure {
        tls_builder
            .dangerous()
            .with_custom_certificate_verifier(SkipServerVerification::new())
            .with_no_client_auth()
    } else {
        let roots = load_roots_from_pem(args.ca_cert.as_deref().unwrap())?;
        tls_builder
            .with_root_certificates(roots)
            .with_no_client_auth()
    };
    tls.enable_early_data = true;
    tls.alpn_protocols = vec![b"h3".to_vec()];

    let mut client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(tls)?,
    ));
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    client_config.transport_config(Arc::new(transport));

    let bind_addr: SocketAddr = if target_addr.is_ipv4() {
        "0.0.0.0:0".parse()?
    } else {
        "[::]:0".parse()?
    };
    let mut endpoint = quinn::Endpoint::client(bind_addr)?;
    endpoint.set_default_client_config(client_config);

    let connecting = endpoint.connect(target_addr, &args.server_name)?;
    let conn = connecting.await?;
    let (driver, sender) = h3::client::builder()
        .build(h3_quinn::Connection::new(conn))
        .await?;

    let driver_handle = tokio::spawn(async move {
        let mut driver = driver;
        let _ = poll_fn(|cx| driver.poll_close(cx)).await;
    });

    let body = if let Some(path) = args.body_file.as_deref() {
        Bytes::from(tokio::fs::read(path).await?)
    } else {
        Bytes::new()
    };

    let timeout = Duration::from_millis(args.timeout_ms);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(args.concurrency));

    let mut tasks = Vec::with_capacity(args.requests);
    for _ in 0..args.requests {
        let permit = Arc::clone(&semaphore).acquire_owned().await?;
        let send = sender.clone();
        let req_uri = uri.clone();
        let req_method = method.clone();
        let req_body = body.clone();

        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            let outcome = tokio::time::timeout(
                timeout,
                single_request(send.clone(), req_method, req_uri, req_body),
            )
            .await;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

            match outcome {
                Ok(Ok(status)) => (status.to_string(), elapsed_ms),
                Ok(Err(_)) => ("ERR".to_string(), elapsed_ms),
                Err(_) => ("TIMEOUT".to_string(), elapsed_ms),
            }
        }));
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut latencies_ms = Vec::with_capacity(args.requests);

    for task in tasks {
        let (status, elapsed_ms) = task.await?;
        *counts.entry(status).or_insert(0) += 1;
        latencies_ms.push(elapsed_ms);
    }

    // Dropping sender eventually closes the H3 connection. The driver may still
    // be blocked on connection shutdown; abort it since metrics are already captured.
    drop(sender);
    driver_handle.abort();

    let p50 = percentile_ms(&latencies_ms, 50.0);
    let p95 = percentile_ms(&latencies_ms, 95.0);

    println!("TOTAL {}", args.requests);
    for (status, count) in counts {
        println!("STATUS {} {}", status, count);
    }
    println!("LAT_P50_MS {:.3}", p50);
    println!("LAT_P95_MS {:.3}", p95);

    Ok(())
}
