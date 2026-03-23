//! Fast upstream server for benchmarking. Compile with:
//!   rustc --edition 2021 -O bench/upstream_fast.rs -o bench/upstream_fast \
//!     $(pkg-config --libs --cflags openssl 2>/dev/null || true)
//! Or just use `cargo` — but this is standalone for simplicity.
//!
//! Instead, we'll run it via the workspace binary approach below.

use std::convert::Infallible;
use std::net::SocketAddr;

use bytes::Bytes;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

async fn handle(_req: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from(
        "Hello from upstream!\n",
    ))))
}

#[tokio::main]
async fn main() {
    let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let listener = TcpListener::bind(addr).await.unwrap();
    eprintln!("Fast upstream listening on {}", addr);

    loop {
        let (stream, _) = listener.accept().await.unwrap();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let _ = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service_fn(handle))
                .await;
        });
    }
}
