use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::handlers::rewrite::rewrite_request;

pub fn forward_request_to_upstream(
    client_stream: &mut TcpStream,
    request: &[u8],
    servers: &[String],
    request_counter: Arc<AtomicUsize>,
) {
    let server_index = request_counter.fetch_add(1, Ordering::Relaxed) % servers.len();
    let upstream_server = &servers[server_index];

    match TcpStream::connect(upstream_server) {
        Ok(mut upstream_stream) => {
            let rewritten_request = rewrite_request(request, upstream_server);
            if upstream_stream.write_all(&rewritten_request).is_ok() {
                relay_response_to_client(client_stream, &mut upstream_stream);
            }
        }
        Err(e) => {
            eprintln!(
                "Failed to connect to upstream server {}: {}",
                upstream_server, e
            );
        }
    }
}

fn relay_response_to_client(client_stream: &mut TcpStream, upstream_stream: &mut TcpStream) {
    let mut response_buffer = [0; 4096];
    if let Ok(response_bytes) = upstream_stream.read(&mut response_buffer) {
        let _ = client_stream.write_all(&response_buffer[..response_bytes]);
    }
}
