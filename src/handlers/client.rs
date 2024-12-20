use std::io::Read;
use std::net::TcpStream;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use crate::handlers::upstream::forward_request_to_upstream;
use crate::router::PathResolution;
use crate::utils::extract_path;

pub fn handle_client_request(
    mut client_stream: TcpStream,
    router: Arc<impl PathResolution>,
    request_counter: Arc<AtomicUsize>,
) {
    let mut request_buffer = [0; 4096];

    if let Ok(bytes_read) = client_stream.read(&mut request_buffer) {
        let path = extract_path(&request_buffer);

        match router.get_servers(&path) {
            Some(servers) => {
                forward_request_to_upstream(
                    &mut client_stream,
                    &request_buffer[..bytes_read],
                    servers,
                    request_counter,
                );
            }
            None => {
                eprintln!("No route found for path: {}", path);
            }
        }
    }
}
