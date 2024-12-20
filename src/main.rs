use std::env;
use std::net::TcpListener;
use std::process::exit;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

mod handlers;
mod router;
mod thread_pool;
mod utils;

use handlers::client::handle_client_request;
use router::Router;
use thread_pool::ThreadPool;
use utils::load_port_from_config;

const THREAD_POOL_SIZE: usize = 8;

#[cfg(not(tarpaulin_include))]
fn start_server(config_file: &str) -> std::io::Result<()> {
    let port = load_port_from_config(config_file).unwrap_or_else(|err| {
        eprintln!("Failed to load port from config: {}", err);
        exit(1);
    });
    let address = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&address)?;

    println!("Listening on {}...", address);

    let router = Arc::new(Router::load_from_file(config_file).expect("Failed to load routes"));
    let request_counter = Arc::new(AtomicUsize::new(0));
    let thread_pool = ThreadPool::new(THREAD_POOL_SIZE);

    for stream in listener.incoming() {
        match stream {
            Ok(client_stream) => {
                let router_clone = Arc::clone(&router);
                let counter_clone = Arc::clone(&request_counter);

                thread_pool.execute(move || {
                    handle_client_request(client_stream, router_clone, counter_clone);
                });
            }
            Err(e) => eprintln!("Failed to accept connection: {}", e),
        }
    }

    Ok(())
}

#[cfg(not(tarpaulin_include))]
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config file>", args[0]);
        exit(0);
    }

    if let Err(e) = start_server(&args[1]) {
        eprintln!("Server error: {}", e);
        exit(1);
    }
}
