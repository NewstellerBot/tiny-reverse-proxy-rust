// Mock router for testing purposes

#[cfg(test)]
mod tests {
    use std::{net::Shutdown, str, time::Duration};
    use tiny_reverse_proxy_rust::{
        handlers::{client::handle_client_request, rewrite::rewrite_request},
        router::PathResolution,
    };

    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::thread;

    // Mock router for testing purposes
    struct MockRouter {
        mock_addresses: Vec<String>,
    }

    impl PathResolution for MockRouter {
        fn get_servers(&self, path: &str) -> Option<&Vec<String>> {
            if path == "/test" {
                Some(&self.mock_addresses)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_rewrite_request() {
        let original_request =
            b"GET / HTTP/1.1\r\nHost: oldhost.com\r\nUser-Agent: TestAgent\r\n\r\n";
        let new_host = "newhost.com";

        let rewritten = rewrite_request(original_request, new_host);
        let rewritten_str =
            str::from_utf8(&rewritten).expect("Rewritten request should be valid UTF-8");

        // Assert that the request line remains unchanged
        assert!(rewritten_str.starts_with("GET / HTTP/1.1\r\n"));

        // Assert that the old Host header is replaced
        assert!(rewritten_str.contains("Host: newhost.com\r\n"));
        assert!(!rewritten_str.contains("Host: oldhost.com"));

        // Assert that the User-Agent header remains
        assert!(rewritten_str.contains("User-Agent: TestAgent\r\n"));

        // Assert that the request ends with \r\n\r\n
        assert!(
            rewritten_str.ends_with("\r\n\r\n"),
            "Request should end with double CRLF"
        );
    }

    #[test]
    fn test_rewrite_request_no_existing_host() {
        let original_request = b"GET /test HTTP/1.1\r\nUser-Agent: TestAgent\r\n\r\n";
        let new_host = "example.org";

        let rewritten = rewrite_request(original_request, new_host);
        let rewritten_str = std::str::from_utf8(&rewritten).expect("UTF-8 decode error");

        // The Host header should be inserted after the request line
        // Original lines:
        // GET /test HTTP/1.1
        // User-Agent: TestAgent
        //
        // After rewrite:
        // GET /test HTTP/1.1
        // Host: example.org
        // User-Agent: TestAgent
        //
        assert!(rewritten_str.starts_with(
            "GET /test HTTP/1.1\r\nHost: example.org\r\nUser-Agent: TestAgent\r\n\r\n"
        ));
    }

    #[test]
    fn test_rewrite_request_multiple_host_lines() {
        let original_request = b"GET / HTTP/1.1\r\nHost: oldhost.com\r\nX-Forwarded-For: 1.1.1.1\r\nHost: anotherhost.com\r\n\r\n";
        let new_host = "finalhost.com";

        let rewritten = rewrite_request(original_request, new_host);
        let rewritten_str = std::str::from_utf8(&rewritten).expect("UTF-8 decode error");

        // All Host headers should be removed except the newly inserted one
        assert!(rewritten_str.contains("Host: finalhost.com\r\n"));
        assert!(!rewritten_str.contains("Host: oldhost.com"));
        assert!(!rewritten_str.contains("Host: anotherhost.com"));
        assert!(rewritten_str.contains("X-Forwarded-For: 1.1.1.1\r\n"));
        assert!(rewritten_str.ends_with("\r\n\r\n"));
    }

    #[test]
    fn test_handle_client_request() {
        // Start a mock upstream server
        let upstream_listener =
            TcpListener::bind("127.0.0.1:0").expect("Failed to bind upstream server");
        let upstream_addr = upstream_listener.local_addr().unwrap().to_string();

        // Spawn a thread to handle the upstream server connection
        let upstream_thread = thread::spawn(move || {
            let (mut upstream_conn, _) =
                upstream_listener.accept().expect("Upstream accept failed");

            // Read the request from the forwarder
            let mut buffer = [0u8; 4096];
            let bytes_read = upstream_conn
                .read(&mut buffer)
                .expect("Failed to read from upstream conn");
            let received_request = &buffer[..bytes_read];

            // Check that we got the expected forwarded request (depends on what your request looks like)
            assert!(std::str::from_utf8(received_request)
                .unwrap()
                .contains("GET /test HTTP/1.1"));

            // Send a mock response back to the forwarder
            let response = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nHello";
            upstream_conn
                .write_all(response)
                .expect("Upstream write failed");
        });

        // Set up a mock router that returns the upstream server for the "/test" path
        let router = Arc::new(MockRouter {
            mock_addresses: vec![upstream_addr],
        });

        // Start a local listener to simulate the client connecting to the reverse proxy
        let client_listener =
            TcpListener::bind("127.0.0.1:0").expect("Failed to bind client listener");
        let client_addr = client_listener.local_addr().unwrap();

        // Spawn a thread to act as the client
        let client_thread = thread::spawn(move || {
            // Connect as a client to the handle_client_request function
            let mut client_stream = TcpStream::connect(client_addr).expect("Client connect failed");

            // Send a test HTTP request
            let request = b"GET /test HTTP/1.1\r\nHost: localhost\r\n\r\n";
            client_stream
                .write_all(request)
                .expect("Failed to write request");

            // Read the response from handle_client_request (which should be from the upstream server)
            let mut buffer = [0u8; 4096];
            let bytes_read = client_stream
                .read(&mut buffer)
                .expect("Failed to read from client");
            let response = &buffer[..bytes_read];

            // Check the response from the upstream is received by the client
            assert!(std::str::from_utf8(response)
                .unwrap()
                .contains("HTTP/1.1 200 OK"));
            assert!(std::str::from_utf8(response).unwrap().contains("Hello"));
        });

        // Accept the client connection and handle it
        let (client_connection, _) = client_listener
            .accept()
            .expect("Failed to accept client connection");
        let request_counter = Arc::new(AtomicUsize::new(0));

        // We need to override extract_path and forward_request_to_upstream if they are not in the same crate or we can rely on the real ones.
        // If extract_path is in a different module, ensure it's accessible. If not, use mock_extract_path above by temporarily redefining.
        // Similarly, ensure forward_request_to_upstream is accessible or mocked.

        // Call the function under test
        handle_client_request(client_connection, router, request_counter);

        // Wait for threads to finish
        upstream_thread.join().expect("Upstream thread panicked");
        client_thread.join().expect("Client thread panicked");
    }

    #[test]
    fn test_handle_client_request_no_data_read() {
        // Start a local listener to simulate the client connecting to the reverse proxy
        let client_listener =
            TcpListener::bind("127.0.0.1:0").expect("Failed to bind client listener");
        let client_addr = client_listener.local_addr().unwrap();

        // Set up a mock router that returns the upstream server for the "/test" path
        let upstream_listener =
            TcpListener::bind("127.0.0.1:0").expect("Failed to bind upstream server");
        let upstream_addr = upstream_listener.local_addr().unwrap().to_string();

        let router = Arc::new(MockRouter {
            mock_addresses: vec![upstream_addr.clone()],
        });

        let request_counter = Arc::new(AtomicUsize::new(0));

        // Spawn a thread to act as the client
        let client_thread = thread::spawn(move || {
            // Connect as a client to the handle_client_request function
            let mut client_stream = TcpStream::connect(client_addr).expect("Client connect failed");

            // Immediately shutdown the write side to signal end-of-stream without sending data
            client_stream
                .shutdown(Shutdown::Write)
                .expect("Failed to shutdown write");

            // wait a bit to ensure the server processes the shutdown
            thread::sleep(Duration::from_millis(100));

            // Attempt to read the response (expecting none)
            let mut buffer = [0u8; 4096];
            let bytes_read = client_stream
                .read(&mut buffer)
                .expect("Failed to read from client");
            let response = &buffer[..bytes_read];

            // Check that the response is empty
            assert!(
                response.is_empty(),
                "Expected empty response, got {:?}",
                response
            );
        });

        // Accept the client connection
        let (client_connection, _) = client_listener
            .accept()
            .expect("Failed to accept client connection");

        // Optionally, spawn a thread to handle the server side if needed
        let server_thread = thread::spawn(move || {
            handle_client_request(client_connection, router, request_counter);
        });

        // Wait for threads to finish
        client_thread.join().expect("Client thread panicked");
        server_thread.join().expect("Server thread panicked");
    }

    #[test]
    fn test_handle_client_request_no_route_found() {
        // Start a mock TCP listener
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener");
        let addr = listener.local_addr().unwrap();

        // Start a thread to act as the client
        let client_thread = thread::spawn(move || {
            // Connect to the listener
            let mut stream = TcpStream::connect(addr).expect("Failed to connect");

            // Send a request with a path that has no route in the mock router
            let request = b"GET /invalid_path HTTP/1.1\r\nHost: localhost\r\n\r\n";
            stream.write_all(request).expect("Failed to send request");

            // Wait for the response (should be an error response or no response)
            let mut buffer = [0u8; 4096];
            let bytes_read = stream
                .read(&mut buffer)
                .expect("Failed to read from stream");

            // Assert that we received no data (since no route was found)
            assert_eq!(bytes_read, 0);
        });

        // Accept the client connection and simulate handling
        let (client_connection, _) = listener
            .accept()
            .expect("Failed to accept client connection");
        let request_counter = Arc::new(AtomicUsize::new(0));

        // Call the function under test with an invalid path (router.get_servers returns None)
        handle_client_request(
            client_connection,
            Arc::new(MockRouter {
                mock_addresses: Vec::new(),
            }),
            request_counter,
        );

        client_thread.join().expect("Client thread panicked");
    }

    #[test]
    fn test_handle_client_request_tcp_connect_error() {
        // Start a mock TCP listener
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener");
        let addr = listener.local_addr().unwrap();

        // Start a thread to act as the client
        let client_thread = thread::spawn(move || {
            // Connect to the listener
            let mut stream = TcpStream::connect(addr).expect("Failed to connect");

            // Send a request with a path that has no route in the mock router
            let request = b"GET /invalid_path HTTP/1.1\r\nHost: localhost\r\n\r\n";
            stream.write_all(request).expect("Failed to send request");

            // Wait for the response (should be an error response or no response)
            let mut buffer = [0u8; 4096];
            let bytes_read = stream
                .read(&mut buffer)
                .expect("Failed to read from stream");

            // Assert that we received no data (since no route was found)
            assert_eq!(bytes_read, 0);
        });

        // Accept the client connection and simulate handling
        let (client_connection, _) = listener
            .accept()
            .expect("Failed to accept client connection");
        let request_counter = Arc::new(AtomicUsize::new(0));

        // Call the function under test with an invalid path (router.get_servers returns None)
        handle_client_request(
            client_connection,
            Arc::new(MockRouter {
                mock_addresses: vec!["127.0.01:1".to_string()],
            }),
            request_counter,
        );

        client_thread.join().expect("Client thread panicked");
    }
}
