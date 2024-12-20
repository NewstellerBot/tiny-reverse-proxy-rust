pub fn rewrite_request(request: &[u8], host: &str) -> Vec<u8> {
    let request_string = String::from_utf8_lossy(request);
    let mut lines = request_string.lines().collect::<Vec<&str>>();

    // Replace or add the Host header
    lines.retain(|line| !line.to_lowercase().starts_with("host: "));
    let host_header = format!("Host: {}", host);
    lines.insert(1, &host_header);

    // Ensure the request ends with two CRLF sequences
    lines.push("");
    lines.join("\r\n").into_bytes()
}
