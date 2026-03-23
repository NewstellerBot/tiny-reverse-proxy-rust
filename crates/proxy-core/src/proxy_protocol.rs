use std::net::SocketAddr;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

/// Parse PROXY protocol header from a TCP stream.
/// Returns the real client address and the remaining bytes that were read ahead.
///
/// Supports PROXY protocol v1 (text) and v2 (binary).
pub async fn parse_proxy_protocol(
    stream: &mut TcpStream,
) -> Result<(SocketAddr, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    // Read the fixed prefix first so we can distinguish v1 vs v2 without
    // over-reading into the HTTP request bytes.
    let mut prefix = [0u8; 12];
    stream.read_exact(&mut prefix).await?;

    // PROXY protocol v2 signature: \r\n\r\n\0\r\nQUIT\n
    let v2_sig = b"\r\n\r\n\0\r\nQUIT\n";
    if prefix == v2_sig[..] {
        let mut header_rest = [0u8; 4];
        stream.read_exact(&mut header_rest).await?;

        let payload_len = u16::from_be_bytes([header_rest[2], header_rest[3]]) as usize;
        let mut payload = vec![0u8; payload_len];
        stream.read_exact(&mut payload).await?;

        let mut full = Vec::with_capacity(16 + payload_len);
        full.extend_from_slice(&prefix);
        full.extend_from_slice(&header_rest);
        full.extend_from_slice(&payload);
        return parse_v2(&full);
    }

    // PROXY protocol v1 always starts with "PROXY ".
    if &prefix[..6] != b"PROXY " {
        return Err("invalid proxy protocol header".into());
    }

    // Continue reading one byte at a time until CRLF to avoid consuming any
    // bytes beyond the v1 header.
    let mut header = Vec::with_capacity(108);
    header.extend_from_slice(&prefix);
    while header.len() < 107 {
        if header.ends_with(b"\r\n") {
            break;
        }
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).await?;
        header.push(byte[0]);
    }

    if !header.ends_with(b"\r\n") {
        return Err("no CRLF in proxy protocol v1 header".into());
    }

    parse_v1(&header)
}

fn parse_v1(buf: &[u8]) -> Result<(SocketAddr, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    let header_str = std::str::from_utf8(buf)?;
    let end = header_str
        .find("\r\n")
        .ok_or("no CRLF in proxy protocol v1 header")?;
    let header_line = &header_str[..end];
    let remaining = buf[end + 2..].to_vec();

    // Format: "PROXY TCP4 <src_ip> <dst_ip> <src_port> <dst_port>"
    let parts: Vec<&str> = header_line.split(' ').collect();
    if parts.len() < 6 || parts[0] != "PROXY" {
        return Err("invalid proxy protocol v1 header format".into());
    }

    let src_ip: std::net::IpAddr = parts[2].parse()?;
    let src_port: u16 = parts[4].parse()?;
    let addr = SocketAddr::new(src_ip, src_port);

    Ok((addr, remaining))
}

fn parse_v2(buf: &[u8]) -> Result<(SocketAddr, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    if buf.len() < 16 {
        return Err("proxy protocol v2 header too short".into());
    }

    let ver_cmd = buf[12];
    let family = buf[13];
    let len = u16::from_be_bytes([buf[14], buf[15]]) as usize;

    if ver_cmd & 0xF0 != 0x20 {
        return Err("invalid proxy protocol v2 version".into());
    }

    let header_len = 16 + len;
    if buf.len() < header_len {
        return Err("proxy protocol v2 header incomplete".into());
    }

    let remaining = buf[header_len..].to_vec();

    // Parse address based on family
    match family {
        0x11 => {
            // TCP over IPv4
            if len < 12 {
                return Err("proxy protocol v2 IPv4 address too short".into());
            }
            let src_ip = std::net::Ipv4Addr::new(buf[16], buf[17], buf[18], buf[19]);
            let src_port = u16::from_be_bytes([buf[24], buf[25]]);
            Ok((
                SocketAddr::new(std::net::IpAddr::V4(src_ip), src_port),
                remaining,
            ))
        }
        0x21 => {
            // TCP over IPv6
            if len < 36 {
                return Err("proxy protocol v2 IPv6 address too short".into());
            }
            let mut src_ip = [0u8; 16];
            src_ip.copy_from_slice(&buf[16..32]);
            let src_port = u16::from_be_bytes([buf[48], buf[49]]);
            let ip = std::net::Ipv6Addr::from(src_ip);
            Ok((
                SocketAddr::new(std::net::IpAddr::V6(ip), src_port),
                remaining,
            ))
        }
        _ => Err(format!("unsupported proxy protocol family: {:#x}", family).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn test_parse_v1_tcp4() {
        let header = b"PROXY TCP4 192.168.1.1 10.0.0.1 56324 443\r\nextra data";
        let (addr, remaining) = parse_v1(header).unwrap();
        assert_eq!(addr, "192.168.1.1:56324".parse::<SocketAddr>().unwrap());
        assert_eq!(remaining, b"extra data");
    }

    #[test]
    fn test_parse_v1_tcp6() {
        let header = b"PROXY TCP6 2001:db8::1 2001:db8::2 56324 443\r\n";
        let (addr, remaining) = parse_v1(header).unwrap();
        assert_eq!(addr, "[2001:db8::1]:56324".parse::<SocketAddr>().unwrap());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_v2_tcp4() {
        // Build a valid PROXY protocol v2 header for TCP over IPv4
        let mut buf = Vec::new();
        // Signature: \r\n\r\n\0\r\nQUIT\n
        buf.extend_from_slice(b"\r\n\r\n\0\r\nQUIT\n");
        // Version (0x2) + PROXY command (0x1) = 0x21
        buf.push(0x21);
        // Address family: AF_INET (0x1) + transport STREAM (0x1) = 0x11
        buf.push(0x11);
        // Address length: 12 bytes (4 src_ip + 4 dst_ip + 2 src_port + 2 dst_port)
        buf.extend_from_slice(&12u16.to_be_bytes());
        // Source IP: 192.168.1.100
        buf.extend_from_slice(&[192, 168, 1, 100]);
        // Destination IP: 10.0.0.1
        buf.extend_from_slice(&[10, 0, 0, 1]);
        // Source port: 12345
        buf.extend_from_slice(&12345u16.to_be_bytes());
        // Destination port: 443
        buf.extend_from_slice(&443u16.to_be_bytes());

        let (addr, remaining) = parse_v2(&buf).unwrap();
        assert_eq!(addr, "192.168.1.100:12345".parse::<SocketAddr>().unwrap());
        assert!(remaining.is_empty());
    }

    #[tokio::test]
    async fn parse_proxy_header_does_not_consume_http_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream
                .write_all(b"PROXY TCP4 203.0.113.10 10.0.0.1 54321 443\r\nGET / HTTP/1.1\r\n\r\n")
                .await
                .unwrap();
        });

        let (mut stream, _) = listener.accept().await.unwrap();
        let (real_addr, remaining) = parse_proxy_protocol(&mut stream).await.unwrap();
        assert_eq!(
            real_addr,
            "203.0.113.10:54321".parse::<SocketAddr>().unwrap()
        );
        assert!(remaining.is_empty());

        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"GET / HTTP/1.1\r\n\r\n");

        client.await.unwrap();
    }
}
