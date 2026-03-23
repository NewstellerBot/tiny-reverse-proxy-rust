use async_compression::tokio::bufread::{BrotliEncoder, GzipEncoder, ZstdEncoder};
use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderMap, HeaderValue, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, VARY};
use hyper::Response;
use tokio::io::AsyncReadExt;

/// Content types that should NOT be compressed (already compressed or binary).
const SKIP_TYPES: &[&str] = &[
    "application/grpc",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/zip",
    "application/gzip",
    "application/x-bzip2",
    "application/zstd",
    "video/",
    "audio/",
];

/// Minimum body size in bytes below which compression is skipped.
const MIN_COMPRESS_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Encoding {
    Brotli,
    Zstd,
    Gzip,
}

/// Parse the `Accept-Encoding` header and pick the best encoding.
/// Preference order: brotli > zstd > gzip.
fn negotiate_encoding(accept_encoding: Option<&HeaderValue>) -> Option<Encoding> {
    fn rank(encoding: Encoding) -> u8 {
        match encoding {
            Encoding::Brotli => 3,
            Encoding::Zstd => 2,
            Encoding::Gzip => 1,
        }
    }

    let val = accept_encoding?.to_str().ok()?;
    let mut best: Option<(Encoding, f32)> = None;

    for token in val.split(',') {
        let mut parts = token.trim().split(';');
        let encoding_name = parts.next()?.trim().to_ascii_lowercase();
        if encoding_name.is_empty() {
            continue;
        }

        let mut q = 1.0_f32;
        for param in parts {
            let param = param.trim();
            if let Some(raw_q) = param.strip_prefix("q=") {
                // Invalid q-values are treated as not acceptable.
                q = raw_q.parse::<f32>().ok().unwrap_or(0.0);
            }
        }

        if q <= 0.0 {
            continue;
        }

        let encoding = match encoding_name.as_str() {
            "br" => Some(Encoding::Brotli),
            "zstd" => Some(Encoding::Zstd),
            "gzip" => Some(Encoding::Gzip),
            "*" => Some(Encoding::Brotli),
            _ => None,
        };

        if let Some(encoding) = encoding {
            match best {
                None => best = Some((encoding, q)),
                Some((current, current_q)) => {
                    if q > current_q || (q == current_q && rank(encoding) > rank(current)) {
                        best = Some((encoding, q));
                    }
                }
            }
        }
    }

    best.map(|(encoding, _)| encoding)
}

/// Returns `true` when the response should skip compression based on its headers.
fn should_skip_compression(headers: &HeaderMap) -> bool {
    // Already encoded upstream -- do not double-compress.
    if headers.contains_key(CONTENT_ENCODING) {
        return true;
    }

    if let Some(ct) = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()) {
        let ct_lower = ct.to_lowercase();
        return SKIP_TYPES.iter().any(|skip| ct_lower.starts_with(skip));
    }

    false
}

/// Helper to build a `BoxBody` from raw `Bytes` with the error type that the
/// rest of the proxy expects (`hyper::Error`).
fn full_body(data: Bytes) -> BoxBody<Bytes, hyper::Error> {
    Full::new(data).map_err(|never| match never {}).boxed()
}

fn ensure_vary_accept_encoding(headers: &mut HeaderMap) {
    let already_varies = headers
        .get(VARY)
        .and_then(|v| v.to_str().ok())
        .map(|vary| {
            vary.split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("accept-encoding"))
        })
        .unwrap_or(false);
    if already_varies {
        return;
    }

    if let Some(existing) = headers.get(VARY).and_then(|v| v.to_str().ok()) {
        if let Ok(updated) = HeaderValue::from_str(&format!("{existing}, Accept-Encoding")) {
            headers.insert(VARY, updated);
        }
    } else {
        headers.insert(VARY, HeaderValue::from_static("Accept-Encoding"));
    }
}

/// Compress the response body if the client supports it and the content type
/// is eligible.  The full body is collected into memory, compressed, and then
/// returned as a single `Full<Bytes>` body.  For a reverse proxy that already
/// buffers retryable responses this is acceptable.
pub async fn maybe_compress_response(
    accept_encoding: Option<HeaderValue>,
    resp: Response<BoxBody<Bytes, hyper::Error>>,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    let encoding = match negotiate_encoding(accept_encoding.as_ref()) {
        Some(e) => e,
        None => return resp,
    };

    if should_skip_compression(resp.headers()) {
        return resp;
    }

    let (mut parts, body) = resp.into_parts();

    // Collect the body into contiguous bytes.
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return Response::from_parts(parts, full_body(Bytes::new()));
        }
    };

    // Skip compression for very small bodies -- the overhead is not worth it.
    if body_bytes.len() < MIN_COMPRESS_SIZE {
        return Response::from_parts(parts, full_body(body_bytes));
    }

    // Compress through an async-compression encoder that implements AsyncRead.
    let compressed = match encoding {
        Encoding::Gzip => {
            let mut encoder = GzipEncoder::new(&body_bytes[..]);
            let mut buf = Vec::new();
            encoder.read_to_end(&mut buf).await.ok().map(|_| buf)
        }
        Encoding::Brotli => {
            let mut encoder = BrotliEncoder::new(&body_bytes[..]);
            let mut buf = Vec::new();
            encoder.read_to_end(&mut buf).await.ok().map(|_| buf)
        }
        Encoding::Zstd => {
            let mut encoder = ZstdEncoder::new(&body_bytes[..]);
            let mut buf = Vec::new();
            encoder.read_to_end(&mut buf).await.ok().map(|_| buf)
        }
    };

    match compressed {
        Some(data) => {
            let encoding_str = match encoding {
                Encoding::Brotli => "br",
                Encoding::Zstd => "zstd",
                Encoding::Gzip => "gzip",
            };
            parts.headers.remove(CONTENT_LENGTH);
            parts
                .headers
                .insert(CONTENT_ENCODING, HeaderValue::from_static(encoding_str));
            ensure_vary_accept_encoding(&mut parts.headers);
            Response::from_parts(parts, full_body(Bytes::from(data)))
        }
        None => {
            // Compression failed; return the original uncompressed body.
            Response::from_parts(parts, full_body(body_bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    // ---------------------------------------------------------------
    // negotiate_encoding
    // ---------------------------------------------------------------

    #[test]
    fn test_negotiate_encoding_brotli() {
        let val = HeaderValue::from_static("gzip, br, zstd");
        assert_eq!(negotiate_encoding(Some(&val)), Some(Encoding::Brotli));
    }

    #[test]
    fn test_negotiate_encoding_gzip() {
        let val = HeaderValue::from_static("gzip, deflate");
        assert_eq!(negotiate_encoding(Some(&val)), Some(Encoding::Gzip));
    }

    #[test]
    fn test_negotiate_encoding_respects_q_values() {
        let val = HeaderValue::from_static("gzip;q=1.0, br;q=0.3, zstd;q=0.2");
        assert_eq!(negotiate_encoding(Some(&val)), Some(Encoding::Gzip));
    }

    #[test]
    fn test_negotiate_encoding_q_zero_disables_encoding() {
        let val = HeaderValue::from_static("br;q=0, zstd;q=0, gzip;q=0");
        assert_eq!(negotiate_encoding(Some(&val)), None);
    }

    #[test]
    fn test_negotiate_encoding_wildcard() {
        let val = HeaderValue::from_static("*;q=0.9");
        assert_eq!(negotiate_encoding(Some(&val)), Some(Encoding::Brotli));
    }

    #[test]
    fn test_negotiate_encoding_none() {
        assert_eq!(negotiate_encoding(None), None);

        let val = HeaderValue::from_static("deflate");
        assert_eq!(negotiate_encoding(Some(&val)), None);
    }

    // ---------------------------------------------------------------
    // should_skip_compression
    // ---------------------------------------------------------------

    #[test]
    fn test_skip_compression_grpc() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/grpc"));
        assert!(should_skip_compression(&headers));
    }

    #[test]
    fn test_skip_already_encoded() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        assert!(should_skip_compression(&headers));
    }

    #[test]
    fn test_ensure_vary_accept_encoding_adds_header() {
        let mut headers = HeaderMap::new();
        ensure_vary_accept_encoding(&mut headers);
        assert_eq!(
            headers.get(VARY).and_then(|v| v.to_str().ok()),
            Some("Accept-Encoding")
        );
    }

    #[test]
    fn test_ensure_vary_accept_encoding_appends_once() {
        let mut headers = HeaderMap::new();
        headers.insert(VARY, HeaderValue::from_static("Accept-Language"));
        ensure_vary_accept_encoding(&mut headers);
        ensure_vary_accept_encoding(&mut headers);
        assert_eq!(
            headers.get(VARY).and_then(|v| v.to_str().ok()),
            Some("Accept-Language, Accept-Encoding")
        );
    }
}
