use bytes::Bytes;
use hyper::header::{HeaderMap, CACHE_CONTROL, VARY};
use hyper::{Method, StatusCode};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct CachedResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    created: Instant,
    ttl: Duration,
}

impl CachedResponse {
    fn is_expired(&self) -> bool {
        self.created.elapsed() > self.ttl
    }
}

#[derive(Clone)]
pub struct ResponseCache {
    cache: Arc<Mutex<LruCache<String, CachedResponse>>>,
    default_ttl: Duration,
    max_entry_size: usize, // per-entry size limit
}

impl ResponseCache {
    pub fn new(max_size_mb: u64, default_ttl_secs: u64) -> Self {
        // Rough estimate: assume average entry is ~10KB, so max entries = max_size_mb * 1024 * 1024 / 10240
        let max_entries = ((max_size_mb * 1024 * 1024) / 10240).max(100) as usize;
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(max_entries).unwrap(),
            ))),
            default_ttl: Duration::from_secs(default_ttl_secs),
            max_entry_size: (max_size_mb as usize * 1024 * 1024) / 10, // 10% of total for single entry
        }
    }

    /// Build a cache key from method, path, and Vary headers.
    pub fn cache_key(
        method: &Method,
        path: &str,
        vary_headers: &HeaderMap,
        request_headers: &HeaderMap,
    ) -> String {
        let mut key = format!("{}:{}", method, path);
        // If the cached response had Vary headers, include those request header values in the key
        if let Some(vary) = vary_headers.get(VARY) {
            if let Ok(vary_str) = vary.to_str() {
                for header_name in vary_str.split(',') {
                    let name = header_name.trim().to_lowercase();
                    if name == "*" {
                        return format!("{}:*:uncacheable", key);
                    }
                    let val = request_headers
                        .get(name.as_str())
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    key.push_str(&format!(":{}={}", name, val));
                }
            }
        }
        key
    }

    /// Check if a request is cacheable (only GET/HEAD, check Cache-Control).
    pub fn is_cacheable_request(method: &Method, headers: &HeaderMap) -> bool {
        if !matches!(*method, Method::GET | Method::HEAD) {
            return false;
        }
        // Check for no-store in request
        if let Some(cc) = headers.get(CACHE_CONTROL).and_then(|v| v.to_str().ok()) {
            if cc.contains("no-store") || cc.contains("no-cache") {
                return false;
            }
        }
        true
    }

    /// Check if a response is cacheable.
    fn is_cacheable_response(status: StatusCode, headers: &HeaderMap) -> bool {
        if !status.is_success() {
            return false;
        }
        if let Some(cc) = headers.get(CACHE_CONTROL).and_then(|v| v.to_str().ok()) {
            if cc.contains("no-store") || cc.contains("private") {
                return false;
            }
        }
        true
    }

    /// Parse TTL from Cache-Control header (s-maxage > max-age > default).
    fn parse_ttl(&self, headers: &HeaderMap) -> Duration {
        if let Some(cc) = headers.get(CACHE_CONTROL).and_then(|v| v.to_str().ok()) {
            // Try s-maxage first
            if let Some(secs) = parse_max_age(cc, "s-maxage") {
                return Duration::from_secs(secs);
            }
            // Then max-age
            if let Some(secs) = parse_max_age(cc, "max-age") {
                return Duration::from_secs(secs);
            }
        }
        self.default_ttl
    }

    /// Get a cached response. Returns None if not found or expired.
    pub fn get(&self, key: &str) -> Option<(StatusCode, HeaderMap, Bytes)> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                cache.pop(key);
                return None;
            }
            return Some((entry.status, entry.headers.clone(), entry.body.clone()));
        }
        None
    }

    /// Resolve a cache entry for the given request, including Vary-based variants.
    pub fn get_for_request(
        &self,
        method: &Method,
        path: &str,
        request_headers: &HeaderMap,
    ) -> Option<(StatusCode, HeaderMap, Bytes)> {
        let base_key = format!("{}:{}", method, path);
        let variant_prefix = format!("{base_key}:");
        let mut cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get(&base_key) {
            if entry.is_expired() {
                cache.pop(&base_key);
            } else {
                return Some((entry.status, entry.headers.clone(), entry.body.clone()));
            }
        }

        let mut expired_keys = Vec::new();
        let mut matching_key = None;
        for (key, entry) in cache.iter() {
            if key != &base_key && !key.starts_with(&variant_prefix) {
                continue;
            }
            if entry.is_expired() {
                expired_keys.push(key.clone());
                continue;
            }

            let expected_key = Self::cache_key(method, path, &entry.headers, request_headers);
            if expected_key == *key {
                matching_key = Some(key.clone());
                break;
            }
        }

        for key in expired_keys {
            cache.pop(&key);
        }

        let matching_key = matching_key?;
        let entry = cache.get(&matching_key)?;
        Some((entry.status, entry.headers.clone(), entry.body.clone()))
    }

    /// Store a response in the cache.
    pub fn put(&self, key: String, status: StatusCode, headers: &HeaderMap, body: &Bytes) {
        if !Self::is_cacheable_response(status, headers) {
            return;
        }
        if body.len() > self.max_entry_size {
            return;
        }
        let ttl = self.parse_ttl(headers);
        let entry = CachedResponse {
            status,
            headers: headers.clone(),
            body: body.clone(),
            created: Instant::now(),
            ttl,
        };
        self.cache.lock().unwrap().put(key, entry);
    }
}

fn parse_max_age(cc: &str, directive: &str) -> Option<u64> {
    let prefix = format!("{}=", directive);
    for part in cc.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix(&prefix) {
            return rest.trim().parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    #[test]
    fn test_cache_hit_and_miss() {
        let cache = ResponseCache::new(1, 60);
        let headers = HeaderMap::new();
        let body = Bytes::from("hello world");

        cache.put("GET:/test".to_string(), StatusCode::OK, &headers, &body);

        // Cache hit
        let result = cache.get("GET:/test");
        assert!(result.is_some());
        let (status, _, cached_body) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(cached_body, body);

        // Cache miss
        let result = cache.get("GET:/other");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_no_store_not_cached() {
        let cache = ResponseCache::new(1, 60);
        let mut headers = HeaderMap::new();
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        let body = Bytes::from("secret");

        cache.put("GET:/secret".to_string(), StatusCode::OK, &headers, &body);

        // Should not be cached due to no-store
        let result = cache.get("GET:/secret");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_expired_entry() {
        let cache = ResponseCache::new(1, 0); // 0-second default TTL
        let headers = HeaderMap::new();
        let body = Bytes::from("ephemeral");

        cache.put("GET:/expire".to_string(), StatusCode::OK, &headers, &body);

        // Sleep briefly to ensure expiration
        std::thread::sleep(Duration::from_millis(10));

        let result = cache.get("GET:/expire");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_key_with_vary() {
        let mut vary_headers = HeaderMap::new();
        vary_headers.insert(VARY, HeaderValue::from_static("Accept-Encoding"));

        let mut req_headers_gzip = HeaderMap::new();
        req_headers_gzip.insert("accept-encoding", HeaderValue::from_static("gzip"));

        let mut req_headers_br = HeaderMap::new();
        req_headers_br.insert("accept-encoding", HeaderValue::from_static("br"));

        let key1 =
            ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req_headers_gzip);
        let key2 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req_headers_br);

        // Keys should differ based on the Vary header value
        assert_ne!(key1, key2);
        assert!(key1.contains("accept-encoding=gzip"));
        assert!(key2.contains("accept-encoding=br"));
    }

    #[test]
    fn test_get_for_request_returns_matching_vary_variant() {
        let cache = ResponseCache::new(1, 60);
        let mut vary_headers = HeaderMap::new();
        vary_headers.insert(VARY, HeaderValue::from_static("Accept-Encoding"));

        let mut gzip_req = HeaderMap::new();
        gzip_req.insert("accept-encoding", HeaderValue::from_static("gzip"));
        let gzip_key = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &gzip_req);
        cache.put(
            gzip_key,
            StatusCode::OK,
            &vary_headers,
            &Bytes::from("gzip-body"),
        );

        let mut br_req = HeaderMap::new();
        br_req.insert("accept-encoding", HeaderValue::from_static("br"));
        let br_key = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &br_req);
        cache.put(
            br_key,
            StatusCode::OK,
            &vary_headers,
            &Bytes::from("br-body"),
        );

        let (_, _, gzip_body) = cache
            .get_for_request(&Method::GET, "/page", &gzip_req)
            .expect("gzip variant should resolve");
        assert_eq!(gzip_body, Bytes::from("gzip-body"));

        let (_, _, br_body) = cache
            .get_for_request(&Method::GET, "/page", &br_req)
            .expect("br variant should resolve");
        assert_eq!(br_body, Bytes::from("br-body"));
    }

    #[test]
    fn test_only_get_head_cacheable() {
        let headers = HeaderMap::new();

        assert!(ResponseCache::is_cacheable_request(&Method::GET, &headers));
        assert!(ResponseCache::is_cacheable_request(&Method::HEAD, &headers));
        assert!(!ResponseCache::is_cacheable_request(
            &Method::POST,
            &headers
        ));
        assert!(!ResponseCache::is_cacheable_request(&Method::PUT, &headers));
        assert!(!ResponseCache::is_cacheable_request(
            &Method::DELETE,
            &headers
        ));
    }

    #[test]
    fn test_max_age_zero_expires_immediately() {
        // max-age=0 should result in a 0-second TTL, making the entry expire immediately
        let cache = ResponseCache::new(1, 300);
        let mut headers = HeaderMap::new();
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("max-age=0"));
        let body = Bytes::from("should expire");

        cache.put("GET:/zero".to_string(), StatusCode::OK, &headers, &body);

        // Even a tiny sleep should cause expiration with TTL=0
        std::thread::sleep(Duration::from_millis(1));
        let result = cache.get("GET:/zero");
        assert!(
            result.is_none(),
            "max-age=0 entry should expire immediately"
        );
    }

    #[test]
    fn test_negative_max_age_falls_back_to_default() {
        // max-age=-10 is malformed; parse::<u64> will fail, so default TTL is used
        let cache = ResponseCache::new(1, 60);
        let mut headers = HeaderMap::new();
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("max-age=-10"));
        let body = Bytes::from("negative ttl");

        cache.put("GET:/neg".to_string(), StatusCode::OK, &headers, &body);

        // Should still be cached (using default 60s TTL, not expired yet)
        let result = cache.get("GET:/neg");
        assert!(
            result.is_some(),
            "negative max-age should fall back to default TTL"
        );
    }

    #[test]
    fn test_vary_multiple_headers_different_keys() {
        // Vary: Accept-Encoding, Accept-Language should produce different cache keys
        // when different header values are present
        let mut vary_headers = HeaderMap::new();
        vary_headers.insert(
            VARY,
            HeaderValue::from_static("Accept-Encoding, Accept-Language"),
        );

        // Request 1: gzip + en
        let mut req1 = HeaderMap::new();
        req1.insert("accept-encoding", HeaderValue::from_static("gzip"));
        req1.insert("accept-language", HeaderValue::from_static("en"));

        // Request 2: gzip + fr (different language)
        let mut req2 = HeaderMap::new();
        req2.insert("accept-encoding", HeaderValue::from_static("gzip"));
        req2.insert("accept-language", HeaderValue::from_static("fr"));

        // Request 3: br + en (different encoding)
        let mut req3 = HeaderMap::new();
        req3.insert("accept-encoding", HeaderValue::from_static("br"));
        req3.insert("accept-language", HeaderValue::from_static("en"));

        let key1 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req1);
        let key2 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req2);
        let key3 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req3);

        // All keys should be different
        assert_ne!(
            key1, key2,
            "different Accept-Language should produce different keys"
        );
        assert_ne!(
            key1, key3,
            "different Accept-Encoding should produce different keys"
        );
        assert_ne!(key2, key3, "both headers differ, keys should differ");

        // Verify keys contain both header values
        assert!(key1.contains("accept-encoding=gzip"));
        assert!(key1.contains("accept-language=en"));
        assert!(key2.contains("accept-language=fr"));
        assert!(key3.contains("accept-encoding=br"));
    }

    #[test]
    fn test_vary_same_headers_same_key() {
        // Same Vary header values should produce identical cache keys
        let mut vary_headers = HeaderMap::new();
        vary_headers.insert(
            VARY,
            HeaderValue::from_static("Accept-Encoding, Accept-Language"),
        );

        let mut req = HeaderMap::new();
        req.insert("accept-encoding", HeaderValue::from_static("gzip"));
        req.insert("accept-language", HeaderValue::from_static("en"));

        let key1 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req);
        let key2 = ResponseCache::cache_key(&Method::GET, "/page", &vary_headers, &req);

        assert_eq!(
            key1, key2,
            "identical Vary headers should produce the same key"
        );
    }

    #[test]
    fn test_lru_eviction_oldest_entry() {
        // ResponseCache::new(1, 60) gives max_entries = 102
        let cache = ResponseCache::new(1, 60);
        let headers = HeaderMap::new();
        let body = Bytes::from("x");

        // Fill cache to capacity (102 entries)
        for i in 0..102 {
            cache.put(format!("GET:/entry{}", i), StatusCode::OK, &headers, &body);
        }

        // Verify the first entry is still present
        assert!(
            cache.get("GET:/entry0").is_some(),
            "entry0 should exist before eviction"
        );

        // Add one more entry to trigger eviction
        cache.put("GET:/overflow".to_string(), StatusCode::OK, &headers, &body);

        // The oldest entry (entry0) should have been evicted.
        // Note: we accessed entry0 via get() above, which promotes it in LRU order,
        // so entry1 (the actual least-recently-used) should be evicted instead.
        assert!(
            cache.get("GET:/entry0").is_some(),
            "entry0 was recently accessed, should survive"
        );
        assert!(
            cache.get("GET:/entry1").is_none(),
            "entry1 (true LRU) should be evicted"
        );

        // The new entry should be present
        assert!(
            cache.get("GET:/overflow").is_some(),
            "new entry should be present"
        );
    }

    #[test]
    fn test_lru_eviction_without_access() {
        // Verify that without any get() calls, the first inserted entry is evicted
        let cache = ResponseCache::new(1, 60);
        let headers = HeaderMap::new();
        let body = Bytes::from("x");

        // Fill cache to capacity (102 entries)
        for i in 0..102 {
            cache.put(format!("GET:/e{}", i), StatusCode::OK, &headers, &body);
        }

        // Add one more to trigger eviction (no prior get() calls)
        cache.put("GET:/extra".to_string(), StatusCode::OK, &headers, &body);

        // entry0 should be evicted (oldest, never accessed)
        assert!(
            cache.get("GET:/e0").is_none(),
            "oldest entry should be evicted"
        );
        // Most recent entries should remain
        assert!(
            cache.get("GET:/e101").is_some(),
            "recent entry should still exist"
        );
        assert!(
            cache.get("GET:/extra").is_some(),
            "newest entry should exist"
        );
    }

    #[tokio::test]
    async fn test_concurrent_cache_get_put_no_panic() {
        let cache = Arc::new(ResponseCache::new(1, 60));
        let mut handles = Vec::new();

        for i in 0..20 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                let headers = HeaderMap::new();
                let body = Bytes::from(format!("body-{}", i));
                let key = format!("GET:/concurrent{}", i % 5);

                // Rapidly alternate between put and get
                for _ in 0..100 {
                    cache.put(key.clone(), StatusCode::OK, &headers, &body);
                    let _ = cache.get(&key);
                }
            }));
        }

        // All tasks should complete without panics
        for handle in handles {
            handle
                .await
                .expect("concurrent cache task should not panic");
        }
    }
}
