use dashmap::DashMap;
use std::hash::Hash;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    rate: f64,      // tokens per second
    max_burst: f64, // maximum tokens
}

impl TokenBucket {
    fn new(rate: f64, burst: f64) -> Self {
        Self {
            tokens: burst,
            last_refill: Instant::now(),
            rate,
            max_burst: burst,
        }
    }

    fn try_consume(&mut self, count: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.max_burst);
        self.last_refill = now;
        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Return current token level without consuming (updates refill state).
    fn peek_remaining(&mut self) -> f64 {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.max_burst);
        self.last_refill = now;
        self.tokens
    }
}

/// Generic token-bucket rate limiter keyed by `K`.
///
/// Each key gets its own bucket with the configured rate and burst.
/// Use [`try_consume`](Self::try_consume) for variable-cost operations or
/// [`check`](Self::check) as a convenience for consuming exactly 1 token.
#[derive(Clone)]
pub struct TokenBucketMap<K: Eq + Hash> {
    buckets: Arc<DashMap<K, TokenBucket>>,
    rate: f64,
    burst: f64,
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static> TokenBucketMap<K> {
    pub fn new(rate_per_second: f64, burst: f64) -> Self {
        Self {
            buckets: Arc::new(DashMap::new()),
            rate: rate_per_second,
            burst,
        }
    }

    /// Try to consume `count` tokens for the given key.
    /// Returns `true` if allowed, `false` if rate-limited.
    pub fn try_consume(&self, key: K, count: f64) -> bool {
        let mut entry = self
            .buckets
            .entry(key)
            .or_insert_with(|| TokenBucket::new(self.rate, self.burst));
        entry.try_consume(count)
    }

    /// Convenience for `try_consume(key, 1.0)`.
    pub fn check(&self, key: K) -> bool {
        self.try_consume(key, 1.0)
    }

    /// Remove entries that haven't been accessed recently.
    /// Call this periodically to prevent memory leaks.
    pub fn cleanup(&self) {
        let cutoff = Instant::now() - std::time::Duration::from_secs(300); // 5 min
        self.buckets.retain(|_, bucket| bucket.last_refill > cutoff);
    }

    /// Peek at remaining tokens for a key (updates refill, does not consume).
    /// Returns `None` if the key has no bucket yet.
    pub fn remaining_for_key(&self, key: &K) -> Option<f64> {
        self.buckets.get_mut(key).map(|mut b| b.peek_remaining())
    }

    /// Number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Rate per second for each bucket.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Burst size for each bucket.
    pub fn burst(&self) -> f64 {
        self.burst
    }

    /// Spawn a background cleanup task that runs every 60 seconds.
    pub fn spawn_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let map = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                map.cleanup();
            }
        })
    }
}

/// IP-based rate limiter (1 token per request).
#[derive(Clone)]
pub struct RateLimiter {
    inner: TokenBucketMap<IpAddr>,
}

impl RateLimiter {
    pub fn new(requests_per_second: f64, burst: u32) -> Self {
        Self {
            inner: TokenBucketMap::new(requests_per_second, burst as f64),
        }
    }

    /// Returns true if the request should be allowed, false if rate limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        self.inner.check(ip)
    }

    /// Remove entries that haven't been accessed recently.
    pub fn cleanup(&self) {
        self.inner.cleanup();
    }
}

/// Spawn a background cleanup task that runs every 60 seconds.
pub fn spawn_cleanup_task(limiter: RateLimiter) -> tokio::task::JoinHandle<()> {
    limiter.inner.spawn_cleanup_task()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn localhost() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
    }

    // --- RateLimiter (IP-based, consume-1) tests ---

    #[test]
    fn test_allows_within_burst() {
        let limiter = RateLimiter::new(10.0, 5);
        let ip = localhost();

        // All 5 burst requests should be allowed
        for i in 0..5 {
            assert!(
                limiter.check(ip),
                "request {} should be allowed within burst",
                i
            );
        }
    }

    #[test]
    fn test_rejects_over_burst() {
        let limiter = RateLimiter::new(10.0, 3);
        let ip = localhost();

        // Consume all 3 burst tokens
        for _ in 0..3 {
            assert!(limiter.check(ip));
        }

        // The next request should be rejected
        assert!(
            !limiter.check(ip),
            "request beyond burst should be rejected"
        );
    }

    #[test]
    fn test_refill_after_wait() {
        let limiter = RateLimiter::new(1000.0, 1);
        let ip = localhost();

        // Consume the single burst token
        assert!(limiter.check(ip));
        assert!(
            !limiter.check(ip),
            "should be rejected after burst exhausted"
        );

        // Wait a small amount of time for at least 1 token to refill
        // At 1000 tokens/sec, 2ms should refill ~2 tokens
        std::thread::sleep(std::time::Duration::from_millis(2));

        assert!(limiter.check(ip), "should be allowed after token refill");
    }

    #[test]
    fn test_cleanup_removes_stale_entries() {
        let limiter = RateLimiter::new(10.0, 5);
        let ip = localhost();

        // Generate an entry
        limiter.check(ip);
        assert!(
            limiter.inner.buckets.contains_key(&ip),
            "entry should exist before cleanup"
        );

        // Manually set last_refill to 10 minutes ago to simulate a stale entry
        {
            let mut entry = limiter.inner.buckets.get_mut(&ip).unwrap();
            entry.last_refill = Instant::now() - std::time::Duration::from_secs(600);
        }

        limiter.cleanup();
        assert!(
            !limiter.inner.buckets.contains_key(&ip),
            "stale entry should be removed after cleanup"
        );
    }

    // --- TokenBucketMap (generic, variable consumption) tests ---

    #[test]
    fn test_generic_variable_consumption() {
        let map: TokenBucketMap<String> = TokenBucketMap::new(1.0, 100.0);

        // Consume 60 of 100 burst tokens
        assert!(map.try_consume("key".into(), 60.0));
        // 40 remaining, try to consume 50 — should fail
        assert!(!map.try_consume("key".into(), 50.0));
        // 40 remaining, consume 30 — should succeed
        assert!(map.try_consume("key".into(), 30.0));
    }

    #[test]
    fn test_generic_separate_keys() {
        let map: TokenBucketMap<String> = TokenBucketMap::new(1.0, 10.0);

        // Exhaust key A
        assert!(map.try_consume("a".into(), 10.0));
        assert!(!map.try_consume("a".into(), 1.0));

        // Key B should be independent
        assert!(map.try_consume("b".into(), 5.0));
    }

    #[test]
    fn test_zero_rate_and_burst_rejects_all() {
        // Zero rate + zero burst means no tokens are ever available
        let map: TokenBucketMap<String> = TokenBucketMap::new(0.0, 0.0);

        // Should reject: can't consume 1.0 token from a 0.0 burst bucket
        assert!(
            !map.check("key".into()),
            "zero burst should reject consuming 1 token"
        );

        // Waiting shouldn't help since rate is 0
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            !map.check("key".into()),
            "zero rate should never refill tokens"
        );
    }

    #[test]
    fn test_zero_burst_nonzero_rate_rejects_initially() {
        // Zero burst but nonzero rate: bucket starts empty, tokens accumulate over time
        let map: TokenBucketMap<String> = TokenBucketMap::new(1000.0, 0.0);

        // Initially empty
        assert!(!map.check("key".into()), "zero burst should start empty");

        // After waiting, tokens should accumulate but be capped at max_burst (0.0)
        // So even with refill, min(accumulated, 0.0) = 0.0
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            !map.check("key".into()),
            "tokens should be capped at max_burst=0"
        );
    }

    #[test]
    fn test_rapid_successive_calls_no_panic() {
        // Rapid successive calls with near-zero time deltas should not panic
        // or produce NaN/infinity from floating-point math
        let map: TokenBucketMap<String> = TokenBucketMap::new(100.0, 10.0);

        // Consume most tokens
        assert!(map.try_consume("key".into(), 9.0));

        // Rapid calls with near-zero delta
        for _ in 0..1000 {
            let _ = map.check("key".into());
        }

        // Verify the bucket is still in a valid state by checking a known operation
        // At 100 tokens/s, microseconds of elapsed time won't refill much
        // The key thing is no panic/NaN occurred
        let _ = map.try_consume("key".into(), 0.5);
    }

    #[test]
    fn test_ip_rate_limiter_zero_burst() {
        // RateLimiter with burst=0 should reject all requests
        let limiter = RateLimiter::new(10.0, 0);
        let ip = localhost();

        assert!(!limiter.check(ip), "burst=0 should reject all requests");

        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            !limiter.check(ip),
            "burst=0 should still reject after wait (capped at 0)"
        );
    }
}
