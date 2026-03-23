use dashmap::DashMap;
use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
/// Trait for load balancing strategies.
pub trait LoadBalancer: Send + Sync {
    /// Pick the index of the next server from the given list of servers.
    fn pick(&self, servers: &[&String], key: Option<&str>) -> usize;
    /// Called when a request starts (for least-connections tracking).
    fn on_request_start(&self, server: &str);
    /// Called when a request ends (for least-connections tracking).
    fn on_request_end(&self, server: &str);
}

/// Simple round-robin load balancer.
pub struct RoundRobin {
    counter: AtomicUsize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for RoundRobin {
    fn pick(&self, servers: &[&String], _key: Option<&str>) -> usize {
        if servers.is_empty() {
            return 0;
        }
        self.counter.fetch_add(1, Ordering::Relaxed) % servers.len()
    }
    fn on_request_start(&self, _server: &str) {}
    fn on_request_end(&self, _server: &str) {}
}

/// Least-connections load balancer.
pub struct LeastConnections {
    connections: DashMap<String, AtomicU32>,
}

impl LeastConnections {
    pub fn new() -> Self {
        Self {
            connections: DashMap::new(),
        }
    }
}

impl Default for LeastConnections {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for LeastConnections {
    fn pick(&self, servers: &[&String], _key: Option<&str>) -> usize {
        if servers.is_empty() {
            return 0;
        }
        let mut min_idx = 0;
        let mut min_count = u32::MAX;
        for (i, server) in servers.iter().enumerate() {
            let count = self
                .connections
                .get(server.as_str())
                .map(|v| v.value().load(Ordering::Relaxed))
                .unwrap_or(0);
            if count < min_count {
                min_count = count;
                min_idx = i;
            }
        }
        min_idx
    }
    fn on_request_start(&self, server: &str) {
        self.connections
            .entry(server.to_string())
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
    fn on_request_end(&self, server: &str) {
        if let Some(entry) = self.connections.get(server) {
            entry.value().fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Consistent hash load balancer (hashes on key, typically client IP or path).
pub struct ConsistentHash;

impl ConsistentHash {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConsistentHash {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer for ConsistentHash {
    fn pick(&self, servers: &[&String], key: Option<&str>) -> usize {
        if servers.is_empty() {
            return 0;
        }
        let key = key.unwrap_or("default");
        let mut hasher = SipHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as usize) % servers.len()
    }
    fn on_request_start(&self, _server: &str) {}
    fn on_request_end(&self, _server: &str) {}
}

/// Weighted round-robin load balancer.
pub struct WeightedRoundRobin {
    counter: AtomicUsize,
    expanded: Vec<usize>, // expanded index list based on weights
}

impl WeightedRoundRobin {
    pub fn new(weights: Vec<u32>) -> Self {
        let mut expanded = Vec::new();
        for (i, &w) in weights.iter().enumerate() {
            for _ in 0..w {
                expanded.push(i);
            }
        }
        if expanded.is_empty() {
            expanded.push(0);
        }
        Self {
            counter: AtomicUsize::new(0),
            expanded,
        }
    }
}

impl LoadBalancer for WeightedRoundRobin {
    fn pick(&self, servers: &[&String], _key: Option<&str>) -> usize {
        if servers.is_empty() {
            return 0;
        }
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.expanded.len();
        self.expanded[idx] % servers.len()
    }
    fn on_request_start(&self, _server: &str) {}
    fn on_request_end(&self, _server: &str) {}
}

/// Create a load balancer from a strategy name.
pub fn create_load_balancer(
    strategy: &crate::config::LbStrategy,
    weights: Option<&[u32]>,
) -> Box<dyn LoadBalancer> {
    use crate::config::LbStrategy;
    match strategy {
        LbStrategy::RoundRobin => Box::new(RoundRobin::new()),
        LbStrategy::LeastConnections => Box::new(LeastConnections::new()),
        LbStrategy::ConsistentHash => Box::new(ConsistentHash::new()),
        LbStrategy::WeightedRoundRobin => {
            let w = weights.unwrap_or(&[1]).to_vec();
            Box::new(WeightedRoundRobin::new(w))
        }
    }
}

/// Create a shared load balancer instance.
pub fn create_load_balancer_arc(
    strategy: &crate::config::LbStrategy,
    weights: Option<&[u32]>,
) -> Arc<dyn LoadBalancer> {
    Arc::from(create_load_balancer(strategy, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_cycles() {
        let lb = RoundRobin::new();
        let s1 = String::from("a");
        let s2 = String::from("b");
        let s3 = String::from("c");
        let servers: Vec<&String> = vec![&s1, &s2, &s3];

        let picks: Vec<usize> = (0..9).map(|_| lb.pick(&servers, None)).collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_least_connections_favors_idle() {
        let lb = LeastConnections::new();
        let s1 = String::from("server-a");
        let s2 = String::from("server-b");
        let s3 = String::from("server-c");
        let servers: Vec<&String> = vec![&s1, &s2, &s3];

        // All idle: should pick the first one (index 0)
        assert_eq!(lb.pick(&servers, None), 0);

        // Add connections to server-a and server-b
        lb.on_request_start("server-a");
        lb.on_request_start("server-a");
        lb.on_request_start("server-b");

        // server-c has 0 connections, should be picked
        assert_eq!(lb.pick(&servers, None), 2);

        // End one connection on server-a, add one on server-c
        lb.on_request_end("server-a");
        lb.on_request_start("server-c");

        // Now: server-a=1, server-b=1, server-c=1 -- all tied, picks first
        assert_eq!(lb.pick(&servers, None), 0);

        // End all connections on server-b
        lb.on_request_end("server-b");

        // Now: server-a=1, server-b=0, server-c=1 -- picks server-b
        assert_eq!(lb.pick(&servers, None), 1);
    }

    #[test]
    fn test_consistent_hash_stable() {
        let lb = ConsistentHash::new();
        let s1 = String::from("x");
        let s2 = String::from("y");
        let s3 = String::from("z");
        let servers: Vec<&String> = vec![&s1, &s2, &s3];

        let key = "192.168.1.42";
        let first_pick = lb.pick(&servers, Some(key));

        // Same key must always produce the same pick
        for _ in 0..100 {
            assert_eq!(lb.pick(&servers, Some(key)), first_pick);
        }

        // Different keys may (and likely do) produce different picks
        let other_pick = lb.pick(&servers, Some("10.0.0.1"));
        // We just verify it returns a valid index; it may or may not differ
        assert!(other_pick < servers.len());
    }

    #[test]
    fn test_weighted_round_robin_distribution() {
        // Weights: server-0 gets weight 3, server-1 gets weight 1
        let lb = WeightedRoundRobin::new(vec![3, 1]);
        let s1 = String::from("heavy");
        let s2 = String::from("light");
        let servers: Vec<&String> = vec![&s1, &s2];

        let mut counts = [0u32; 2];
        let total = 400;
        for _ in 0..total {
            let idx = lb.pick(&servers, None);
            counts[idx] += 1;
        }

        // With weights [3,1], expanded is [0,0,0,1] so server-0 gets 3/4 and
        // server-1 gets 1/4 of the traffic.
        assert_eq!(counts[0], 300);
        assert_eq!(counts[1], 100);
    }
}
