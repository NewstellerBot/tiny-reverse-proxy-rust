use dashmap::DashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

struct CircuitBreakerEntry {
    state: CircuitState,
    failure_count: u32,
    last_failure: Option<Instant>,
    last_state_change: Instant,
    probe_in_flight: bool,
}

#[derive(Clone)]
pub struct CircuitBreaker {
    circuits: Arc<DashMap<String, Mutex<CircuitBreakerEntry>>>,
    failure_threshold: u32,
    cooldown: Duration,
    window: Duration,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown_secs: u64, window_secs: u64) -> Self {
        Self {
            circuits: Arc::new(DashMap::new()),
            failure_threshold,
            cooldown: Duration::from_secs(cooldown_secs),
            window: Duration::from_secs(window_secs),
        }
    }

    /// Returns true if the upstream is allowed (circuit not open).
    pub fn is_allowed(&self, upstream: &str) -> bool {
        let entry = self
            .circuits
            .entry(upstream.to_string())
            .or_insert_with(|| {
                Mutex::new(CircuitBreakerEntry {
                    state: CircuitState::Closed,
                    failure_count: 0,
                    last_failure: None,
                    last_state_change: Instant::now(),
                    probe_in_flight: false,
                })
            });
        let mut e = entry.value().lock().unwrap();
        match e.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if cooldown has elapsed -> transition to half-open
                if e.last_state_change.elapsed() >= self.cooldown {
                    e.state = CircuitState::HalfOpen;
                    e.last_state_change = Instant::now();
                    e.probe_in_flight = true;
                    true // Allow one probe request
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                if e.probe_in_flight {
                    false
                } else {
                    e.probe_in_flight = true;
                    true
                }
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&self, upstream: &str) {
        if let Some(entry) = self.circuits.get(upstream) {
            let mut e = entry.value().lock().unwrap();
            if e.state == CircuitState::HalfOpen {
                // Recovery: transition back to closed
                e.state = CircuitState::Closed;
                e.failure_count = 0;
                e.last_failure = None;
                e.last_state_change = Instant::now();
                e.probe_in_flight = false;
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self, upstream: &str) {
        let entry = self
            .circuits
            .entry(upstream.to_string())
            .or_insert_with(|| {
                Mutex::new(CircuitBreakerEntry {
                    state: CircuitState::Closed,
                    failure_count: 0,
                    last_failure: None,
                    last_state_change: Instant::now(),
                    probe_in_flight: false,
                })
            });
        let mut e = entry.value().lock().unwrap();
        match e.state {
            CircuitState::Closed => {
                // Check if failure is within window
                let now = Instant::now();
                if let Some(last) = e.last_failure {
                    if now.duration_since(last) > self.window {
                        // Outside window, reset count
                        e.failure_count = 0;
                    }
                }
                e.failure_count += 1;
                e.last_failure = Some(now);
                if e.failure_count >= self.failure_threshold {
                    e.state = CircuitState::Open;
                    e.last_state_change = now;
                    tracing::warn!(upstream = %upstream, "circuit breaker opened");
                }
            }
            CircuitState::HalfOpen => {
                // Failed during probe -> back to open
                e.state = CircuitState::Open;
                e.last_state_change = Instant::now();
                e.probe_in_flight = false;
                tracing::warn!(upstream = %upstream, "circuit breaker re-opened after probe failure");
            }
            CircuitState::Open => {} // Already open, ignore
        }
    }

    /// Get the current state of a circuit.
    pub fn state(&self, upstream: &str) -> CircuitState {
        self.circuits
            .get(upstream)
            .map(|entry| entry.value().lock().unwrap().state)
            .unwrap_or(CircuitState::Closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_starts_closed() {
        let cb = CircuitBreaker::new(3, 10, 60);
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        assert!(cb.is_allowed("upstream1"));
    }

    #[test]
    fn test_circuit_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, 10, 60);
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);
    }

    #[test]
    fn test_circuit_blocks_when_open() {
        let cb = CircuitBreaker::new(2, 10, 60);
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);
        assert!(!cb.is_allowed("upstream1"));
    }

    #[test]
    fn test_half_open_after_cooldown() {
        let cb = CircuitBreaker::new(2, 0, 60); // 0-second cooldown for testing
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);

        // With 0-second cooldown, the next is_allowed call should transition to HalfOpen
        std::thread::sleep(Duration::from_millis(10));
        assert!(cb.is_allowed("upstream1"));
        assert_eq!(cb.state("upstream1"), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_allows_only_one_probe_at_a_time() {
        let cb = CircuitBreaker::new(2, 0, 60);
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(10));
        assert!(cb.is_allowed("upstream1"));
        assert_eq!(cb.state("upstream1"), CircuitState::HalfOpen);

        assert!(
            !cb.is_allowed("upstream1"),
            "second concurrent half-open probe must be blocked"
        );

        cb.record_success("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        assert!(cb.is_allowed("upstream1"));
    }

    #[test]
    fn test_recovery_on_success() {
        let cb = CircuitBreaker::new(2, 0, 60);
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);

        // Transition to half-open
        std::thread::sleep(Duration::from_millis(10));
        assert!(cb.is_allowed("upstream1"));
        assert_eq!(cb.state("upstream1"), CircuitState::HalfOpen);

        // Successful probe should close the circuit
        cb.record_success("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        assert!(cb.is_allowed("upstream1"));
    }

    #[test]
    fn test_reopen_on_half_open_failure() {
        let cb = CircuitBreaker::new(2, 10, 60); // 10-second cooldown
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);

        // Manually transition to half-open by using a 0-second cooldown breaker
        let cb = CircuitBreaker::new(2, 0, 60);
        cb.record_failure("upstream1");
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);

        // Transition to half-open
        std::thread::sleep(Duration::from_millis(10));
        assert!(cb.is_allowed("upstream1"));
        assert_eq!(cb.state("upstream1"), CircuitState::HalfOpen);

        // Failed probe should re-open the circuit
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);
        // With 0-second cooldown, is_allowed immediately transitions to HalfOpen,
        // so we verify the state directly instead.
    }

    #[test]
    fn test_threshold_zero_opens_on_first_failure() {
        // With threshold=0, failure_count (u32) is always >= 0 after increment,
        // so the circuit opens on the very first failure.
        // NOTE: threshold=0 behaves identically to threshold=1 because the
        // check is `failure_count >= threshold` and failure_count is incremented
        // before the comparison.
        let cb = CircuitBreaker::new(0, 10, 60);

        // Before any failure, circuit is closed and allows requests
        assert_eq!(cb.state("upstream1"), CircuitState::Closed);
        assert!(cb.is_allowed("upstream1"));

        // First failure should open the circuit
        cb.record_failure("upstream1");
        assert_eq!(cb.state("upstream1"), CircuitState::Open);
        assert!(!cb.is_allowed("upstream1"));
    }
}
