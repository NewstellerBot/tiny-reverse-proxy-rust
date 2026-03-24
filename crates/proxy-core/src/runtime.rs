//! Process-local probe and drain state.
//!
//! This state is authoritative only for the current process and is intentionally not shared across
//! nodes. Operators should treat it as liveness/readiness signal, not as replicated control-plane
//! state.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeSnapshot {
    ready: bool,
    draining: bool,
    reason: Option<String>,
}

impl Default for ProbeSnapshot {
    fn default() -> Self {
        Self {
            ready: false,
            draining: false,
            reason: Some("starting".to_string()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProbeState {
    inner: Arc<RwLock<ProbeSnapshot>>,
}

impl ProbeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_ready(&self) {
        let mut snapshot = self
            .inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        snapshot.ready = true;
        snapshot.draining = false;
        snapshot.reason = None;
    }

    pub fn mark_not_ready(&self, reason: impl Into<String>) {
        let mut snapshot = self
            .inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        snapshot.ready = false;
        snapshot.draining = false;
        snapshot.reason = Some(reason.into());
    }

    pub fn mark_draining(&self, reason: impl Into<String>) {
        let mut snapshot = self
            .inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        snapshot.ready = false;
        snapshot.draining = true;
        snapshot.reason = Some(reason.into());
    }

    pub fn is_ready(&self) -> bool {
        self.snapshot().ready
    }

    pub fn is_draining(&self) -> bool {
        self.snapshot().draining
    }

    pub fn reason(&self) -> Option<String> {
        self.snapshot().reason
    }

    pub fn snapshot(&self) -> ProbeStateSnapshot {
        let snapshot = self
            .inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        ProbeStateSnapshot {
            ready: snapshot.ready,
            draining: snapshot.draining,
            reason: snapshot.reason.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeStateSnapshot {
    pub ready: bool,
    pub draining: bool,
    pub reason: Option<String>,
}

/// Process-local reliability state derived from the hot-path inflight counter and the probe state.
///
/// This is intentionally not replicated across nodes. It is meant for local operator visibility
/// and management/status reporting.
#[derive(Debug, Clone)]
pub struct RuntimeReliabilityState {
    inflight_requests: Arc<AtomicUsize>,
    max_inflight_requests: Option<usize>,
    brownout_inflight_requests: Option<usize>,
    probe_state: ProbeState,
}

impl RuntimeReliabilityState {
    pub fn new(
        inflight_requests: Arc<AtomicUsize>,
        max_inflight_requests: Option<usize>,
        brownout_inflight_requests: Option<usize>,
        probe_state: ProbeState,
    ) -> Self {
        Self {
            inflight_requests,
            max_inflight_requests,
            brownout_inflight_requests,
            probe_state,
        }
    }

    pub fn snapshot(&self) -> RuntimeReliabilitySnapshot {
        let inflight_requests = self.inflight_requests.load(Ordering::SeqCst);
        let brownout_threshold = self.brownout_inflight_requests;
        let probe = self.probe_state.snapshot();
        RuntimeReliabilitySnapshot {
            inflight_requests,
            max_inflight_requests: self.max_inflight_requests,
            brownout_active: brownout_threshold.is_some_and(|limit| inflight_requests >= limit),
            brownout_threshold,
            draining: probe.draining,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeReliabilitySnapshot {
    pub inflight_requests: usize,
    pub max_inflight_requests: Option<usize>,
    pub brownout_active: bool,
    pub brownout_threshold: Option<usize>,
    pub draining: bool,
}
