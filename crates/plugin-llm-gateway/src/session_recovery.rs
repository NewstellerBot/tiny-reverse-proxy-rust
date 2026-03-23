use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::MissedTickBehavior;

use crate::store::{GatewayStore, SessionRecord, Store, StoreError};

pub const SESSION_RECOVERY_REQUIRED_REASON: &str = "owner lease expired; recovery required";
pub const SESSION_CANCELLED_AFTER_OWNER_EXPIRY_REASON: &str =
    "cancel request finalized after owner lease expired";
pub const SESSION_CANCELLED_WITHOUT_OWNER_REASON: &str =
    "cancel request finalized without active owner";

const SESSION_RECOVERY_SCAN_LIMIT: u32 = 128;
const SESSION_RECOVERY_INTERVAL_SECS: u64 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionReconcileAction {
    RecoveryRequired,
    CancelledAfterOwnerExpiry,
    CancelledWithoutOwner,
}

impl SessionReconcileAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RecoveryRequired => "recovery_required",
            Self::CancelledAfterOwnerExpiry => "cancelled_after_owner_expiry",
            Self::CancelledWithoutOwner => "cancelled_without_owner",
        }
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::RecoveryRequired => SESSION_RECOVERY_REQUIRED_REASON,
            Self::CancelledAfterOwnerExpiry => SESSION_CANCELLED_AFTER_OWNER_EXPIRY_REASON,
            Self::CancelledWithoutOwner => SESSION_CANCELLED_WITHOUT_OWNER_REASON,
        }
    }
}

pub fn session_owner_is_stale(record: &SessionRecord, now: i64) -> bool {
    record.owner_id.is_some()
        && record
            .lease_expires_at_unix
            .map(|expires_at| expires_at <= now)
            .unwrap_or(true)
}

pub fn session_cancel_is_pending(record: &SessionRecord) -> bool {
    !session_is_terminal(record) && record.cancel_requested_at_unix.is_some()
}

pub fn session_handoff_is_pending(record: &SessionRecord) -> bool {
    record.handoff_target_owner_id.is_some() && record.handoff_requested_at_unix.is_some()
}

pub fn session_recovery_required(record: &SessionRecord, now: i64) -> bool {
    if session_is_terminal(record) || session_cancel_is_pending(record) {
        return false;
    }
    if session_owner_is_stale(record, now) {
        return true;
    }
    record.owner_id.is_none()
        && record.status.as_deref() == Some("paused")
        && record.last_transition_reason.as_deref() == Some(SESSION_RECOVERY_REQUIRED_REASON)
}

pub fn session_recovery_reason(record: &SessionRecord, now: i64) -> Option<&'static str> {
    session_recovery_required(record, now).then_some(SESSION_RECOVERY_REQUIRED_REASON)
}

pub fn reconcile_session_record(
    record: &mut SessionRecord,
    now: i64,
) -> Option<SessionReconcileAction> {
    if session_is_terminal(record) {
        return None;
    }

    let owner_stale = session_owner_is_stale(record, now);
    let cancel_pending = session_cancel_is_pending(record);
    if owner_stale {
        clear_session_owner(record);
        clear_session_handoff(record);
        if cancel_pending {
            record.status = Some("cancelled".to_string());
            record.last_transition_reason =
                Some(SESSION_CANCELLED_AFTER_OWNER_EXPIRY_REASON.to_string());
            record.last_transition_at_unix = Some(now);
            record.updated_at_unix = now;
            return Some(SessionReconcileAction::CancelledAfterOwnerExpiry);
        }

        if record.status.as_deref() != Some("paused") {
            record.status = Some("paused".to_string());
        }
        record.last_transition_reason = Some(SESSION_RECOVERY_REQUIRED_REASON.to_string());
        record.last_transition_at_unix = Some(now);
        record.updated_at_unix = now;
        return Some(SessionReconcileAction::RecoveryRequired);
    }

    if cancel_pending && record.owner_id.is_none() {
        clear_session_handoff(record);
        record.status = Some("cancelled".to_string());
        record.last_transition_reason = Some(SESSION_CANCELLED_WITHOUT_OWNER_REASON.to_string());
        record.last_transition_at_unix = Some(now);
        record.updated_at_unix = now;
        return Some(SessionReconcileAction::CancelledWithoutOwner);
    }

    None
}

pub fn spawn_session_recovery_task(store: Arc<Store>) {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(Duration::from_secs(SESSION_RECOVERY_INTERVAL_SECS));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            if let Err(error) = reconcile_sessions_once(
                store.as_ref(),
                current_unix_time(),
                SESSION_RECOVERY_SCAN_LIMIT,
            )
            .await
            {
                tracing::warn!(error = %error, "failed to reconcile stale sessions");
            }
        }
    });
}

pub async fn reconcile_sessions_once(
    store: &Store,
    now: i64,
    limit: u32,
) -> Result<usize, StoreError> {
    let session_ids = store.list_sessions_for_recovery(now, limit).await?;
    let mut updated = 0usize;
    for session_id in session_ids {
        let Some(mut record) = store.get_session(&session_id).await? else {
            continue;
        };
        let Some(action) = reconcile_session_record(&mut record, now) else {
            continue;
        };
        store.upsert_session(&record).await?;
        updated += 1;
        tracing::info!(
            session_id = %session_id,
            action = action.as_str(),
            "reconciled stale session"
        );
    }
    Ok(updated)
}

fn clear_session_owner(record: &mut SessionRecord) {
    record.owner_id = None;
    record.owner_acquired_at_unix = None;
    record.lease_expires_at_unix = None;
}

fn clear_session_handoff(record: &mut SessionRecord) {
    record.handoff_target_owner_id = None;
    record.handoff_requested_at_unix = None;
    record.handoff_reason = None;
}

fn session_is_terminal(record: &SessionRecord) -> bool {
    matches!(
        record.status.as_deref(),
        Some("completed" | "cancelled" | "failed")
    )
}

fn current_unix_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_session() -> SessionRecord {
        SessionRecord {
            session_id: "session-1".to_string(),
            project_id: Some("project-a".to_string()),
            project_ids_json: Some(r#"["project-a"]"#.to_string()),
            first_request_unix: Some(10),
            last_request_unix: Some(20),
            updated_at_unix: 20,
            request_count: 1,
            streaming_request_count: 0,
            total_input_tokens: 10,
            total_output_tokens: 5,
            total_cost: 0.01,
            providers_json: None,
            models_json: None,
            prompt_names_json: None,
            prompt_versions_json: None,
            tool_names_json: None,
            latest_request_json: None,
            safety_event_count: 0,
            semantic_event_count: 0,
            semantic_degraded_count: 0,
            tool_call_count: 0,
            tool_error_count: 0,
            status: Some("active".to_string()),
            owner_id: Some("worker-a".to_string()),
            owner_acquired_at_unix: Some(15),
            last_transition_at_unix: Some(15),
            last_transition_reason: Some("claimed".to_string()),
            last_heartbeat_unix: Some(19),
            lease_expires_at_unix: Some(20),
            cancel_requested_at_unix: None,
            cancel_requested_by: None,
            cancel_reason: None,
            handoff_target_owner_id: None,
            handoff_requested_at_unix: None,
            handoff_reason: None,
            state_json: None,
            metadata_json: None,
        }
    }

    #[test]
    fn reconcile_expired_owner_marks_session_for_recovery() {
        let mut record = base_session();
        let action = reconcile_session_record(&mut record, 20);
        assert_eq!(action, Some(SessionReconcileAction::RecoveryRequired));
        assert_eq!(record.status.as_deref(), Some("paused"));
        assert_eq!(
            record.last_transition_reason.as_deref(),
            Some(SESSION_RECOVERY_REQUIRED_REASON)
        );
        assert!(record.owner_id.is_none());
        assert!(session_recovery_required(&record, 20));
    }

    #[test]
    fn reconcile_expired_owner_finalizes_pending_cancel() {
        let mut record = base_session();
        record.cancel_requested_at_unix = Some(18);
        record.cancel_requested_by = Some("operator-a".to_string());
        record.cancel_reason = Some("stop".to_string());
        record.handoff_target_owner_id = Some("worker-b".to_string());
        record.handoff_requested_at_unix = Some(17);
        record.handoff_reason = Some("move".to_string());

        let action = reconcile_session_record(&mut record, 20);
        assert_eq!(
            action,
            Some(SessionReconcileAction::CancelledAfterOwnerExpiry)
        );
        assert_eq!(record.status.as_deref(), Some("cancelled"));
        assert_eq!(
            record.last_transition_reason.as_deref(),
            Some(SESSION_CANCELLED_AFTER_OWNER_EXPIRY_REASON)
        );
        assert!(!session_handoff_is_pending(&record));
        assert!(!session_recovery_required(&record, 20));
    }

    #[test]
    fn reconcile_pending_cancel_without_owner_finalizes_session() {
        let mut record = base_session();
        record.owner_id = None;
        record.owner_acquired_at_unix = None;
        record.lease_expires_at_unix = None;
        record.status = Some("paused".to_string());
        record.cancel_requested_at_unix = Some(19);

        let action = reconcile_session_record(&mut record, 20);
        assert_eq!(action, Some(SessionReconcileAction::CancelledWithoutOwner));
        assert_eq!(record.status.as_deref(), Some("cancelled"));
        assert_eq!(
            record.last_transition_reason.as_deref(),
            Some(SESSION_CANCELLED_WITHOUT_OWNER_REASON)
        );
    }
}
