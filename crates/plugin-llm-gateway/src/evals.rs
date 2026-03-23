use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderName, HeaderValue, CONTENT_TYPE};
use hyper::{Method, Request, StatusCode, Uri};
use regex::Regex;
use serde_json::{json, Value};
use tokio::time::timeout;

use proxy_core::handlers::proxy::build_client;

use crate::governance::current_timestamp_string;
use crate::store::{
    GatewayStore, ProjectDatasetItemRecord, ProjectEvalRunItemRecord, ProjectEvalRunRecord, Store,
};

static EVAL_RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct ProjectEvalRunRequest {
    pub dataset_name: String,
    pub target_url: String,
    pub headers: HashMap<String, String>,
    pub timeout_ms: Option<u64>,
    pub judge_url: Option<String>,
    pub judge_kind: Option<String>,
    pub judge_model: Option<String>,
    pub judge_headers: HashMap<String, String>,
    pub judge_timeout_ms: Option<u64>,
    pub prompt_name: Option<String>,
    pub prompt_version: Option<String>,
    pub provider_name: Option<String>,
    pub model: Option<String>,
    pub route_path: Option<String>,
    pub safety_profile: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunExecution {
    pub run: ProjectEvalRunRecord,
    pub items: Vec<ProjectEvalRunItemRecord>,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunComparison {
    pub baseline_run: ProjectEvalRunRecord,
    pub candidate_run: ProjectEvalRunRecord,
    pub summary: ProjectEvalRunComparisonSummary,
    pub context: ProjectEvalRunComparisonContext,
    pub items: Vec<ProjectEvalRunItemComparison>,
    pub gate: Option<ProjectEvalRunComparisonGate>,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunComparisonSummary {
    pub baseline_pass_rate: f64,
    pub candidate_pass_rate: f64,
    pub delta_pass_rate: f64,
    pub baseline_average_latency_ms: f64,
    pub candidate_average_latency_ms: f64,
    pub delta_average_latency_ms: f64,
    pub baseline_total_cost: f64,
    pub candidate_total_cost: f64,
    pub delta_total_cost: f64,
    pub improved_items: u32,
    pub regressed_items: u32,
    pub unchanged_items: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProjectEvalRunComparisonGateRequest {
    pub preset: Option<ProjectEvalRunComparisonGatePreset>,
    pub max_regressions: Option<u32>,
    pub min_candidate_pass_rate: Option<f64>,
    pub min_pass_rate_delta: Option<f64>,
    pub max_latency_increase_ms: Option<f64>,
    pub max_cost_increase: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectEvalRunComparisonGatePreset {
    Strict,
    Balanced,
    Exploratory,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunComparisonContext {
    pub baseline: Value,
    pub candidate: Value,
    pub changed_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunComparisonGate {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub preset: Option<ProjectEvalRunComparisonGatePreset>,
    pub thresholds: ProjectEvalRunComparisonGateRequest,
    pub recommendation: ProjectEvalRunRolloutRecommendation,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunRolloutRecommendation {
    pub action: ProjectEvalRunRolloutAction,
    pub summary: String,
    pub changed_context_fields: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectEvalRunRolloutAction {
    Promote,
    Canary,
    Review,
    Hold,
}

#[derive(Debug, Clone)]
pub struct ProjectEvalRunItemComparison {
    pub item_id: String,
    pub baseline_passed: Option<bool>,
    pub candidate_passed: Option<bool>,
    pub improved: bool,
    pub regressed: bool,
    pub changed: bool,
    pub baseline_output_text: Option<String>,
    pub candidate_output_text: Option<String>,
    pub baseline_evaluation_json: Option<String>,
    pub candidate_evaluation_json: Option<String>,
}

#[derive(Clone)]
struct PreparedHeaders(Vec<(HeaderName, HeaderValue)>);

struct PreparedEvalRun {
    run: ProjectEvalRunRecord,
    dataset_items: Vec<ProjectDatasetItemRecord>,
    prepared_headers: PreparedHeaders,
    uri: Uri,
    judge: Option<PreparedJudge>,
    request_timeout: Duration,
    context_json: Value,
    request_json: Value,
}

#[derive(Clone, Copy)]
enum JudgeMode {
    Webhook,
    OpenAi,
    Anthropic,
}

#[derive(Clone)]
struct PreparedJudge {
    mode: JudgeMode,
    uri: Uri,
    model: Option<String>,
    headers: PreparedHeaders,
    request_timeout: Duration,
}

#[derive(Debug, Clone)]
struct ExtractedUsage {
    input_tokens: u64,
    output_tokens: u64,
    cost: f64,
}

#[derive(Debug, Clone)]
struct EvalOutcome {
    passed: bool,
    evaluation_json: Option<String>,
    output_text: Option<String>,
    status_code: Option<u16>,
    error: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    cost: f64,
    latency_ms: u64,
}

#[derive(Debug, Clone)]
struct JudgeExpectation {
    rubric: String,
    min_score: Option<f64>,
    require_passed: bool,
}

pub async fn execute_project_eval_run(
    store: Arc<Store>,
    project_id: &str,
    request: ProjectEvalRunRequest,
) -> Result<ProjectEvalRunExecution, Box<dyn std::error::Error>> {
    let mut prepared =
        prepare_eval_run(project_id, &request, Arc::clone(&store), "running").await?;
    store.upsert_project_eval_run(&prepared.run).await?;
    execute_prepared_eval_run(store, &mut prepared, false).await
}

pub async fn queue_project_eval_run(
    store: Arc<Store>,
    project_id: &str,
    request: ProjectEvalRunRequest,
) -> Result<ProjectEvalRunRecord, Box<dyn std::error::Error>> {
    let mut prepared = prepare_eval_run(project_id, &request, Arc::clone(&store), "queued").await?;
    store.upsert_project_eval_run(&prepared.run).await?;
    let queued_run = prepared.run.clone();
    let store_for_task = Arc::clone(&store);
    tokio::spawn(async move {
        let execution_result =
            execute_prepared_eval_run(store_for_task.clone(), &mut prepared, true)
                .await
                .map_err(|error| error.to_string());
        if let Err(error_message) = execution_result {
            let _ = mark_eval_run_failed(
                store_for_task,
                &mut prepared.run,
                &prepared.context_json,
                &prepared.request_json,
                prepared.request_timeout,
                &error_message,
            )
            .await;
        }
    });
    Ok(queued_run)
}

pub async fn compare_project_eval_runs(
    store: Arc<Store>,
    project_id: &str,
    baseline_run_id: &str,
    candidate_run_id: &str,
    gate_request: Option<ProjectEvalRunComparisonGateRequest>,
) -> Result<ProjectEvalRunComparison, Box<dyn std::error::Error>> {
    let baseline_run = store
        .get_project_eval_run(project_id, baseline_run_id)
        .await?
        .ok_or_else(|| format!("eval run not found: {}", baseline_run_id))?;
    let candidate_run = store
        .get_project_eval_run(project_id, candidate_run_id)
        .await?
        .ok_or_else(|| format!("eval run not found: {}", candidate_run_id))?;
    if baseline_run.dataset_name != candidate_run.dataset_name {
        return Err(format!(
            "cannot compare eval runs from different datasets: '{}' vs '{}'",
            baseline_run.dataset_name, candidate_run.dataset_name
        )
        .into());
    }

    let baseline_items = store
        .get_project_eval_run_items(project_id, baseline_run_id)
        .await?;
    let candidate_items = store
        .get_project_eval_run_items(project_id, candidate_run_id)
        .await?;
    let mut baseline_by_item = baseline_items
        .into_iter()
        .map(|item| (item.item_id.clone(), item))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_by_item = candidate_items
        .into_iter()
        .map(|item| (item.item_id.clone(), item))
        .collect::<BTreeMap<_, _>>();
    let item_ids = baseline_by_item
        .keys()
        .chain(candidate_by_item.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();

    let mut improved_items = 0u32;
    let mut regressed_items = 0u32;
    let mut unchanged_items = 0u32;
    let mut items = Vec::with_capacity(item_ids.len());
    for item_id in item_ids {
        let baseline = baseline_by_item.remove(&item_id);
        let candidate = candidate_by_item.remove(&item_id);
        let baseline_passed = baseline.as_ref().map(|item| item.passed);
        let candidate_passed = candidate.as_ref().map(|item| item.passed);
        let improved = baseline_passed == Some(false) && candidate_passed == Some(true);
        let regressed = baseline_passed == Some(true) && candidate_passed == Some(false);
        let changed = baseline_passed != candidate_passed
            || baseline
                .as_ref()
                .and_then(|item| item.output_text.as_deref())
                != candidate
                    .as_ref()
                    .and_then(|item| item.output_text.as_deref());
        if improved {
            improved_items += 1;
        } else if regressed {
            regressed_items += 1;
        } else {
            unchanged_items += 1;
        }
        items.push(ProjectEvalRunItemComparison {
            item_id,
            baseline_passed,
            candidate_passed,
            improved,
            regressed,
            changed,
            baseline_output_text: baseline.as_ref().and_then(|item| item.output_text.clone()),
            candidate_output_text: candidate.as_ref().and_then(|item| item.output_text.clone()),
            baseline_evaluation_json: baseline
                .as_ref()
                .and_then(|item| item.evaluation_json.clone()),
            candidate_evaluation_json: candidate
                .as_ref()
                .and_then(|item| item.evaluation_json.clone()),
        });
    }

    let baseline_pass_rate = pass_rate(&baseline_run);
    let candidate_pass_rate = pass_rate(&candidate_run);
    let context = build_comparison_context(&baseline_run, &candidate_run);
    let summary = ProjectEvalRunComparisonSummary {
        baseline_pass_rate,
        candidate_pass_rate,
        delta_pass_rate: candidate_pass_rate - baseline_pass_rate,
        baseline_average_latency_ms: baseline_run.average_latency_ms,
        candidate_average_latency_ms: candidate_run.average_latency_ms,
        delta_average_latency_ms: candidate_run.average_latency_ms
            - baseline_run.average_latency_ms,
        baseline_total_cost: baseline_run.total_cost,
        candidate_total_cost: candidate_run.total_cost,
        delta_total_cost: candidate_run.total_cost - baseline_run.total_cost,
        improved_items,
        regressed_items,
        unchanged_items,
    };
    let gate = gate_request
        .filter(comparison_gate_requested)
        .map(|request| evaluate_comparison_gate(&summary, &context, request));

    Ok(ProjectEvalRunComparison {
        baseline_run: baseline_run.clone(),
        candidate_run: candidate_run.clone(),
        summary,
        context,
        items,
        gate,
    })
}

async fn prepare_eval_run(
    project_id: &str,
    request: &ProjectEvalRunRequest,
    store: Arc<Store>,
    initial_status: &str,
) -> Result<PreparedEvalRun, Box<dyn std::error::Error>> {
    let dataset = store
        .get_project_dataset(project_id, &request.dataset_name)
        .await?
        .ok_or_else(|| format!("dataset not found: {}", request.dataset_name))?;
    let dataset_items = store
        .get_project_dataset_items(project_id, &request.dataset_name)
        .await?;
    let prepared_headers = prepare_headers(&request.headers)?;
    let uri: Uri = request.target_url.parse()?;
    let request_timeout = Duration::from_millis(request.timeout_ms.unwrap_or(15_000));
    let judge = prepare_judge(
        request.judge_url.as_deref(),
        request.judge_kind.as_deref(),
        request.judge_model.as_deref(),
        &request.judge_headers,
        request.judge_timeout_ms,
        request_timeout,
    )?;
    let run_id = generate_eval_run_id();
    let created_at = current_timestamp_string();
    let context_json = eval_context_json(request);
    let request_json = eval_request_json(request);

    Ok(PreparedEvalRun {
        run: ProjectEvalRunRecord {
            run_id,
            project_id: project_id.to_string(),
            dataset_name: dataset.dataset_name,
            target_url: request.target_url.clone(),
            status: initial_status.to_string(),
            total_items: dataset_items.len() as u32,
            passed_items: 0,
            failed_items: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cost: 0.0,
            average_latency_ms: 0.0,
            summary_json: Some(build_eval_summary_json(
                0.0,
                request_timeout,
                &context_json,
                &request_json,
                None,
            )),
            created_at,
            completed_at: None,
        },
        dataset_items,
        prepared_headers,
        uri,
        judge,
        request_timeout,
        context_json,
        request_json,
    })
}

fn prepare_eval_run_from_record(
    run: ProjectEvalRunRecord,
    dataset_items: Vec<ProjectDatasetItemRecord>,
) -> Result<PreparedEvalRun, Box<dyn std::error::Error>> {
    let summary = parse_eval_summary(run.summary_json.as_deref());
    let context_json = summary
        .as_ref()
        .and_then(|value| value.get("context"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let request_json = summary
        .as_ref()
        .and_then(|value| value.get("request"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let request_timeout = summary
        .as_ref()
        .and_then(|value| value.get("timeout_ms"))
        .and_then(Value::as_u64)
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(15_000));
    let prepared_headers = prepare_headers(&request_headers_from_json(&request_json)?)?;
    let judge = prepare_judge_from_request_json(&request_json, request_timeout)?;
    let uri: Uri = run.target_url.parse()?;

    let mut recovered_run = run;
    recovered_run.total_items = dataset_items.len() as u32;
    recovered_run.passed_items = 0;
    recovered_run.failed_items = 0;
    recovered_run.total_input_tokens = 0;
    recovered_run.total_output_tokens = 0;
    recovered_run.total_cost = 0.0;
    recovered_run.average_latency_ms = 0.0;
    recovered_run.completed_at = None;
    recovered_run.summary_json = Some(build_eval_summary_json(
        0.0,
        request_timeout,
        &context_json,
        &request_json,
        Some(json!({
            "recovered": true,
            "previous_status": recovered_run.status,
        })),
    ));

    Ok(PreparedEvalRun {
        run: recovered_run,
        dataset_items,
        prepared_headers,
        uri,
        judge,
        request_timeout,
        context_json,
        request_json,
    })
}

async fn execute_prepared_eval_run(
    store: Arc<Store>,
    prepared: &mut PreparedEvalRun,
    transition_from_queued: bool,
) -> Result<ProjectEvalRunExecution, Box<dyn std::error::Error>> {
    if transition_from_queued {
        prepared.run.status = "running".to_string();
        store.upsert_project_eval_run(&prepared.run).await?;
    }

    let client = build_client();
    let mut item_results = Vec::with_capacity(prepared.dataset_items.len());
    let mut total_latency_ms = 0_u64;

    for item in &prepared.dataset_items {
        let outcome = evaluate_dataset_item(
            &client,
            &prepared.uri,
            &prepared.prepared_headers,
            prepared.judge.as_ref(),
            prepared.request_timeout,
            &prepared.context_json,
            item,
        )
        .await;
        total_latency_ms += outcome.latency_ms;
        if outcome.passed {
            prepared.run.passed_items += 1;
        } else {
            prepared.run.failed_items += 1;
        }
        prepared.run.total_input_tokens += outcome.input_tokens;
        prepared.run.total_output_tokens += outcome.output_tokens;
        prepared.run.total_cost += outcome.cost;

        let item_record = ProjectEvalRunItemRecord {
            run_id: prepared.run.run_id.clone(),
            project_id: prepared.run.project_id.clone(),
            dataset_name: prepared.run.dataset_name.clone(),
            item_id: item.item_id.clone(),
            passed: outcome.passed,
            status_code: outcome.status_code,
            latency_ms: outcome.latency_ms,
            output_text: outcome.output_text,
            evaluation_json: outcome.evaluation_json,
            error: outcome.error,
            input_tokens: outcome.input_tokens,
            output_tokens: outcome.output_tokens,
            cost: outcome.cost,
            created_at: current_timestamp_string(),
        };
        store.upsert_project_eval_run_item(&item_record).await?;
        item_results.push(item_record);
    }

    prepared.run.status = "completed".to_string();
    prepared.run.completed_at = Some(current_timestamp_string());
    if prepared.run.total_items > 0 {
        prepared.run.average_latency_ms = total_latency_ms as f64 / prepared.run.total_items as f64;
    }
    prepared.run.summary_json = Some(build_eval_summary_json(
        pass_rate(&prepared.run),
        prepared.request_timeout,
        &prepared.context_json,
        &prepared.request_json,
        None,
    ));
    store.upsert_project_eval_run(&prepared.run).await?;

    Ok(ProjectEvalRunExecution {
        run: prepared.run.clone(),
        items: item_results,
    })
}

async fn mark_eval_run_failed(
    store: Arc<Store>,
    run: &mut ProjectEvalRunRecord,
    context_json: &Value,
    request_json: &Value,
    request_timeout: Duration,
    error: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    run.status = "failed".to_string();
    run.completed_at = Some(current_timestamp_string());
    run.summary_json = Some(build_eval_summary_json(
        pass_rate(run),
        request_timeout,
        context_json,
        request_json,
        Some(json!({
            "failure": {
                "kind": "background_error",
                "message": error,
            }
        })),
    ));
    store.upsert_project_eval_run(run).await?;
    Ok(())
}

fn pass_rate(run: &ProjectEvalRunRecord) -> f64 {
    if run.total_items == 0 {
        0.0
    } else {
        run.passed_items as f64 / run.total_items as f64
    }
}

fn build_eval_summary_json(
    pass_rate: f64,
    request_timeout: Duration,
    context_json: &Value,
    request_json: &Value,
    extra: Option<Value>,
) -> String {
    let mut summary = serde_json::Map::new();
    summary.insert("pass_rate".to_string(), json!(pass_rate));
    summary.insert(
        "timeout_ms".to_string(),
        json!(request_timeout.as_millis() as u64),
    );
    summary.insert("context".to_string(), context_json.clone());
    summary.insert("request".to_string(), request_json.clone());
    if let Some(extra) = extra {
        if let Value::Object(extra_map) = extra {
            for (key, value) in extra_map {
                summary.insert(key, value);
            }
        } else {
            summary.insert("details".to_string(), extra);
        }
    }
    Value::Object(summary).to_string()
}

fn eval_context_json(request: &ProjectEvalRunRequest) -> Value {
    let mut context = serde_json::Map::new();
    if let Some(prompt_name) = &request.prompt_name {
        context.insert("prompt_name".to_string(), json!(prompt_name));
    }
    if let Some(prompt_version) = &request.prompt_version {
        context.insert("prompt_version".to_string(), json!(prompt_version));
    }
    if let Some(provider_name) = &request.provider_name {
        context.insert("provider_name".to_string(), json!(provider_name));
    }
    if let Some(model) = &request.model {
        context.insert("model".to_string(), json!(model));
    }
    if let Some(route_path) = &request.route_path {
        context.insert("route_path".to_string(), json!(route_path));
    }
    if let Some(safety_profile) = &request.safety_profile {
        context.insert("safety_profile".to_string(), json!(safety_profile));
    }
    if request.judge_url.is_some() {
        context.insert("judge_enabled".to_string(), Value::Bool(true));
    }
    if let Some(judge_kind) = &request.judge_kind {
        context.insert("judge_kind".to_string(), json!(judge_kind));
    }
    if let Some(judge_model) = &request.judge_model {
        context.insert("judge_model".to_string(), json!(judge_model));
    }
    Value::Object(context)
}

fn eval_request_json(request: &ProjectEvalRunRequest) -> Value {
    json!({
        "headers": request.headers,
        "judge_url": request.judge_url,
        "judge_kind": request.judge_kind,
        "judge_model": request.judge_model,
        "judge_headers": request.judge_headers,
        "judge_timeout_ms": request.judge_timeout_ms,
    })
}

fn parse_eval_summary(summary_json: Option<&str>) -> Option<Value> {
    summary_json.and_then(|summary_json| serde_json::from_str(summary_json).ok())
}

fn string_map_from_json_field(
    request_json: &Value,
    field: &str,
    description: &str,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut headers = HashMap::new();
    let Some(headers_json) = request_json.get(field) else {
        return Ok(headers);
    };
    let Some(header_map) = headers_json.as_object() else {
        return Err(format!("{description} must be an object").into());
    };
    for (name, value) in header_map {
        let Some(value) = value.as_str() else {
            return Err(format!("{description} entry '{name}' must be a string").into());
        };
        headers.insert(name.clone(), value.to_string());
    }
    Ok(headers)
}

fn request_headers_from_json(
    request_json: &Value,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    string_map_from_json_field(request_json, "headers", "eval summary request.headers")
}

fn judge_headers_from_json(
    request_json: &Value,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    string_map_from_json_field(
        request_json,
        "judge_headers",
        "eval summary request.judge_headers",
    )
}

fn parse_judge_mode(judge_kind: Option<&str>) -> Result<JudgeMode, Box<dyn std::error::Error>> {
    match judge_kind.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("webhook") => Ok(JudgeMode::Webhook),
        Some("openai") => Ok(JudgeMode::OpenAi),
        Some("anthropic") => Ok(JudgeMode::Anthropic),
        Some(other) => Err(format!("unsupported judge_kind: {other}").into()),
    }
}

fn prepare_judge(
    judge_url: Option<&str>,
    judge_kind: Option<&str>,
    judge_model: Option<&str>,
    judge_headers: &HashMap<String, String>,
    judge_timeout_ms: Option<u64>,
    default_timeout: Duration,
) -> Result<Option<PreparedJudge>, Box<dyn std::error::Error>> {
    let Some(judge_url) = judge_url else {
        return Ok(None);
    };
    let mode = parse_judge_mode(judge_kind)?;
    let model = judge_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    if matches!(mode, JudgeMode::OpenAi | JudgeMode::Anthropic) && model.is_none() {
        return Err(format!(
            "judge_model is required when judge_kind is '{}'",
            judge_kind.unwrap_or_default()
        )
        .into());
    }
    let uri: Uri = judge_url.parse()?;
    Ok(Some(PreparedJudge {
        mode,
        uri,
        model,
        headers: prepare_headers(judge_headers)?,
        request_timeout: Duration::from_millis(
            judge_timeout_ms.unwrap_or(default_timeout.as_millis() as u64),
        ),
    }))
}

fn prepare_judge_from_request_json(
    request_json: &Value,
    default_timeout: Duration,
) -> Result<Option<PreparedJudge>, Box<dyn std::error::Error>> {
    let judge_headers = judge_headers_from_json(request_json)?;
    prepare_judge(
        request_json.get("judge_url").and_then(Value::as_str),
        request_json.get("judge_kind").and_then(Value::as_str),
        request_json.get("judge_model").and_then(Value::as_str),
        &judge_headers,
        request_json.get("judge_timeout_ms").and_then(Value::as_u64),
        default_timeout,
    )
}

pub fn spawn_eval_recovery_task(store: Arc<Store>) {
    tokio::spawn(async move {
        let pending_runs = match store
            .get_project_eval_runs_by_status(&["queued", "running"], 128)
            .await
        {
            Ok(runs) => runs,
            Err(error) => {
                tracing::warn!(error = %error, "failed to load pending eval runs for recovery");
                return;
            }
        };

        for run in pending_runs {
            let project_id = run.project_id.clone();
            let run_id = run.run_id.clone();
            if let Err(error) = recover_project_eval_run(Arc::clone(&store), run).await {
                tracing::warn!(
                    project_id,
                    run_id,
                    error = %error,
                    "failed to recover pending eval run"
                );
            }
        }
    });
}

async fn recover_project_eval_run(
    store: Arc<Store>,
    run: ProjectEvalRunRecord,
) -> Result<ProjectEvalRunExecution, Box<dyn std::error::Error>> {
    let dataset_items = store
        .get_project_dataset_items(&run.project_id, &run.dataset_name)
        .await?;
    let mut prepared = prepare_eval_run_from_record(run, dataset_items)?;
    execute_prepared_eval_run(store, &mut prepared, true).await
}

fn prepare_headers(
    raw_headers: &HashMap<String, String>,
) -> Result<PreparedHeaders, Box<dyn std::error::Error>> {
    let mut headers = Vec::with_capacity(raw_headers.len());
    for (name, value) in raw_headers {
        headers.push((
            HeaderName::try_from(name.as_str())?,
            HeaderValue::try_from(value.as_str())?,
        ));
    }
    Ok(PreparedHeaders(headers))
}

async fn evaluate_dataset_item(
    client: &proxy_core::handlers::proxy::HttpClient,
    uri: &Uri,
    prepared_headers: &PreparedHeaders,
    judge: Option<&PreparedJudge>,
    request_timeout: Duration,
    context_json: &Value,
    item: &crate::store::ProjectDatasetItemRecord,
) -> EvalOutcome {
    let body = Bytes::from(item.input_json.clone());
    let request = match build_eval_request(uri, prepared_headers, body) {
        Ok(request) => request,
        Err(error) => {
            return EvalOutcome {
                passed: false,
                evaluation_json: Some(
                    json!({ "kind": "request_error", "message": error.to_string() }).to_string(),
                ),
                output_text: None,
                status_code: None,
                error: Some(error.to_string()),
                input_tokens: 0,
                output_tokens: 0,
                cost: 0.0,
                latency_ms: 0,
            };
        }
    };

    let started = Instant::now();
    let response = match timeout(request_timeout, client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            return EvalOutcome {
                passed: false,
                evaluation_json: Some(
                    json!({ "kind": "request_error", "message": error.to_string() }).to_string(),
                ),
                output_text: None,
                status_code: None,
                error: Some(error.to_string()),
                input_tokens: 0,
                output_tokens: 0,
                cost: 0.0,
                latency_ms: started.elapsed().as_millis() as u64,
            };
        }
        Err(_) => {
            return EvalOutcome {
                passed: false,
                evaluation_json: Some(
                    json!({ "kind": "request_timeout", "timeout_ms": request_timeout.as_millis() as u64 })
                        .to_string(),
                ),
                output_text: None,
                status_code: None,
                error: Some("request timed out".to_string()),
                input_tokens: 0,
                output_tokens: 0,
                cost: 0.0,
                latency_ms: started.elapsed().as_millis() as u64,
            };
        }
    };

    let latency_ms = started.elapsed().as_millis() as u64;
    let status = response.status();
    let status_code = status.as_u16();
    let body_bytes = match response.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            return EvalOutcome {
                passed: false,
                evaluation_json: Some(
                    json!({ "kind": "response_error", "message": error.to_string() }).to_string(),
                ),
                output_text: None,
                status_code: Some(status_code),
                error: Some(error.to_string()),
                input_tokens: 0,
                output_tokens: 0,
                cost: 0.0,
                latency_ms,
            };
        }
    };
    let response_json = serde_json::from_slice::<Value>(&body_bytes).ok();
    let output_text = response_json.as_ref().and_then(extract_output_text);
    let usage = response_json
        .as_ref()
        .map(extract_usage)
        .unwrap_or(ExtractedUsage {
            input_tokens: 0,
            output_tokens: 0,
            cost: 0.0,
        });
    let expected_value = match parse_expected_output_json(item.expected_output_json.as_deref()) {
        Ok(value) => value,
        Err(error) => {
            return EvalOutcome {
                passed: false,
                evaluation_json: Some(
                    json!({
                        "kind": "invalid_expected_output",
                        "message": error,
                        "passed": false,
                    })
                    .to_string(),
                ),
                output_text,
                status_code: Some(status_code),
                error: Some("invalid expected output".to_string()),
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cost: usage.cost,
                latency_ms,
            };
        }
    };
    let structured_output = normalized_output_json(response_json.as_ref(), output_text.as_deref());
    let (mut passed, mut evaluation_json) = evaluate_expectation(
        expected_value.as_ref(),
        output_text.as_deref(),
        response_json.as_ref(),
        status,
    );
    let judge_config_present = expected_value
        .as_ref()
        .and_then(|value| value.get("judge"))
        .is_some();
    let judge_expectation = judge_expectation_from_expected(expected_value.as_ref());
    if judge_config_present && judge_expectation.is_none() {
        return EvalOutcome {
            passed: false,
            evaluation_json: Some(
                json!({
                    "kind": "invalid_expected_output",
                    "message": "expected_output.judge must include a non-empty rubric",
                    "passed": false,
                })
                .to_string(),
            ),
            output_text,
            status_code: Some(status_code),
            error: Some("invalid expected output".to_string()),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cost: usage.cost,
            latency_ms,
        };
    }
    if let Some(judge_expectation) = judge_expectation {
        let judge_result = match judge {
            Some(judge) => {
                evaluate_with_judge(
                    client,
                    judge,
                    item,
                    context_json,
                    output_text.as_deref(),
                    structured_output.as_ref(),
                    status,
                    &judge_expectation,
                )
                .await
            }
            None => json!({
                "kind": "judge_error",
                "message": "judge_url is required when expected_output.judge is configured",
                "passed": false,
            }),
        };
        let judge_passed = judge_result
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        passed &= judge_passed;
        if let Value::Object(map) = &mut evaluation_json {
            map.insert("judge".to_string(), judge_result);
            map.insert("passed".to_string(), Value::Bool(passed));
        }
    }

    EvalOutcome {
        passed,
        evaluation_json: Some(evaluation_json.to_string()),
        output_text,
        status_code: Some(status_code),
        error: if status.is_success() {
            None
        } else {
            Some(format!("upstream returned HTTP {}", status_code))
        },
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cost: usage.cost,
        latency_ms,
    }
}

fn build_eval_request(
    uri: &Uri,
    prepared_headers: &PreparedHeaders,
    body: Bytes,
) -> Result<
    Request<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    Box<dyn std::error::Error>,
> {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri(uri.clone())
        .body(Full::new(body).map_err(|never| match never {}).boxed())?;
    request
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in &prepared_headers.0 {
        request.headers_mut().insert(name.clone(), value.clone());
    }
    Ok(request)
}

fn evaluate_expectation(
    expected_value: Option<&Value>,
    output_text: Option<&str>,
    response_json: Option<&Value>,
    status: StatusCode,
) -> (bool, Value) {
    let output = output_text.unwrap_or_default();
    let output_lower = output.to_lowercase();
    let Some(expected_value) = expected_value else {
        return (
            status.is_success(),
            json!({
                "kind": "status_only",
                "status_code": status.as_u16(),
                "passed": status.is_success(),
            }),
        );
    };

    let contains_terms = expected_value
        .get("contains")
        .and_then(|value| match value {
            Value::String(single) => Some(vec![single.clone()]),
            Value::Array(values) => Some(
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(ToString::to_string))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();
    let equals_value = expected_value
        .get("equals")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let not_contains_terms = expected_value
        .get("not_contains")
        .and_then(|value| match value {
            Value::String(single) => Some(vec![single.clone()]),
            Value::Array(values) => Some(
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(ToString::to_string))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();
    let starts_with = expected_value
        .get("starts_with")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let ends_with = expected_value
        .get("ends_with")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let regex_pattern = expected_value
        .get("regex")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let expected_status_code = expected_value
        .get("status_code")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok());
    let json_path_exists = string_or_array_values(expected_value.get("json_path_exists"));
    let json_path_equals = expected_value
        .get("json_path_equals")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let json_path_contains = expected_value
        .get("json_path_contains")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let structured_output = normalized_output_json(response_json, output_text);
    let structured_output_source = structured_output_source(response_json, output_text);
    let structured_matcher_defined = !json_path_exists.is_empty()
        || !json_path_equals.is_empty()
        || !json_path_contains.is_empty();

    let contains_pass = contains_terms
        .iter()
        .all(|term| output_lower.contains(&term.to_lowercase()));
    let not_contains_pass = not_contains_terms
        .iter()
        .all(|term| !output_lower.contains(&term.to_lowercase()));
    let equals_pass = equals_value
        .as_deref()
        .map(|value| output == value)
        .unwrap_or(true);
    let starts_with_pass = starts_with
        .as_deref()
        .map(|value| output_lower.starts_with(&value.to_lowercase()))
        .unwrap_or(true);
    let ends_with_pass = ends_with
        .as_deref()
        .map(|value| output_lower.ends_with(&value.to_lowercase()))
        .unwrap_or(true);
    let (regex_pass, regex_error) = match regex_pattern.as_deref() {
        Some(pattern) => match Regex::new(pattern) {
            Ok(regex) => (regex.is_match(output), None),
            Err(error) => (false, Some(error.to_string())),
        },
        None => (true, None),
    };
    let status_pass = expected_status_code
        .map(|value| status.as_u16() == value)
        .unwrap_or(true);
    let (json_path_exists_pass, json_path_exists_results) =
        evaluate_json_path_exists(structured_output.as_ref(), &json_path_exists);
    let (json_path_equals_pass, json_path_equals_results) =
        evaluate_json_path_equals(structured_output.as_ref(), &json_path_equals);
    let (json_path_contains_pass, json_path_contains_results) =
        evaluate_json_path_contains(structured_output.as_ref(), &json_path_contains);
    let base_status_pass = expected_status_code.is_some() || status.is_success();
    let matcher_defined = !contains_terms.is_empty()
        || !not_contains_terms.is_empty()
        || equals_value.is_some()
        || starts_with.is_some()
        || ends_with.is_some()
        || regex_pattern.is_some()
        || expected_status_code.is_some()
        || structured_matcher_defined;
    let passed = base_status_pass
        && (!matcher_defined
            || (contains_pass
                && not_contains_pass
                && equals_pass
                && starts_with_pass
                && ends_with_pass
                && regex_pass
                && status_pass
                && json_path_exists_pass
                && json_path_equals_pass
                && json_path_contains_pass));

    (
        passed,
        json!({
            "kind": "expectation",
            "status_code": status.as_u16(),
            "expected_status_code": expected_status_code,
            "contains": contains_terms,
            "not_contains": not_contains_terms,
            "equals": equals_value,
            "starts_with": starts_with,
            "ends_with": ends_with,
            "regex": regex_pattern,
            "regex_error": regex_error,
            "output_text": output_text,
            "structured_output_source": structured_output_source,
            "structured_output": structured_output.clone().unwrap_or(Value::Null),
            "json_path_exists": json_path_exists,
            "json_path_exists_results": json_path_exists_results,
            "json_path_equals": Value::Object(json_path_equals),
            "json_path_equals_results": json_path_equals_results,
            "json_path_contains": Value::Object(json_path_contains),
            "json_path_contains_results": json_path_contains_results,
            "passed": passed,
        }),
    )
}

fn parse_expected_output_json(expected_output_json: Option<&str>) -> Result<Option<Value>, String> {
    match expected_output_json {
        Some(expected_output_json) => serde_json::from_str::<Value>(expected_output_json)
            .map(Some)
            .map_err(|error| error.to_string()),
        None => Ok(None),
    }
}

fn judge_expectation_from_expected(expected_value: Option<&Value>) -> Option<JudgeExpectation> {
    let judge = expected_value?.get("judge")?.as_object()?;
    let rubric = judge.get("rubric")?.as_str()?.trim();
    if rubric.is_empty() {
        return None;
    }
    Some(JudgeExpectation {
        rubric: rubric.to_string(),
        min_score: judge.get("min_score").and_then(Value::as_f64),
        require_passed: judge
            .get("require_passed")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

async fn evaluate_with_judge(
    client: &proxy_core::handlers::proxy::HttpClient,
    judge: &PreparedJudge,
    item: &ProjectDatasetItemRecord,
    context_json: &Value,
    output_text: Option<&str>,
    output_json: Option<&Value>,
    status: StatusCode,
    expectation: &JudgeExpectation,
) -> Value {
    match judge.mode {
        JudgeMode::Webhook => {
            evaluate_with_webhook_judge(
                client,
                judge,
                item,
                context_json,
                output_text,
                output_json,
                status,
                expectation,
            )
            .await
        }
        JudgeMode::OpenAi => {
            evaluate_with_openai_judge(
                client,
                judge,
                item,
                context_json,
                output_text,
                output_json,
                status,
                expectation,
            )
            .await
        }
        JudgeMode::Anthropic => {
            evaluate_with_anthropic_judge(
                client,
                judge,
                item,
                context_json,
                output_text,
                output_json,
                status,
                expectation,
            )
            .await
        }
    }
}

async fn evaluate_with_webhook_judge(
    client: &proxy_core::handlers::proxy::HttpClient,
    judge: &PreparedJudge,
    item: &ProjectDatasetItemRecord,
    context_json: &Value,
    output_text: Option<&str>,
    output_json: Option<&Value>,
    status: StatusCode,
    expectation: &JudgeExpectation,
) -> Value {
    let request_body = json!({
        "item_id": item.item_id.clone(),
        "input": serde_json::from_str::<Value>(&item.input_json).unwrap_or_else(|_| Value::String(item.input_json.clone())),
        "metadata": item
            .metadata_json
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .unwrap_or(Value::Null),
        "output_text": output_text,
        "output_json": output_json.cloned().unwrap_or(Value::Null),
        "status_code": status.as_u16(),
        "context": context_json,
        "judge": {
            "rubric": expectation.rubric.clone(),
            "min_score": expectation.min_score,
            "require_passed": expectation.require_passed,
        }
    });
    let request = match build_eval_request(
        &judge.uri,
        &judge.headers,
        Bytes::from(request_body.to_string()),
    ) {
        Ok(request) => request,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };

    let response = match timeout(judge.request_timeout, client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
        Err(_) => {
            return json!({
                "kind": "judge_timeout",
                "timeout_ms": judge.request_timeout.as_millis() as u64,
                "passed": false,
            });
        }
    };

    let response_status = response.status();
    let body_bytes = match response.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };
    let response_json = match serde_json::from_slice::<Value>(&body_bytes) {
        Ok(response_json) => response_json,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": format!("invalid judge response: {}", error),
                "judge_status_code": response_status.as_u16(),
                "passed": false,
            });
        }
    };
    if !response_status.is_success() {
        return json!({
            "kind": "judge_error",
            "message": format!("judge returned HTTP {}", response_status.as_u16()),
            "judge_status_code": response_status.as_u16(),
            "response": response_json,
            "passed": false,
        });
    }

    let response_passed = response_json.get("passed").and_then(Value::as_bool);
    let score = response_json.get("score").and_then(Value::as_f64);
    let score_passed = expectation
        .min_score
        .map(|min_score| score.map(|score| score >= min_score).unwrap_or(false))
        .unwrap_or(true);
    let explicit_passed = if expectation.require_passed {
        response_passed.unwrap_or(false)
    } else {
        true
    };
    let passed = explicit_passed && score_passed;

    json!({
        "kind": "judge",
        "rubric": expectation.rubric.clone(),
        "min_score": expectation.min_score,
        "require_passed": expectation.require_passed,
        "judge_status_code": response_status.as_u16(),
        "score": score,
        "judge_passed": response_passed,
        "response": response_json,
        "passed": passed,
    })
}

async fn evaluate_with_openai_judge(
    client: &proxy_core::handlers::proxy::HttpClient,
    judge: &PreparedJudge,
    item: &ProjectDatasetItemRecord,
    context_json: &Value,
    output_text: Option<&str>,
    output_json: Option<&Value>,
    status: StatusCode,
    expectation: &JudgeExpectation,
) -> Value {
    let Some(model) = judge.model.as_deref() else {
        return json!({
            "kind": "judge_error",
            "message": "judge_model is required when judge_kind is 'openai'",
            "passed": false,
        });
    };
    let input_value = serde_json::from_str::<Value>(&item.input_json)
        .unwrap_or_else(|_| Value::String(item.input_json.clone()));
    let metadata_value = item
        .metadata_json
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .unwrap_or(Value::Null);
    let judge_prompt = json!({
        "rubric": expectation.rubric,
        "min_score": expectation.min_score,
        "require_passed": expectation.require_passed,
        "item_id": item.item_id,
        "input": input_value,
        "metadata": metadata_value,
        "output_text": output_text,
        "output_json": output_json.cloned().unwrap_or(Value::Null),
        "status_code": status.as_u16(),
        "context": context_json,
    });
    let request_body = json!({
        "model": model,
        "temperature": 0,
        "response_format": { "type": "json_object" },
        "messages": [
            {
                "role": "system",
                "content": "You are an evaluation judge. Return JSON only with keys passed, score, and reasoning."
            },
            {
                "role": "user",
                "content": judge_prompt.to_string()
            }
        ]
    });
    let request = match build_eval_request(
        &judge.uri,
        &judge.headers,
        Bytes::from(request_body.to_string()),
    ) {
        Ok(request) => request,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };

    let response = match timeout(judge.request_timeout, client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
        Err(_) => {
            return json!({
                "kind": "judge_timeout",
                "timeout_ms": judge.request_timeout.as_millis() as u64,
                "passed": false,
            });
        }
    };

    let response_status = response.status();
    let body_bytes = match response.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };
    let response_json = match serde_json::from_slice::<Value>(&body_bytes) {
        Ok(response_json) => response_json,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": format!("invalid OpenAI judge response: {}", error),
                "judge_status_code": response_status.as_u16(),
                "passed": false,
            });
        }
    };
    if !response_status.is_success() {
        return json!({
            "kind": "judge_error",
            "message": format!("judge returned HTTP {}", response_status.as_u16()),
            "judge_status_code": response_status.as_u16(),
            "response": response_json,
            "passed": false,
        });
    }

    let Some(content) = extract_openai_judge_content(&response_json) else {
        return json!({
            "kind": "judge_error",
            "message": "OpenAI judge response was missing choices[0].message.content",
            "judge_status_code": response_status.as_u16(),
            "response": response_json,
            "passed": false,
        });
    };
    let parsed_response = match serde_json::from_str::<Value>(&content) {
        Ok(parsed) => parsed,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": format!("invalid OpenAI judge JSON payload: {}", error),
                "judge_status_code": response_status.as_u16(),
                "response": response_json,
                "passed": false,
            });
        }
    };

    let response_passed = parsed_response.get("passed").and_then(Value::as_bool);
    let score = parsed_response.get("score").and_then(Value::as_f64);
    let score_passed = expectation
        .min_score
        .map(|min_score| score.map(|score| score >= min_score).unwrap_or(false))
        .unwrap_or(true);
    let explicit_passed = if expectation.require_passed {
        response_passed.unwrap_or(false)
    } else {
        true
    };
    let passed = explicit_passed && score_passed;

    json!({
        "kind": "judge_openai",
        "rubric": expectation.rubric.clone(),
        "model": model,
        "min_score": expectation.min_score,
        "require_passed": expectation.require_passed,
        "judge_status_code": response_status.as_u16(),
        "score": score,
        "judge_passed": response_passed,
        "parsed_response": parsed_response,
        "response": response_json,
        "passed": passed,
    })
}

async fn evaluate_with_anthropic_judge(
    client: &proxy_core::handlers::proxy::HttpClient,
    judge: &PreparedJudge,
    item: &ProjectDatasetItemRecord,
    context_json: &Value,
    output_text: Option<&str>,
    output_json: Option<&Value>,
    status: StatusCode,
    expectation: &JudgeExpectation,
) -> Value {
    let Some(model) = judge.model.as_deref() else {
        return json!({
            "kind": "judge_error",
            "message": "judge_model is required when judge_kind is 'anthropic'",
            "passed": false,
        });
    };
    let input_value = serde_json::from_str::<Value>(&item.input_json)
        .unwrap_or_else(|_| Value::String(item.input_json.clone()));
    let metadata_value = item
        .metadata_json
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .unwrap_or(Value::Null);
    let judge_prompt = json!({
        "rubric": expectation.rubric,
        "min_score": expectation.min_score,
        "require_passed": expectation.require_passed,
        "item_id": item.item_id,
        "input": input_value,
        "metadata": metadata_value,
        "output_text": output_text,
        "output_json": output_json.cloned().unwrap_or(Value::Null),
        "status_code": status.as_u16(),
        "context": context_json,
    });
    let request_body = json!({
        "model": model,
        "max_tokens": 256,
        "temperature": 0,
        "system": "You are an evaluation judge. Return JSON only with keys passed, score, and reasoning.",
        "messages": [
            {
                "role": "user",
                "content": judge_prompt.to_string()
            }
        ]
    });
    let request = match build_eval_request(
        &judge.uri,
        &judge.headers,
        Bytes::from(request_body.to_string()),
    ) {
        Ok(request) => request,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };

    let response = match timeout(judge.request_timeout, client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
        Err(_) => {
            return json!({
                "kind": "judge_timeout",
                "timeout_ms": judge.request_timeout.as_millis() as u64,
                "passed": false,
            });
        }
    };

    let response_status = response.status();
    let body_bytes = match response.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": error.to_string(),
                "passed": false,
            });
        }
    };
    let response_json = match serde_json::from_slice::<Value>(&body_bytes) {
        Ok(response_json) => response_json,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": format!("invalid Anthropic judge response: {}", error),
                "judge_status_code": response_status.as_u16(),
                "passed": false,
            });
        }
    };
    if !response_status.is_success() {
        return json!({
            "kind": "judge_error",
            "message": format!("judge returned HTTP {}", response_status.as_u16()),
            "judge_status_code": response_status.as_u16(),
            "response": response_json,
            "passed": false,
        });
    }

    let Some(content) = extract_anthropic_judge_content(&response_json) else {
        return json!({
            "kind": "judge_error",
            "message": "Anthropic judge response was missing content text",
            "judge_status_code": response_status.as_u16(),
            "response": response_json,
            "passed": false,
        });
    };
    let parsed_response = match serde_json::from_str::<Value>(&content) {
        Ok(parsed) => parsed,
        Err(error) => {
            return json!({
                "kind": "judge_error",
                "message": format!("invalid Anthropic judge JSON payload: {}", error),
                "judge_status_code": response_status.as_u16(),
                "response": response_json,
                "passed": false,
            });
        }
    };

    let response_passed = parsed_response.get("passed").and_then(Value::as_bool);
    let score = parsed_response.get("score").and_then(Value::as_f64);
    let score_passed = expectation
        .min_score
        .map(|min_score| score.map(|score| score >= min_score).unwrap_or(false))
        .unwrap_or(true);
    let explicit_passed = if expectation.require_passed {
        response_passed.unwrap_or(false)
    } else {
        true
    };
    let passed = explicit_passed && score_passed;

    json!({
        "kind": "judge_anthropic",
        "rubric": expectation.rubric.clone(),
        "model": model,
        "min_score": expectation.min_score,
        "require_passed": expectation.require_passed,
        "judge_status_code": response_status.as_u16(),
        "score": score,
        "judge_passed": response_passed,
        "parsed_response": parsed_response,
        "response": response_json,
        "passed": passed,
    })
}

fn extract_openai_judge_content(response_json: &Value) -> Option<String> {
    let message_content = response_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))?;
    if let Some(content) = message_content.as_str() {
        return Some(content.to_string());
    }
    let parts = message_content.as_array()?;
    let text = parts
        .iter()
        .filter_map(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .or_else(|| part.get("content").and_then(Value::as_str))
        })
        .collect::<String>();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn extract_anthropic_judge_content(response_json: &Value) -> Option<String> {
    if let Some(content) = response_json.get("content").and_then(Value::as_str) {
        let trimmed = content.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let parts = response_json.get("content")?.as_array()?;
    let text = parts
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn string_or_array_values(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::String(single)) => vec![single.clone()],
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str().map(ToString::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn structured_output_source(
    response_json: Option<&Value>,
    output_text: Option<&str>,
) -> Option<&'static str> {
    let Some(response_json) = response_json else {
        return output_text
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .map(|_| "output_text_json");
    };
    if response_json.get("output_json").is_some() {
        return Some("output_json");
    }
    if let Some(value) = extract_output_value(response_json) {
        return match value {
            Value::Object(_) | Value::Array(_) => Some("provider_output_value"),
            Value::String(text) if serde_json::from_str::<Value>(text).is_ok() => {
                Some("provider_output_text_json")
            }
            _ => output_text
                .and_then(|text| serde_json::from_str::<Value>(text).ok())
                .map(|_| "output_text_json"),
        };
    }
    output_text
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .map(|_| "output_text_json")
}

fn normalized_output_json(
    response_json: Option<&Value>,
    output_text: Option<&str>,
) -> Option<Value> {
    if let Some(response_json) = response_json {
        if let Some(output_json) = response_json.get("output_json") {
            return Some(output_json.clone());
        }
        if let Some(value) = extract_output_value(response_json) {
            match value {
                Value::Object(_) | Value::Array(_) => return Some(value.clone()),
                Value::String(text) => {
                    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                        return Some(parsed);
                    }
                }
                _ => {}
            }
        }
    }
    output_text.and_then(|text| serde_json::from_str::<Value>(text).ok())
}

fn extract_output_value(response_json: &Value) -> Option<&Value> {
    if let Some(content) = response_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
    {
        return Some(content);
    }

    if let Some(content) = response_json.get("content") {
        return Some(content);
    }

    response_json.get("output_json")
}

fn evaluate_json_path_exists(output_json: Option<&Value>, paths: &[String]) -> (bool, Value) {
    if paths.is_empty() {
        return (true, Value::Object(Default::default()));
    }
    let Some(output_json) = output_json else {
        let results = paths
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    json!({"passed": false, "error": "structured output unavailable"}),
                )
            })
            .collect();
        return (false, Value::Object(results));
    };

    let mut passed = true;
    let mut results = serde_json::Map::new();
    for path in paths {
        let exists = resolve_json_path(output_json, path).is_some();
        if !exists {
            passed = false;
        }
        results.insert(path.clone(), json!({"passed": exists}));
    }
    (passed, Value::Object(results))
}

fn evaluate_json_path_equals(
    output_json: Option<&Value>,
    expectations: &serde_json::Map<String, Value>,
) -> (bool, Value) {
    if expectations.is_empty() {
        return (true, Value::Object(Default::default()));
    }
    let Some(output_json) = output_json else {
        let results = expectations
            .keys()
            .map(|path| {
                (
                    path.clone(),
                    json!({"passed": false, "error": "structured output unavailable"}),
                )
            })
            .collect();
        return (false, Value::Object(results));
    };

    let mut passed = true;
    let mut results = serde_json::Map::new();
    for (path, expected_value) in expectations {
        let actual = resolve_json_path(output_json, path);
        let path_passed = actual == Some(expected_value);
        if !path_passed {
            passed = false;
        }
        results.insert(
            path.clone(),
            json!({
                "passed": path_passed,
                "expected": expected_value,
                "actual": actual.cloned().unwrap_or(Value::Null),
            }),
        );
    }
    (passed, Value::Object(results))
}

fn evaluate_json_path_contains(
    output_json: Option<&Value>,
    expectations: &serde_json::Map<String, Value>,
) -> (bool, Value) {
    if expectations.is_empty() {
        return (true, Value::Object(Default::default()));
    }
    let Some(output_json) = output_json else {
        let results = expectations
            .keys()
            .map(|path| {
                (
                    path.clone(),
                    json!({"passed": false, "error": "structured output unavailable"}),
                )
            })
            .collect();
        return (false, Value::Object(results));
    };

    let mut passed = true;
    let mut results = serde_json::Map::new();
    for (path, expected_value) in expectations {
        let expected_terms = string_or_array_values(Some(expected_value));
        let actual = resolve_json_path(output_json, path).cloned();
        let actual_text = actual.as_ref().map(json_value_to_text).unwrap_or_default();
        let actual_text_lower = actual_text.to_lowercase();
        let path_passed = !expected_terms.is_empty()
            && expected_terms
                .iter()
                .all(|term| actual_text_lower.contains(&term.to_lowercase()));
        if !path_passed {
            passed = false;
        }
        results.insert(
            path.clone(),
            json!({
                "passed": path_passed,
                "expected_terms": expected_terms,
                "actual": actual.unwrap_or(Value::Null),
            }),
        );
    }
    (passed, Value::Object(results))
}

fn resolve_json_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(value);
    }
    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            return None;
        }
        current = match current {
            Value::Object(map) => map.get(segment)?,
            Value::Array(values) => {
                let index = segment.parse::<usize>().ok()?;
                values.get(index)?
            }
            _ => return None,
        };
    }
    Some(current)
}

fn json_value_to_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn comparison_gate_requested(request: &ProjectEvalRunComparisonGateRequest) -> bool {
    !request.is_empty()
}

impl ProjectEvalRunComparisonGatePreset {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Balanced => "balanced",
            Self::Exploratory => "exploratory",
        }
    }

    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value {
            "strict" => Some(Self::Strict),
            "balanced" => Some(Self::Balanced),
            "exploratory" => Some(Self::Exploratory),
            _ => None,
        }
    }

    fn default_thresholds(self) -> ProjectEvalRunComparisonGateRequest {
        match self {
            Self::Strict => ProjectEvalRunComparisonGateRequest {
                preset: Some(self),
                max_regressions: Some(0),
                min_candidate_pass_rate: None,
                min_pass_rate_delta: Some(0.0),
                max_latency_increase_ms: Some(25.0),
                max_cost_increase: Some(0.01),
            },
            Self::Balanced => ProjectEvalRunComparisonGateRequest {
                preset: Some(self),
                max_regressions: Some(1),
                min_candidate_pass_rate: None,
                min_pass_rate_delta: Some(-0.02),
                max_latency_increase_ms: Some(75.0),
                max_cost_increase: Some(0.05),
            },
            Self::Exploratory => ProjectEvalRunComparisonGateRequest {
                preset: Some(self),
                max_regressions: Some(2),
                min_candidate_pass_rate: None,
                min_pass_rate_delta: Some(-0.05),
                max_latency_increase_ms: Some(250.0),
                max_cost_increase: Some(0.25),
            },
        }
    }
}

impl ProjectEvalRunComparisonGateRequest {
    pub fn is_empty(&self) -> bool {
        self.preset.is_none()
            && self.max_regressions.is_none()
            && self.min_candidate_pass_rate.is_none()
            && self.min_pass_rate_delta.is_none()
            && self.max_latency_increase_ms.is_none()
            && self.max_cost_increase.is_none()
    }

    pub fn merged_with(&self, overrides: &Self) -> Self {
        Self {
            preset: overrides.preset.or(self.preset),
            max_regressions: overrides.max_regressions.or(self.max_regressions),
            min_candidate_pass_rate: overrides
                .min_candidate_pass_rate
                .or(self.min_candidate_pass_rate),
            min_pass_rate_delta: overrides.min_pass_rate_delta.or(self.min_pass_rate_delta),
            max_latency_increase_ms: overrides
                .max_latency_increase_ms
                .or(self.max_latency_increase_ms),
            max_cost_increase: overrides.max_cost_increase.or(self.max_cost_increase),
        }
    }
}

impl ProjectEvalRunRolloutAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Promote => "promote",
            Self::Canary => "canary",
            Self::Review => "review",
            Self::Hold => "hold",
        }
    }
}

fn build_comparison_context(
    baseline_run: &ProjectEvalRunRecord,
    candidate_run: &ProjectEvalRunRecord,
) -> ProjectEvalRunComparisonContext {
    let baseline = extract_eval_context_from_run(baseline_run);
    let candidate = extract_eval_context_from_run(candidate_run);
    let baseline_object = baseline.as_object().cloned().unwrap_or_default();
    let candidate_object = candidate.as_object().cloned().unwrap_or_default();
    let mut changed_fields = baseline_object
        .keys()
        .chain(candidate_object.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|key| baseline_object.get(key) != candidate_object.get(key))
        .collect::<Vec<_>>();
    changed_fields.sort();
    ProjectEvalRunComparisonContext {
        baseline: Value::Object(baseline_object),
        candidate: Value::Object(candidate_object),
        changed_fields,
    }
}

fn extract_eval_context_from_run(run: &ProjectEvalRunRecord) -> Value {
    parse_eval_summary(run.summary_json.as_deref())
        .and_then(|summary| summary.get("context").cloned())
        .unwrap_or_else(|| Value::Object(Default::default()))
}

fn resolve_comparison_gate_thresholds(
    request: ProjectEvalRunComparisonGateRequest,
) -> ProjectEvalRunComparisonGateRequest {
    let mut effective = request
        .preset
        .map(ProjectEvalRunComparisonGatePreset::default_thresholds)
        .unwrap_or_default();
    effective.preset = request.preset;
    if request.max_regressions.is_some() {
        effective.max_regressions = request.max_regressions;
    }
    if request.min_candidate_pass_rate.is_some() {
        effective.min_candidate_pass_rate = request.min_candidate_pass_rate;
    }
    if request.min_pass_rate_delta.is_some() {
        effective.min_pass_rate_delta = request.min_pass_rate_delta;
    }
    if request.max_latency_increase_ms.is_some() {
        effective.max_latency_increase_ms = request.max_latency_increase_ms;
    }
    if request.max_cost_increase.is_some() {
        effective.max_cost_increase = request.max_cost_increase;
    }
    effective
}

fn evaluate_comparison_gate(
    summary: &ProjectEvalRunComparisonSummary,
    context: &ProjectEvalRunComparisonContext,
    request: ProjectEvalRunComparisonGateRequest,
) -> ProjectEvalRunComparisonGate {
    let thresholds = resolve_comparison_gate_thresholds(request);
    let mut reasons = Vec::new();
    if let Some(max_regressions) = thresholds.max_regressions {
        if summary.regressed_items > max_regressions {
            reasons.push(format!(
                "regressed_items {} exceeded max_regressions {}",
                summary.regressed_items, max_regressions
            ));
        }
    }
    if let Some(min_candidate_pass_rate) = thresholds.min_candidate_pass_rate {
        if summary.candidate_pass_rate < min_candidate_pass_rate {
            reasons.push(format!(
                "candidate_pass_rate {:.4} was below min_candidate_pass_rate {:.4}",
                summary.candidate_pass_rate, min_candidate_pass_rate
            ));
        }
    }
    if let Some(min_pass_rate_delta) = thresholds.min_pass_rate_delta {
        if summary.delta_pass_rate < min_pass_rate_delta {
            reasons.push(format!(
                "delta_pass_rate {:.4} was below min_pass_rate_delta {:.4}",
                summary.delta_pass_rate, min_pass_rate_delta
            ));
        }
    }
    if let Some(max_latency_increase_ms) = thresholds.max_latency_increase_ms {
        if summary.delta_average_latency_ms > max_latency_increase_ms {
            reasons.push(format!(
                "delta_average_latency_ms {:.4} exceeded max_latency_increase_ms {:.4}",
                summary.delta_average_latency_ms, max_latency_increase_ms
            ));
        }
    }
    if let Some(max_cost_increase) = thresholds.max_cost_increase {
        if summary.delta_total_cost > max_cost_increase {
            reasons.push(format!(
                "delta_total_cost {:.6} exceeded max_cost_increase {:.6}",
                summary.delta_total_cost, max_cost_increase
            ));
        }
    }
    let passed = reasons.is_empty();
    let recommendation = match (thresholds.preset, passed) {
        (Some(ProjectEvalRunComparisonGatePreset::Strict), true) => {
            ProjectEvalRunRolloutRecommendation {
                action: ProjectEvalRunRolloutAction::Promote,
                summary: "Candidate met the strict rollout gate.".to_string(),
                changed_context_fields: context.changed_fields.clone(),
            }
        }
        (Some(ProjectEvalRunComparisonGatePreset::Balanced), true) => {
            ProjectEvalRunRolloutRecommendation {
                action: ProjectEvalRunRolloutAction::Canary,
                summary: "Candidate met the balanced rollout gate.".to_string(),
                changed_context_fields: context.changed_fields.clone(),
            }
        }
        (Some(ProjectEvalRunComparisonGatePreset::Exploratory), true) => {
            ProjectEvalRunRolloutRecommendation {
                action: ProjectEvalRunRolloutAction::Review,
                summary: "Candidate met the exploratory rollout gate.".to_string(),
                changed_context_fields: context.changed_fields.clone(),
            }
        }
        (Some(preset), false) => ProjectEvalRunRolloutRecommendation {
            action: ProjectEvalRunRolloutAction::Hold,
            summary: format!("Candidate failed the {} rollout gate.", preset.as_str()),
            changed_context_fields: context.changed_fields.clone(),
        },
        (None, true) => ProjectEvalRunRolloutRecommendation {
            action: ProjectEvalRunRolloutAction::Review,
            summary: "Candidate met the requested comparison gate.".to_string(),
            changed_context_fields: context.changed_fields.clone(),
        },
        (None, false) => ProjectEvalRunRolloutRecommendation {
            action: ProjectEvalRunRolloutAction::Hold,
            summary: "Candidate failed the requested comparison gate.".to_string(),
            changed_context_fields: context.changed_fields.clone(),
        },
    };

    ProjectEvalRunComparisonGate {
        passed,
        reasons,
        preset: thresholds.preset,
        thresholds,
        recommendation,
    }
}

fn extract_output_text(response_json: &Value) -> Option<String> {
    if let Some(content) = response_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
    {
        return value_to_text(content);
    }

    if let Some(text) = response_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
    {
        return Some(text.to_string());
    }

    if let Some(content) = response_json.get("content") {
        return value_to_text(content);
    }

    response_json
        .get("output_text")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let segments = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(text) => Some(text.clone()),
                    Value::Object(map) => map
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("\n"))
            }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

fn extract_usage(response_json: &Value) -> ExtractedUsage {
    let usage = response_json.get("usage").unwrap_or(&Value::Null);
    let input_tokens = extract_u64(usage.get("prompt_tokens"))
        .or_else(|| extract_u64(usage.get("input_tokens")))
        .unwrap_or(0);
    let output_tokens = extract_u64(usage.get("completion_tokens"))
        .or_else(|| extract_u64(usage.get("output_tokens")))
        .unwrap_or(0);
    let cost = usage
        .get("total_cost")
        .and_then(Value::as_f64)
        .or_else(|| response_json.get("cost").and_then(Value::as_f64))
        .unwrap_or(0.0);

    ExtractedUsage {
        input_tokens,
        output_tokens,
        cost,
    }
}

fn extract_u64(value: Option<&Value>) -> Option<u64> {
    value.and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|number| u64::try_from(number).ok()))
    })
}

fn generate_eval_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = EVAL_RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("eval-{nanos}-{sequence}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_matchers_pass_when_output_matches() {
        let (passed, evaluation_json) = evaluate_expectation(
            Some(&json!({
                "contains": ["reset", "link"],
                "not_contains": "support",
                "starts_with": "Use",
                "ends_with": "1234.",
                "regex": "reset link \\d{4}\\.$",
                "status_code": 200
            })),
            Some("Use reset link 1234."),
            None,
            StatusCode::OK,
        );
        assert!(passed);
        let evaluation = evaluation_json;
        assert_eq!(evaluation["passed"].as_bool(), Some(true));
        assert_eq!(evaluation["expected_status_code"].as_u64(), Some(200));
    }

    #[test]
    fn regex_matcher_reports_invalid_patterns() {
        let (passed, evaluation_json) = evaluate_expectation(
            Some(&json!({ "regex": "[" })),
            Some("anything"),
            None,
            StatusCode::OK,
        );
        assert!(!passed);
        let evaluation = evaluation_json;
        assert!(evaluation["regex_error"].as_str().is_some());
    }

    #[test]
    fn structured_output_matchers_pass_against_output_json() {
        let response_json = json!({
            "output_json": {
                "decision": {
                    "approved": true,
                    "reason": "Reset link sent to the account owner"
                },
                "citations": ["kb-123"]
            }
        });
        let (passed, evaluation_json) = evaluate_expectation(
            Some(&json!({
                "json_path_exists": ["decision.approved", "citations.0"],
                "json_path_equals": {
                    "decision.approved": true,
                    "citations.0": "kb-123"
                },
                "json_path_contains": {
                    "decision.reason": "reset link"
                },
                "status_code": 200
            })),
            None,
            Some(&response_json),
            StatusCode::OK,
        );
        assert!(passed);
        let evaluation = evaluation_json;
        assert_eq!(
            evaluation["structured_output_source"].as_str(),
            Some("output_json")
        );
        assert_eq!(
            evaluation["json_path_equals_results"]["decision.approved"]["passed"].as_bool(),
            Some(true)
        );
        assert_eq!(
            evaluation["json_path_contains_results"]["decision.reason"]["passed"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn judge_expectation_is_extracted_from_expected_output() {
        let expectation = judge_expectation_from_expected(Some(&json!({
            "judge": {
                "rubric": "Does the answer solve the task?",
                "min_score": 0.8,
                "require_passed": false
            }
        })))
        .expect("judge expectation");
        assert_eq!(expectation.rubric, "Does the answer solve the task?");
        assert_eq!(expectation.min_score, Some(0.8));
        assert!(!expectation.require_passed);
    }

    #[test]
    fn comparison_gate_reports_threshold_failures() {
        let gate = evaluate_comparison_gate(
            &ProjectEvalRunComparisonSummary {
                baseline_pass_rate: 1.0,
                candidate_pass_rate: 0.0,
                delta_pass_rate: -1.0,
                baseline_average_latency_ms: 10.0,
                candidate_average_latency_ms: 35.0,
                delta_average_latency_ms: 25.0,
                baseline_total_cost: 0.01,
                candidate_total_cost: 0.05,
                delta_total_cost: 0.04,
                improved_items: 0,
                regressed_items: 1,
                unchanged_items: 0,
            },
            &ProjectEvalRunComparisonContext {
                baseline: json!({"prompt_version":"v1"}),
                candidate: json!({"prompt_version":"v2"}),
                changed_fields: vec!["prompt_version".to_string()],
            },
            ProjectEvalRunComparisonGateRequest {
                preset: None,
                max_regressions: Some(0),
                min_candidate_pass_rate: Some(0.8),
                min_pass_rate_delta: Some(-0.1),
                max_latency_increase_ms: Some(10.0),
                max_cost_increase: Some(0.01),
            },
        );
        assert!(!gate.passed);
        assert_eq!(gate.reasons.len(), 5);
        assert_eq!(
            gate.recommendation.action,
            ProjectEvalRunRolloutAction::Hold
        );
        assert_eq!(
            gate.recommendation.changed_context_fields,
            vec!["prompt_version".to_string()]
        );
    }

    #[test]
    fn comparison_gate_preset_applies_effective_thresholds_and_rollout_action() {
        let gate = evaluate_comparison_gate(
            &ProjectEvalRunComparisonSummary {
                baseline_pass_rate: 0.9,
                candidate_pass_rate: 0.92,
                delta_pass_rate: 0.02,
                baseline_average_latency_ms: 20.0,
                candidate_average_latency_ms: 35.0,
                delta_average_latency_ms: 15.0,
                baseline_total_cost: 0.03,
                candidate_total_cost: 0.035,
                delta_total_cost: 0.005,
                improved_items: 1,
                regressed_items: 0,
                unchanged_items: 9,
            },
            &ProjectEvalRunComparisonContext {
                baseline: json!({"prompt_version":"v1"}),
                candidate: json!({"prompt_version":"v2","provider_name":"anthropic"}),
                changed_fields: vec!["prompt_version".to_string(), "provider_name".to_string()],
            },
            ProjectEvalRunComparisonGateRequest {
                preset: Some(ProjectEvalRunComparisonGatePreset::Strict),
                ..Default::default()
            },
        );
        assert!(gate.passed);
        assert_eq!(
            gate.preset,
            Some(ProjectEvalRunComparisonGatePreset::Strict)
        );
        assert_eq!(gate.thresholds.max_regressions, Some(0));
        assert_eq!(gate.thresholds.min_pass_rate_delta, Some(0.0));
        assert_eq!(gate.thresholds.max_latency_increase_ms, Some(25.0));
        assert_eq!(gate.thresholds.max_cost_increase, Some(0.01));
        assert_eq!(
            gate.recommendation.action,
            ProjectEvalRunRolloutAction::Promote
        );
        assert_eq!(
            gate.recommendation.changed_context_fields,
            vec!["prompt_version".to_string(), "provider_name".to_string()]
        );
    }
}
