use std::sync::Arc;
use std::time::Duration;

use async_recursion::async_recursion;
use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use regex::Regex;
use serde_json::Value;

use proxy_core::plugin::{Action, Plugin, RequestContext, ResponseContext};

use crate::governance::{GovernanceState, SafetyMode};
use crate::tool_runtime::ForwardedRequestBody;

type VerificationClient = Client<
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>,
    http_body_util::combinators::BoxBody<Bytes, hyper::Error>,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentFilterAction {
    Block,
    Log,
    RedactAndForward,
}

#[derive(Debug, Clone)]
pub struct FilterRule {
    pub pattern: Option<String>,
    pub description: Option<String>,
    pub detector_class: Option<String>,
    pub action: Option<String>,
    pub verification: Option<String>,
    pub replacement: Option<String>,
    pub path_patterns: Vec<String>,
    pub allowlist_patterns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyMatch {
    pub detector_class: String,
    pub description: String,
    pub path: String,
    pub action: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyAudit {
    pub mode: String,
    pub matches: Vec<SafetyMatch>,
}

#[derive(Clone)]
pub struct ContentFilterReplayHandle(ContentFilter);

impl ContentFilterReplayHandle {
    pub async fn on_synthetic_request(&self, ctx: &mut RequestContext) -> Action {
        self.0.on_request(ctx).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveDetectorConfig {
    pub detector_class: String,
    pub display_name: String,
    pub provider: Option<String>,
    pub category: String,
    pub source: String,
    pub source_url: Option<String>,
    pub effective_action: String,
    pub action_source: String,
    pub verification_mode: String,
    pub verifier_kind: Option<String>,
    pub remote_verifier_kind: Option<String>,
    pub replacement: Option<String>,
    pub path_patterns: Vec<String>,
    pub allowlist_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RemoteVerifierConfig {
    pub github_user_url: String,
    pub slack_auth_test_url: String,
    pub anthropic_models_url: String,
    pub timeout: Duration,
}

impl Default for RemoteVerifierConfig {
    fn default() -> Self {
        Self {
            github_user_url: "https://api.github.com/user".to_string(),
            slack_auth_test_url: "https://slack.com/api/auth.test".to_string(),
            anthropic_models_url: "https://api.anthropic.com/v1/models".to_string(),
            timeout: Duration::from_millis(1500),
        }
    }
}

#[derive(Clone, Copy)]
struct DetectorDefinition {
    detector_class: &'static str,
    display_name: &'static str,
    provider: Option<&'static str>,
    category: &'static str,
    pattern: &'static str,
    source: &'static str,
    source_url: Option<&'static str>,
    verifier_kind: Option<VerifierKind>,
    remote_verifier_kind: Option<RemoteVerifierKind>,
    default_verification: VerificationMode,
}

#[derive(Clone)]
struct CompiledRule {
    regex: Regex,
    description: String,
    detector_class: String,
    display_name: String,
    provider: Option<String>,
    category: String,
    source: String,
    source_url: Option<String>,
    action: Option<EnforcementAction>,
    verification: VerificationMode,
    verifier_kind: Option<VerifierKind>,
    remote_verifier_kind: Option<RemoteVerifierKind>,
    replacement: Option<String>,
    action_source: ActionSource,
    path_patterns: Vec<Regex>,
    path_patterns_raw: Vec<String>,
    allowlist_patterns: Vec<Regex>,
    allowlist_patterns_raw: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EnforcementAction {
    Allow,
    ObserveOnly,
    Redact,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationMode {
    Default,
    Disabled,
    Local,
    Remote,
}

impl VerificationMode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "default" => Some(Self::Default),
            "disabled" | "off" => Some(Self::Disabled),
            "local" | "enabled" => Some(Self::Local),
            "remote" => Some(Self::Remote),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Disabled => "disabled",
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifierKind {
    AwsExampleGuard,
    SlackExampleGuard,
    StripeExampleGuard,
}

impl VerifierKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::AwsExampleGuard => "aws_example_guard",
            Self::SlackExampleGuard => "slack_example_guard",
            Self::StripeExampleGuard => "stripe_example_guard",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteVerifierKind {
    GitHubUserApi,
    SlackAuthTestApi,
    AnthropicModelsApi,
}

impl RemoteVerifierKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::GitHubUserApi => "github_user_api",
            Self::SlackAuthTestApi => "slack_auth_test_api",
            Self::AnthropicModelsApi => "anthropic_models_api",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionSource {
    ProjectDefault,
    RuleOverride,
}

impl ActionSource {
    fn as_str(&self) -> &'static str {
        match self {
            Self::ProjectDefault => "project_default",
            Self::RuleOverride => "rule_override",
        }
    }
}

impl EnforcementAction {
    fn from_mode(mode: SafetyMode) -> Self {
        match mode {
            SafetyMode::Block => Self::Block,
            SafetyMode::ObserveOnly => Self::ObserveOnly,
            SafetyMode::RedactAndForward => Self::Redact,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "allow" => Some(Self::Allow),
            "observe_only" | "log" => Some(Self::ObserveOnly),
            "redact" | "redact_and_forward" => Some(Self::Redact),
            "block" => Some(Self::Block),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::ObserveOnly => "observe_only",
            Self::Redact => "redact_and_forward",
            Self::Block => "block",
        }
    }
}

const GITHUB_SUPPORTED_PATTERNS_URL: &str =
    "https://docs.github.com/code-security/secret-scanning/secret-scanning-patterns";
const GITHUB_AUTH_URL: &str =
    "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github";
const SLACK_WEB_API_URL: &str =
    "https://docs.slack.dev/tools/java-slack-sdk/guides/web-api-basics/";
const SLACK_WEBHOOK_URL: &str =
    "https://docs.slack.dev/tools/java-slack-sdk/guides/incoming-webhooks/";
const STRIPE_KEYS_URL: &str = "https://docs.stripe.com/keys";
const CLAUDE_ADMIN_API_URL: &str =
    "https://platform.claude.com/docs/en/build-with-claude/administration-api";
const CLAUDE_API_KEY_URL: &str = "https://platform.claude.com/docs/en/api/admin/api_keys/retrieve";

fn built_in_definitions() -> Vec<DetectorDefinition> {
    vec![
        DetectorDefinition {
            detector_class: "ssn",
            display_name: "US Social Security Number",
            provider: None,
            category: "pii",
            pattern: r"\b\d{3}-\d{2}-\d{4}\b",
            source: "builtin",
            source_url: None,
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "email",
            display_name: "Email Address",
            provider: None,
            category: "pii",
            pattern: r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            source: "builtin",
            source_url: None,
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "aws_access_key",
            display_name: "AWS Access Key",
            provider: Some("aws"),
            category: "secret",
            pattern: r"\bAKIA[0-9A-Z]{16}\b",
            source: "builtin",
            source_url: Some(GITHUB_SUPPORTED_PATTERNS_URL),
            verifier_kind: Some(VerifierKind::AwsExampleGuard),
            remote_verifier_kind: None,
            default_verification: VerificationMode::Local,
        },
        DetectorDefinition {
            detector_class: "github_pat_classic",
            display_name: "GitHub Classic Personal Access Token",
            provider: Some("github"),
            category: "secret",
            pattern: r"\bghp_[A-Za-z0-9]{20,}\b",
            source: "github_docs",
            source_url: Some(GITHUB_AUTH_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::GitHubUserApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "github_pat_fine_grained",
            display_name: "GitHub Fine-Grained Personal Access Token",
            provider: Some("github"),
            category: "secret",
            pattern: r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
            source: "github_docs",
            source_url: Some(GITHUB_AUTH_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::GitHubUserApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "github_oauth_token",
            display_name: "GitHub OAuth Access Token",
            provider: Some("github"),
            category: "secret",
            pattern: r"\bgho_[A-Za-z0-9]{20,}\b",
            source: "github_docs",
            source_url: Some(GITHUB_AUTH_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::GitHubUserApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "github_app_user_token",
            display_name: "GitHub App User Access Token",
            provider: Some("github"),
            category: "secret",
            pattern: r"\bghu_[A-Za-z0-9]{20,}\b",
            source: "github_docs",
            source_url: Some(GITHUB_AUTH_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::GitHubUserApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "github_app_installation_token",
            display_name: "GitHub App Installation Token",
            provider: Some("github"),
            category: "secret",
            pattern: r"\bghs_[A-Za-z0-9]{20,}\b",
            source: "github_docs",
            source_url: Some(GITHUB_AUTH_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::GitHubUserApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "slack_bot_token",
            display_name: "Slack Bot Token",
            provider: Some("slack"),
            category: "secret",
            pattern: r"\bxoxb-[A-Za-z0-9-]{10,}\b",
            source: "slack_docs",
            source_url: Some(SLACK_WEB_API_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::SlackAuthTestApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "slack_user_token",
            display_name: "Slack User Token",
            provider: Some("slack"),
            category: "secret",
            pattern: r"\bxoxp-[A-Za-z0-9-]{10,}\b",
            source: "slack_docs",
            source_url: Some(SLACK_WEB_API_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::SlackAuthTestApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "slack_incoming_webhook_url",
            display_name: "Slack Incoming Webhook URL",
            provider: Some("slack"),
            category: "secret",
            pattern: r"https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+",
            source: "slack_docs",
            source_url: Some(SLACK_WEBHOOK_URL),
            verifier_kind: Some(VerifierKind::SlackExampleGuard),
            remote_verifier_kind: None,
            default_verification: VerificationMode::Local,
        },
        DetectorDefinition {
            detector_class: "stripe_live_secret_key",
            display_name: "Stripe Live Secret Key",
            provider: Some("stripe"),
            category: "secret",
            pattern: r"\bsk_live_[A-Za-z0-9]{16,}\b",
            source: "stripe_docs",
            source_url: Some(STRIPE_KEYS_URL),
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "stripe_test_secret_key",
            display_name: "Stripe Test Secret Key",
            provider: Some("stripe"),
            category: "secret",
            pattern: r"\bsk_test_[A-Za-z0-9]{16,}\b",
            source: "stripe_docs",
            source_url: Some(STRIPE_KEYS_URL),
            verifier_kind: Some(VerifierKind::StripeExampleGuard),
            remote_verifier_kind: None,
            default_verification: VerificationMode::Local,
        },
        DetectorDefinition {
            detector_class: "stripe_live_restricted_key",
            display_name: "Stripe Live Restricted Key",
            provider: Some("stripe"),
            category: "secret",
            pattern: r"\brk_live_[A-Za-z0-9]{16,}\b",
            source: "stripe_docs",
            source_url: Some(STRIPE_KEYS_URL),
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "stripe_test_restricted_key",
            display_name: "Stripe Test Restricted Key",
            provider: Some("stripe"),
            category: "secret",
            pattern: r"\brk_test_[A-Za-z0-9]{16,}\b",
            source: "stripe_docs",
            source_url: Some(STRIPE_KEYS_URL),
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "anthropic_admin_api_key",
            display_name: "Anthropic Admin API Key",
            provider: Some("anthropic"),
            category: "secret",
            pattern: r"\bsk-ant-admin[A-Za-z0-9_-]{8,}\b",
            source: "anthropic_docs",
            source_url: Some(CLAUDE_ADMIN_API_URL),
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "anthropic_api_key",
            display_name: "Anthropic API Key",
            provider: Some("anthropic"),
            category: "secret",
            pattern: r"\bsk-ant-api\d{2}-[A-Za-z0-9_-]{8,}\b",
            source: "anthropic_docs",
            source_url: Some(CLAUDE_API_KEY_URL),
            verifier_kind: None,
            remote_verifier_kind: Some(RemoteVerifierKind::AnthropicModelsApi),
            default_verification: VerificationMode::Disabled,
        },
        DetectorDefinition {
            detector_class: "api_secret",
            display_name: "API Secret Token",
            provider: None,
            category: "secret",
            pattern: r"\bsk-[A-Za-z0-9_-]{12,}\b",
            source: "builtin",
            source_url: Some(GITHUB_SUPPORTED_PATTERNS_URL),
            verifier_kind: None,
            remote_verifier_kind: None,
            default_verification: VerificationMode::Disabled,
        },
    ]
}

fn built_in_definition(detector_class: &str) -> Option<DetectorDefinition> {
    built_in_definitions()
        .into_iter()
        .find(|definition| definition.detector_class == detector_class)
}

fn build_verification_client() -> VerificationClient {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let https_connector = hyper_rustls::HttpsConnectorBuilder::new()
        .with_tls_config(tls_config)
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .build();

    Client::builder(TokioExecutor::new()).build(https_connector)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteVerificationResult {
    Verified,
    Rejected,
    Indeterminate,
}

fn empty_verifier_body() -> http_body_util::combinators::BoxBody<Bytes, hyper::Error> {
    Full::new(Bytes::new())
        .map_err(|never| match never {})
        .boxed()
}

async fn run_remote_verifier(
    client: &VerificationClient,
    config: &RemoteVerifierConfig,
    kind: RemoteVerifierKind,
    matched_text: &str,
) -> RemoteVerificationResult {
    match kind {
        RemoteVerifierKind::GitHubUserApi => {
            let request = match Request::builder()
                .method(Method::GET)
                .uri(&config.github_user_url)
                .header(AUTHORIZATION, format!("Bearer {matched_text}"))
                .header(USER_AGENT, "tiny-reverse-proxy")
                .header(ACCEPT, "application/vnd.github+json")
                .body(empty_verifier_body())
            {
                Ok(request) => request,
                Err(_) => return RemoteVerificationResult::Indeterminate,
            };

            match tokio::time::timeout(config.timeout, client.request(request)).await {
                Ok(Ok(response)) => match response.status() {
                    StatusCode::OK => RemoteVerificationResult::Verified,
                    StatusCode::UNAUTHORIZED => RemoteVerificationResult::Rejected,
                    _ => RemoteVerificationResult::Indeterminate,
                },
                _ => RemoteVerificationResult::Indeterminate,
            }
        }
        RemoteVerifierKind::SlackAuthTestApi => {
            let request = match Request::builder()
                .method(Method::POST)
                .uri(&config.slack_auth_test_url)
                .header(AUTHORIZATION, format!("Bearer {matched_text}"))
                .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(empty_verifier_body())
            {
                Ok(request) => request,
                Err(_) => return RemoteVerificationResult::Indeterminate,
            };

            match tokio::time::timeout(config.timeout, client.request(request)).await {
                Ok(Ok(response)) => {
                    let status = response.status();
                    let body = match response.into_body().collect().await {
                        Ok(body) => body.to_bytes(),
                        Err(_) => return RemoteVerificationResult::Indeterminate,
                    };
                    if !status.is_success() {
                        return RemoteVerificationResult::Indeterminate;
                    }
                    let value: serde_json::Value = match serde_json::from_slice(&body) {
                        Ok(value) => value,
                        Err(_) => return RemoteVerificationResult::Indeterminate,
                    };
                    if value.get("ok").and_then(|value| value.as_bool()) == Some(true) {
                        RemoteVerificationResult::Verified
                    } else if matches!(
                        value.get("error").and_then(|value| value.as_str()),
                        Some("invalid_auth" | "token_revoked")
                    ) {
                        RemoteVerificationResult::Rejected
                    } else {
                        RemoteVerificationResult::Indeterminate
                    }
                }
                _ => RemoteVerificationResult::Indeterminate,
            }
        }
        RemoteVerifierKind::AnthropicModelsApi => {
            let request = match Request::builder()
                .method(Method::GET)
                .uri(&config.anthropic_models_url)
                .header("x-api-key", matched_text)
                .header("anthropic-version", "2023-06-01")
                .header(ACCEPT, "application/json")
                .body(empty_verifier_body())
            {
                Ok(request) => request,
                Err(_) => return RemoteVerificationResult::Indeterminate,
            };

            match tokio::time::timeout(config.timeout, client.request(request)).await {
                Ok(Ok(response)) => match response.status() {
                    StatusCode::OK => RemoteVerificationResult::Verified,
                    StatusCode::UNAUTHORIZED => RemoteVerificationResult::Rejected,
                    _ => RemoteVerificationResult::Indeterminate,
                },
                _ => RemoteVerificationResult::Indeterminate,
            }
        }
    }
}

pub fn effective_detector_catalog(
    governance: Option<&GovernanceState>,
    project_id: Option<&str>,
) -> Result<Vec<EffectiveDetectorConfig>, Box<dyn std::error::Error>> {
    let project_default_action = project_id
        .and_then(|project_id| governance.map(|state| state.safety_mode_for_project(project_id)))
        .map(EnforcementAction::from_mode)
        .unwrap_or(EnforcementAction::Redact);

    let mut rules = built_in_rules()?;
    if let (Some(governance), Some(project_id)) = (governance, project_id) {
        let project_rules = governance
            .safety_rules_for_project(project_id)
            .into_iter()
            .map(|rule| FilterRule {
                pattern: rule.pattern,
                description: rule.description,
                detector_class: rule.detector_class,
                action: rule.action,
                verification: rule.verification,
                replacement: rule.replacement,
                path_patterns: rule.path_patterns,
                allowlist_patterns: rule.allowlist_patterns,
            })
            .collect::<Vec<_>>();
        rules = apply_rule_configs(rules, &project_rules)?;
    }

    Ok(rules
        .into_iter()
        .map(|rule| EffectiveDetectorConfig {
            detector_class: rule.detector_class,
            display_name: rule.display_name,
            provider: rule.provider,
            category: rule.category,
            source: rule.source,
            source_url: rule.source_url,
            effective_action: rule
                .action
                .unwrap_or(project_default_action)
                .as_str()
                .to_string(),
            action_source: rule.action_source.as_str().to_string(),
            verification_mode: rule.verification.as_str().to_string(),
            verifier_kind: rule
                .verifier_kind
                .map(|verifier_kind| verifier_kind.as_str().to_string()),
            remote_verifier_kind: rule
                .remote_verifier_kind
                .map(|verifier_kind| verifier_kind.as_str().to_string()),
            replacement: rule.replacement,
            path_patterns: rule.path_patterns_raw,
            allowlist_patterns: rule.allowlist_patterns_raw,
        })
        .collect())
}

pub struct ContentFilter {
    action: ContentFilterAction,
    input_rules: Vec<FilterRule>,
    output_rules: Vec<FilterRule>,
    governance: Option<Arc<GovernanceState>>,
    verification_client: VerificationClient,
    remote_verifier_config: RemoteVerifierConfig,
}

impl ContentFilter {
    pub fn new(
        action: ContentFilterAction,
        input_rules: Vec<FilterRule>,
        output_rules: Vec<FilterRule>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = apply_rule_configs(built_in_rules()?, &input_rules)?;
        let _ = compile_rules(&output_rules)?;
        Ok(Self {
            action,
            input_rules,
            output_rules,
            governance: None,
            verification_client: build_verification_client(),
            remote_verifier_config: RemoteVerifierConfig::default(),
        })
    }

    pub fn with_governance(mut self, governance: Arc<GovernanceState>) -> Self {
        self.governance = Some(governance);
        self
    }

    pub fn with_remote_verifier_config(mut self, config: RemoteVerifierConfig) -> Self {
        self.remote_verifier_config = config;
        self
    }

    fn default_action(&self, ctx: &RequestContext) -> EnforcementAction {
        let project_id = ctx
            .auth
            .as_ref()
            .and_then(|auth| auth.resolved_project())
            .map(|project| project.0.as_str());
        if let (Some(governance), Some(project_id)) = (&self.governance, project_id) {
            EnforcementAction::from_mode(governance.safety_mode_for_project(project_id))
        } else {
            match self.action {
                ContentFilterAction::Block => EnforcementAction::Block,
                ContentFilterAction::Log => EnforcementAction::ObserveOnly,
                ContentFilterAction::RedactAndForward => EnforcementAction::Redact,
            }
        }
    }

    fn effective_rules(
        &self,
        ctx: &RequestContext,
    ) -> Result<Vec<CompiledRule>, Box<dyn std::error::Error>> {
        let mut rules = built_in_rules()?;
        rules = apply_rule_configs(rules, &self.input_rules)?;

        let project_id = ctx
            .auth
            .as_ref()
            .and_then(|auth| auth.resolved_project())
            .map(|project| project.0.as_str());
        if let (Some(governance), Some(project_id)) = (&self.governance, project_id) {
            let project_rules = governance
                .safety_rules_for_project(project_id)
                .into_iter()
                .map(|rule| FilterRule {
                    pattern: rule.pattern,
                    description: rule.description,
                    detector_class: rule.detector_class,
                    action: rule.action,
                    verification: rule.verification,
                    replacement: rule.replacement,
                    path_patterns: rule.path_patterns,
                    allowlist_patterns: rule.allowlist_patterns,
                })
                .collect::<Vec<_>>();
            rules = apply_rule_configs(rules, &project_rules)?;
        }

        Ok(rules)
    }
}

impl Clone for ContentFilter {
    fn clone(&self) -> Self {
        Self {
            action: self.action,
            input_rules: self.input_rules.clone(),
            output_rules: self.output_rules.clone(),
            governance: self.governance.clone(),
            verification_client: self.verification_client.clone(),
            remote_verifier_config: self.remote_verifier_config.clone(),
        }
    }
}

#[async_trait]
impl Plugin for ContentFilter {
    fn name(&self) -> &str {
        "content_filter"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        ctx.extensions
            .insert(ContentFilterReplayHandle(self.clone()));

        let body = match &ctx.body {
            Some(body) => body,
            None => return Action::Continue,
        };

        let rules = match self.effective_rules(ctx) {
            Ok(rules) => rules,
            Err(error) => {
                tracing::warn!(error = %error, "content_filter: failed to compile rules");
                return Action::Continue;
            }
        };
        if rules.is_empty() {
            return Action::Continue;
        }

        let default_action = self.default_action(ctx);
        let mut matches = Vec::new();
        let mut modified_body = None;
        let mut effective_action = EnforcementAction::Allow;

        if let Ok(mut json) = serde_json::from_slice::<Value>(body) {
            let mut changed = false;
            scrub_json_value(
                &mut json,
                "$",
                false,
                &rules,
                default_action,
                &self.verification_client,
                &self.remote_verifier_config,
                &mut matches,
                &mut changed,
                &mut effective_action,
            )
            .await;
            if changed {
                match serde_json::to_vec(&json) {
                    Ok(bytes) => modified_body = Some(Bytes::from(bytes)),
                    Err(error) => {
                        tracing::warn!(error = %error, "content_filter: failed to serialize redacted body");
                    }
                }
            }
        } else if let Ok(text) = std::str::from_utf8(body) {
            let (redacted, body_matches, body_action) = scrub_text(
                text,
                "$",
                &rules,
                default_action,
                &self.verification_client,
                &self.remote_verifier_config,
            )
            .await;
            matches.extend(body_matches);
            effective_action = effective_action.max(body_action);
            if redacted != text {
                modified_body = Some(Bytes::from(redacted));
            }
        }

        if matches.is_empty() {
            return Action::Continue;
        }

        ctx.extensions.insert(SafetyAudit {
            mode: effective_action.as_str().to_string(),
            matches: matches.clone(),
        });

        match effective_action {
            EnforcementAction::Block => {
                let description = matches
                    .iter()
                    .find(|entry| entry.action == EnforcementAction::Block.as_str())
                    .unwrap_or(&matches[0])
                    .description
                    .clone();
                tracing::warn!(rule = %description, "content_filter: blocked request");
                let body = format!(
                    r#"{{"error":{{"message":"Content policy violation: {}","type":"content_filter_error","code":"content_blocked"}}}}"#,
                    description,
                );
                let mut resp = Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(
                        Full::new(Bytes::from(body))
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap();
                resp.headers_mut()
                    .insert("content-type", HeaderValue::from_static("application/json"));
                Action::Respond(resp)
            }
            EnforcementAction::ObserveOnly => {
                tracing::warn!(
                    matches = matches.len(),
                    "content_filter: observed sensitive input"
                );
                Action::Continue
            }
            EnforcementAction::Redact => {
                if let Some(body) = modified_body {
                    let len = body.len().to_string();
                    ctx.body = Some(body.clone());
                    if let Some(forwarded) = ctx.extensions.get_mut::<ForwardedRequestBody>() {
                        *forwarded = ForwardedRequestBody(body.clone());
                    }
                    if let Ok(value) = HeaderValue::from_str(&len) {
                        ctx.headers.insert("content-length", value);
                    }
                }
                tracing::info!(
                    matches = matches.len(),
                    "content_filter: redacted sensitive input"
                );
                Action::Continue
            }
            EnforcementAction::Allow => Action::Continue,
        }
    }

    async fn on_response(&self, _ctx: &mut RequestContext, resp: &mut ResponseContext) -> Action {
        let _ = &self.output_rules;
        let _ = resp;
        Action::Continue
    }
}

pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    let filter = create_filter(config)?;
    Ok(Box::new(filter))
}

pub fn create_filter(config: &toml::Value) -> Result<ContentFilter, Box<dyn std::error::Error>> {
    let action = match config.get("action").and_then(|v| v.as_str()) {
        Some("block") => ContentFilterAction::Block,
        Some("log") | Some("observe_only") => ContentFilterAction::Log,
        Some("redact_and_forward") | None => ContentFilterAction::RedactAndForward,
        Some(other) => return Err(format!("unknown content_filter action: {}", other).into()),
    };

    let input_rules = parse_rules(config, "input_rules")?;
    let output_rules = parse_rules(config, "output_rules")?;
    let mut verifier_config = RemoteVerifierConfig::default();
    if let Some(timeout_ms) = config
        .get("verification_timeout_ms")
        .and_then(|value| value.as_integer())
    {
        verifier_config.timeout = Duration::from_millis(timeout_ms.max(1) as u64);
    }
    Ok(ContentFilter::new(action, input_rules, output_rules)?
        .with_remote_verifier_config(verifier_config))
}

fn compile_rules(rules: &[FilterRule]) -> Result<Vec<CompiledRule>, Box<dyn std::error::Error>> {
    let mut compiled = Vec::with_capacity(rules.len());
    for rule in rules {
        let pattern = rule
            .pattern
            .as_deref()
            .ok_or_else(|| "filter rule pattern is required".to_string())?;
        let detector_class = rule
            .detector_class
            .clone()
            .unwrap_or_else(|| "custom".to_string());
        let definition = built_in_definition(&detector_class);
        compiled.push(CompiledRule {
            regex: Regex::new(pattern)?,
            description: rule
                .description
                .clone()
                .or_else(|| definition.map(|definition| definition.display_name.to_string()))
                .unwrap_or_else(|| "custom rule".to_string()),
            detector_class,
            display_name: definition
                .map(|definition| definition.display_name.to_string())
                .unwrap_or_else(|| "Custom Detector".to_string()),
            provider: definition
                .and_then(|definition| definition.provider.map(ToString::to_string)),
            category: definition
                .map(|definition| definition.category.to_string())
                .unwrap_or_else(|| "custom".to_string()),
            source: definition
                .map(|definition| definition.source.to_string())
                .unwrap_or_else(|| "custom".to_string()),
            source_url: definition
                .and_then(|definition| definition.source_url.map(ToString::to_string)),
            action: rule.action.as_deref().map(parse_rule_action).transpose()?,
            verification: rule
                .verification
                .as_deref()
                .map(parse_verification_mode)
                .transpose()?
                .unwrap_or_else(|| {
                    definition
                        .map(|definition| definition.default_verification)
                        .unwrap_or(VerificationMode::Disabled)
                }),
            verifier_kind: definition.and_then(|definition| definition.verifier_kind),
            remote_verifier_kind: definition.and_then(|definition| definition.remote_verifier_kind),
            replacement: rule.replacement.clone(),
            action_source: if rule.action.is_some() {
                ActionSource::RuleOverride
            } else {
                ActionSource::ProjectDefault
            },
            path_patterns: compile_regex_list(&rule.path_patterns)?,
            path_patterns_raw: rule.path_patterns.clone(),
            allowlist_patterns: compile_regex_list(&rule.allowlist_patterns)?,
            allowlist_patterns_raw: rule.allowlist_patterns.clone(),
        });
    }
    Ok(compiled)
}

fn built_in_rules() -> Result<Vec<CompiledRule>, Box<dyn std::error::Error>> {
    compile_rules(
        &built_in_definitions()
            .into_iter()
            .map(|definition| FilterRule {
                pattern: Some(definition.pattern.to_string()),
                description: Some(definition.display_name.to_string()),
                detector_class: Some(definition.detector_class.to_string()),
                action: None,
                verification: Some(definition.default_verification.as_str().to_string()),
                replacement: None,
                path_patterns: Vec::new(),
                allowlist_patterns: Vec::new(),
            })
            .collect::<Vec<_>>(),
    )
}

fn apply_rule_configs(
    mut compiled: Vec<CompiledRule>,
    rules: &[FilterRule],
) -> Result<Vec<CompiledRule>, Box<dyn std::error::Error>> {
    for rule in rules {
        let existing = rule.detector_class.as_ref().and_then(|detector_class| {
            compiled
                .iter()
                .position(|existing| existing.detector_class == *detector_class)
        });

        if let Some(index) = existing {
            compiled[index] = merge_rule(&compiled[index], rule)?;
        } else {
            compiled.extend(compile_rules(std::slice::from_ref(rule))?);
        }
    }
    Ok(compiled)
}

fn merge_rule(
    existing: &CompiledRule,
    rule: &FilterRule,
) -> Result<CompiledRule, Box<dyn std::error::Error>> {
    let mut merged = existing.clone();
    if let Some(pattern) = rule.pattern.as_deref() {
        merged.regex = Regex::new(pattern)?;
    }
    if let Some(description) = rule.description.as_ref() {
        merged.description = description.clone();
    }
    if let Some(detector_class) = rule.detector_class.as_ref() {
        merged.detector_class = detector_class.clone();
    }
    if let Some(action) = rule.action.as_deref() {
        merged.action = Some(parse_rule_action(action)?);
        merged.action_source = ActionSource::RuleOverride;
    }
    if let Some(verification) = rule.verification.as_deref() {
        merged.verification = parse_verification_mode(verification)?;
    }
    if let Some(replacement) = rule.replacement.as_ref() {
        merged.replacement = Some(replacement.clone());
    }
    if !rule.path_patterns.is_empty() {
        merged.path_patterns = compile_regex_list(&rule.path_patterns)?;
        merged.path_patterns_raw = rule.path_patterns.clone();
    }
    if !rule.allowlist_patterns.is_empty() {
        merged.allowlist_patterns = compile_regex_list(&rule.allowlist_patterns)?;
        merged.allowlist_patterns_raw = rule.allowlist_patterns.clone();
    }
    Ok(merged)
}

fn compile_regex_list(patterns: &[String]) -> Result<Vec<Regex>, Box<dyn std::error::Error>> {
    let mut compiled = Vec::with_capacity(patterns.len());
    for pattern in patterns {
        compiled.push(Regex::new(pattern)?);
    }
    Ok(compiled)
}

fn parse_rule_action(value: &str) -> Result<EnforcementAction, Box<dyn std::error::Error>> {
    EnforcementAction::parse(value)
        .ok_or_else(|| format!("unknown content filter rule action: {value}").into())
}

fn parse_verification_mode(value: &str) -> Result<VerificationMode, Box<dyn std::error::Error>> {
    VerificationMode::parse(value)
        .ok_or_else(|| format!("unknown content filter verification mode: {value}").into())
}

#[async_recursion]
async fn scrub_json_value(
    value: &mut Value,
    path: &str,
    active_text_field: bool,
    rules: &[CompiledRule],
    default_action: EnforcementAction,
    verification_client: &VerificationClient,
    remote_verifier_config: &RemoteVerifierConfig,
    matches: &mut Vec<SafetyMatch>,
    changed: &mut bool,
    effective_action: &mut EnforcementAction,
) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                let child_path = format!("{path}.{key}");
                scrub_json_value(
                    child,
                    &child_path,
                    active_text_field || is_sensitive_text_field(key),
                    rules,
                    default_action,
                    verification_client,
                    remote_verifier_config,
                    matches,
                    changed,
                    effective_action,
                )
                .await;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter_mut().enumerate() {
                let child_path = format!("{path}[{index}]");
                scrub_json_value(
                    child,
                    &child_path,
                    active_text_field,
                    rules,
                    default_action,
                    verification_client,
                    remote_verifier_config,
                    matches,
                    changed,
                    effective_action,
                )
                .await;
            }
        }
        Value::String(text) if active_text_field => {
            let original = text.clone();
            let (redacted, local_matches, local_action) = scrub_text(
                &original,
                path,
                rules,
                default_action,
                verification_client,
                remote_verifier_config,
            )
            .await;
            matches.extend(local_matches);
            *effective_action = (*effective_action).max(local_action);
            if local_action == EnforcementAction::Redact && redacted != original {
                *text = redacted;
                *changed = true;
            }
        }
        _ => {}
    }
}

async fn scrub_text(
    input: &str,
    path: &str,
    rules: &[CompiledRule],
    default_action: EnforcementAction,
    verification_client: &VerificationClient,
    remote_verifier_config: &RemoteVerifierConfig,
) -> (String, Vec<SafetyMatch>, EnforcementAction) {
    let mut output = input.to_string();
    let mut matches = Vec::new();
    let mut effective_action = EnforcementAction::Allow;

    for rule in rules {
        if !rule.path_patterns.is_empty()
            && !rule
                .path_patterns
                .iter()
                .any(|pattern| pattern.is_match(path))
        {
            continue;
        }

        let action = rule.action.unwrap_or(default_action);
        if action == EnforcementAction::Allow {
            continue;
        }

        let mut matched = false;
        if action == EnforcementAction::Redact {
            output = rule
                .regex
                .replace_all(&output, |captures: &regex::Captures<'_>| {
                    let matched_text = captures.get(0).unwrap().as_str();
                    if rule
                        .allowlist_patterns
                        .iter()
                        .any(|pattern| pattern.is_match(matched_text))
                    {
                        matched_text.to_string()
                    } else {
                        matched_text.to_string()
                    }
                })
                .into_owned();
            let candidate_values = rule
                .regex
                .find_iter(&output)
                .map(|m| m.as_str().to_string())
                .collect::<Vec<_>>();
            for matched_text in candidate_values {
                if rule
                    .allowlist_patterns
                    .iter()
                    .any(|pattern| pattern.is_match(&matched_text))
                    || !verifier_accepts(
                        rule,
                        &matched_text,
                        verification_client,
                        remote_verifier_config,
                    )
                    .await
                {
                    continue;
                }
                matched = true;
                output = output.replacen(
                    &matched_text,
                    &rule
                        .replacement
                        .clone()
                        .unwrap_or_else(|| format!("[REDACTED:{}]", rule.detector_class)),
                    1,
                );
            }
        } else {
            for matched_text in rule.regex.find_iter(&output).map(|m| m.as_str()) {
                if rule
                    .allowlist_patterns
                    .iter()
                    .any(|pattern| pattern.is_match(matched_text))
                    || !verifier_accepts(
                        rule,
                        matched_text,
                        verification_client,
                        remote_verifier_config,
                    )
                    .await
                {
                    continue;
                }
                matched = true;
                break;
            }
        }

        if matched {
            effective_action = effective_action.max(action);
            matches.push(SafetyMatch {
                detector_class: rule.detector_class.clone(),
                description: rule.description.clone(),
                path: path.to_string(),
                action: action.as_str().to_string(),
            });
        }
    }

    (output, matches, effective_action)
}

fn local_verifier_accepts(rule: &CompiledRule, matched_text: &str) -> bool {
    match rule.verifier_kind {
        None => true,
        Some(VerifierKind::AwsExampleGuard) => {
            matched_text != "AKIAIOSFODNN7EXAMPLE" && !matched_text.ends_with("EXAMPLE")
        }
        Some(VerifierKind::SlackExampleGuard) => {
            matched_text
                != [
                    "https://hooks.slack.com/services/",
                    "T1234567",
                    "/",
                    "AAAAAAAA",
                    "/",
                    "ZZZZZZ",
                ]
                .concat()
        }
        Some(VerifierKind::StripeExampleGuard) => {
            matched_text != ["sk", "_test_", "BQokikJOvBiI2HlWgH4olfQ2"].concat()
        }
    }
}

async fn verifier_accepts(
    rule: &CompiledRule,
    matched_text: &str,
    verification_client: &VerificationClient,
    remote_verifier_config: &RemoteVerifierConfig,
) -> bool {
    match rule.verification {
        VerificationMode::Disabled => true,
        VerificationMode::Default | VerificationMode::Local => {
            local_verifier_accepts(rule, matched_text)
        }
        VerificationMode::Remote => {
            let remote_result = match rule.remote_verifier_kind {
                Some(kind) => {
                    run_remote_verifier(
                        verification_client,
                        remote_verifier_config,
                        kind,
                        matched_text,
                    )
                    .await
                }
                None => RemoteVerificationResult::Indeterminate,
            };
            match remote_result {
                RemoteVerificationResult::Verified => true,
                RemoteVerificationResult::Rejected => false,
                RemoteVerificationResult::Indeterminate => {
                    local_verifier_accepts(rule, matched_text)
                }
            }
        }
    }
}

fn is_sensitive_text_field(key: &str) -> bool {
    matches!(
        key,
        "content" | "text" | "prompt" | "input" | "instructions" | "arguments" | "query"
    )
}

fn parse_rules(
    config: &toml::Value,
    key: &str,
) -> Result<Vec<FilterRule>, Box<dyn std::error::Error>> {
    let arr = match config.get(key).and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };

    let mut rules = Vec::with_capacity(arr.len());
    for entry in arr {
        let pattern = entry
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let detector_class = entry
            .get("detector_class")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        if pattern.is_none() && detector_class.is_none() {
            return Err(format!("{key}[] requires either pattern or detector_class").into());
        }
        let description = entry
            .get("description")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let action = entry
            .get("action")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let verification = entry
            .get("verification")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let replacement = entry
            .get("replacement")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let path_patterns =
            parse_toml_string_array(entry.get("path_patterns").or_else(|| entry.get("paths")))
                .ok_or_else(|| format!("{key}[].path_patterns must be an array of strings"))?;
        let allowlist_patterns = parse_toml_string_array(
            entry
                .get("allowlist_patterns")
                .or_else(|| entry.get("allowlist")),
        )
        .ok_or_else(|| format!("{key}[].allowlist_patterns must be an array of strings"))?;
        rules.push(FilterRule {
            pattern,
            description,
            detector_class,
            action,
            verification,
            replacement,
            path_patterns,
            allowlist_patterns,
        });
    }

    Ok(rules)
}

fn parse_toml_string_array(value: Option<&toml::Value>) -> Option<Vec<String>> {
    let Some(value) = value else {
        return Some(Vec::new());
    };
    let array = value.as_array()?;
    Some(
        array
            .iter()
            .map(|entry| entry.as_str().map(ToString::to_string))
            .collect::<Option<Vec<_>>>()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governance::GovernanceState;
    use crate::store::SafetyPolicyRecord;
    use http_body_util::Full;
    use hyper::body::Incoming;
    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, Uri, Version};
    use hyper_util::rt::TokioIo;
    use proxy_auth::AuthContext;
    use std::convert::Infallible;
    use std::sync::Arc;

    fn github_pat_example() -> String {
        ["ghp", "_abcdefghijklmnopqrstuvwxyz123456"].concat()
    }

    fn slack_incoming_webhook_example() -> String {
        [
            "https://hooks.slack.com/services/",
            "TABCDE123",
            "/",
            "BABCDE123",
            "/",
            "secretsecret",
        ]
        .concat()
    }

    fn slack_doc_webhook_example() -> String {
        [
            "https://hooks.slack.com/services/",
            "T1234567",
            "/",
            "AAAAAAAA",
            "/",
            "ZZZZZZ",
        ]
        .concat()
    }

    fn stripe_live_secret_example() -> String {
        ["sk", "_live_", "1234567890abcdefghijklmnop"].concat()
    }

    fn stripe_doc_secret_example() -> String {
        ["sk", "_test_", "BQokikJOvBiI2HlWgH4olfQ2"].concat()
    }

    fn anthropic_admin_example() -> String {
        ["sk", "-ant-admin", "abcdefghijklmnop"].concat()
    }

    fn slack_bot_token_example() -> String {
        ["xoxb", "-1234567890-", "abcdefghijklmnopqrstuvwxyz"].concat()
    }

    fn make_ctx(body: Option<&[u8]>) -> RequestContext {
        RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: body.map(|b| Bytes::from(b.to_vec())),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    }

    async fn start_test_server<F>(handler: F) -> String
    where
        F: Fn(Request<Incoming>) -> Response<Full<Bytes>> + Send + Sync + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => continue,
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |req: Request<Incoming>| {
                        let handler = handler.clone();
                        async move { Ok::<_, Infallible>((handler)(req)) }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        format!("http://{}", addr)
    }

    #[tokio::test]
    async fn blocks_ssn_in_request_body() {
        let filter = ContentFilter::new(ContentFilterAction::Block, vec![], vec![]).unwrap();

        let body = br#"{"messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}"#;
        let mut ctx = make_ctx(Some(body));
        match filter.on_request(&mut ctx).await {
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                let body_str = std::str::from_utf8(&body).unwrap();
                assert!(body_str.contains("Social Security Number"));
                assert!(body_str.contains("content_blocked"));
            }
            Action::Continue => panic!("should block SSN"),
        }
    }

    #[tokio::test]
    async fn logs_but_continues_when_action_is_log() {
        let filter = ContentFilter::new(ContentFilterAction::Log, vec![], vec![]).unwrap();

        let body = br#"{"messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}"#;
        let mut ctx = make_ctx(Some(body));
        match filter.on_request(&mut ctx).await {
            Action::Continue => {
                let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
                assert_eq!(audit.mode, "observe_only");
                assert_eq!(audit.matches.len(), 1);
            }
            Action::Respond(_) => panic!("should not block in log mode"),
        }
    }

    #[tokio::test]
    async fn redacts_supported_text_fields() {
        let filter =
            ContentFilter::new(ContentFilterAction::RedactAndForward, vec![], vec![]).unwrap();

        let body = br#"{"messages":[{"role":"user","content":"email me at alice@example.com"}],"tools":[{"function":{"arguments":"{\"token\":\"sk-secret-secret-secret\"}"}}]}"#;
        let mut ctx = make_ctx(Some(body));
        assert!(matches!(
            filter.on_request(&mut ctx).await,
            Action::Continue
        ));

        let updated_body = ctx.body.unwrap();
        let updated_text = std::str::from_utf8(&updated_body).unwrap();
        assert!(!updated_text.contains("alice@example.com"));
        assert!(!updated_text.contains("sk-secret-secret-secret"));
        assert!(updated_text.contains("[REDACTED:email]"));
        assert!(updated_text.contains("[REDACTED:api_secret]"));

        let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
        assert_eq!(audit.mode, "redact_and_forward");
        assert!(audit
            .matches
            .iter()
            .all(|m| !m.description.contains("alice@example.com")));
    }

    #[tokio::test]
    async fn project_rule_can_override_builtin_detector_action() {
        let governance = Arc::new(GovernanceState::new(None));
        governance
            .upsert_safety_policy(SafetyPolicyRecord {
                project_id: "project-a".to_string(),
                mode: "observe_only".to_string(),
                rules_json: Some(
                    r#"[{"detector_class":"api_secret","action":"block"}]"#.to_string(),
                ),
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let filter = ContentFilter::new(ContentFilterAction::RedactAndForward, vec![], vec![])
            .unwrap()
            .with_governance(governance);

        let body = br#"{"messages":[{"role":"user","content":"token sk-secret-secret-secret"}]}"#;
        let mut ctx = make_ctx(Some(body));
        ctx.auth = Some(AuthContext::runtime("project-a", "runtime-key"));

        match filter.on_request(&mut ctx).await {
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
                let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
                assert_eq!(audit.mode, "block");
                assert_eq!(audit.matches[0].action, "block");
            }
            Action::Continue => panic!("should block project-level api_secret override"),
        }
    }

    #[tokio::test]
    async fn project_rule_can_scope_detector_and_allowlist_values() {
        let governance = Arc::new(GovernanceState::new(None));
        governance
            .upsert_safety_policy(SafetyPolicyRecord {
                project_id: "project-a".to_string(),
                mode: "observe_only".to_string(),
                rules_json: Some(
                    r#"[
                        {
                            "detector_class":"email",
                            "action":"redact",
                            "path_patterns":["^\\$\\.messages\\[[0-9]+\\]\\.content$"],
                            "allowlist_patterns":["(?i)^support@example\\.com$"]
                        }
                    ]"#
                    .to_string(),
                ),
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let filter = ContentFilter::new(ContentFilterAction::RedactAndForward, vec![], vec![])
            .unwrap()
            .with_governance(governance);

        let body = br#"{
            "messages":[
                {"role":"user","content":"contact support@example.com"},
                {"role":"user","content":"contact alice@example.com"}
            ],
            "tools":[
                {"function":{"arguments":"{\"email\":\"tooling@example.com\"}"}}
            ]
        }"#;
        let mut ctx = make_ctx(Some(body));
        ctx.auth = Some(AuthContext::runtime("project-a", "runtime-key"));

        assert!(matches!(
            filter.on_request(&mut ctx).await,
            Action::Continue
        ));

        let updated_body = ctx.body.unwrap();
        let updated_text = std::str::from_utf8(&updated_body).unwrap();
        assert!(updated_text.contains("support@example.com"));
        assert!(!updated_text.contains("alice@example.com"));
        assert!(updated_text.contains("tooling@example.com"));
        assert!(updated_text.contains("[REDACTED:email]"));

        let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
        assert_eq!(audit.mode, "redact_and_forward");
        assert_eq!(audit.matches.len(), 1);
        assert_eq!(audit.matches[0].path, "$.messages[1].content");
        assert_eq!(audit.matches[0].action, "redact_and_forward");
    }

    #[tokio::test]
    async fn custom_rule_can_use_custom_replacement() {
        let filter = ContentFilter::new(
            ContentFilterAction::Log,
            vec![FilterRule {
                pattern: Some(r"\bghp_[A-Za-z0-9]{12,}\b".to_string()),
                description: Some("GitHub personal access token".to_string()),
                detector_class: Some("github_pat".to_string()),
                action: Some("redact".to_string()),
                verification: None,
                replacement: Some("[MASKED:GITHUB]".to_string()),
                path_patterns: vec![r"^\$\.messages\[[0-9]+\]\.content$".to_string()],
                allowlist_patterns: Vec::new(),
            }],
            vec![],
        )
        .unwrap();

        let body = br#"{"messages":[{"role":"user","content":"token ghp_abcdefghijklmnop"}]}"#;
        let mut ctx = make_ctx(Some(body));
        assert!(matches!(
            filter.on_request(&mut ctx).await,
            Action::Continue
        ));

        let updated_body = ctx.body.unwrap();
        let updated_text = std::str::from_utf8(&updated_body).unwrap();
        assert!(!updated_text.contains("ghp_abcdefghijklmnop"));
        assert!(updated_text.contains("[MASKED:GITHUB]"));

        let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
        assert_eq!(audit.mode, "redact_and_forward");
        assert_eq!(audit.matches[0].detector_class, "github_pat");
    }

    #[tokio::test]
    async fn aws_example_key_is_ignored_by_builtin_verifier() {
        let filter = ContentFilter::new(ContentFilterAction::Block, vec![], vec![]).unwrap();

        let body =
            br#"{"messages":[{"role":"user","content":"example key AKIAIOSFODNN7EXAMPLE"}]}"#;
        let mut ctx = make_ctx(Some(body));
        match filter.on_request(&mut ctx).await {
            Action::Continue => {
                assert!(ctx.extensions.get::<SafetyAudit>().is_none());
            }
            Action::Respond(_) => panic!("aws doc example should be ignored by verifier"),
        }
    }

    #[tokio::test]
    async fn provider_specific_secret_formats_are_redacted() {
        let filter =
            ContentFilter::new(ContentFilterAction::RedactAndForward, vec![], vec![]).unwrap();

        let github_pat = github_pat_example();
        let slack_webhook = slack_incoming_webhook_example();
        let stripe_live = stripe_live_secret_example();
        let anthropic_admin = anthropic_admin_example();
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"{github_pat} {slack_webhook} {stripe_live} {anthropic_admin}"}}]}}"#
        );
        let mut ctx = make_ctx(Some(body.as_bytes()));
        assert!(matches!(
            filter.on_request(&mut ctx).await,
            Action::Continue
        ));

        let updated_body = ctx.body.unwrap();
        let updated_text = std::str::from_utf8(&updated_body).unwrap();
        assert!(!updated_text.contains(&github_pat));
        assert!(!updated_text.contains(&slack_webhook));
        assert!(!updated_text.contains(&stripe_live));
        assert!(!updated_text.contains(&anthropic_admin));
        assert!(updated_text.contains("[REDACTED:github_pat_classic]"));
        assert!(updated_text.contains("[REDACTED:slack_incoming_webhook_url]"));
        assert!(updated_text.contains("[REDACTED:stripe_live_secret_key]"));
        assert!(updated_text.contains("[REDACTED:anthropic_admin_api_key]"));
    }

    #[tokio::test]
    async fn slack_and_stripe_doc_examples_are_ignored_by_verifier() {
        let filter = ContentFilter::new(ContentFilterAction::Block, vec![], vec![]).unwrap();

        let slack_doc_webhook = slack_doc_webhook_example();
        let stripe_doc_secret = stripe_doc_secret_example();
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"{slack_doc_webhook} {stripe_doc_secret}"}}]}}"#
        );
        let mut ctx = make_ctx(Some(body.as_bytes()));
        match filter.on_request(&mut ctx).await {
            Action::Continue => {
                assert!(ctx.extensions.get::<SafetyAudit>().is_none());
            }
            Action::Respond(_) => panic!("doc examples should be ignored by verifiers"),
        }
    }

    #[tokio::test]
    async fn remote_github_verifier_rejects_invalid_token() {
        let server = start_test_server(|req| {
            assert_eq!(req.uri().path(), "/github/user");
            Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Full::new(Bytes::new()))
                .unwrap()
        })
        .await;

        let filter = ContentFilter::new(
            ContentFilterAction::Block,
            vec![FilterRule {
                pattern: None,
                description: None,
                detector_class: Some("github_pat_classic".to_string()),
                action: Some("block".to_string()),
                verification: Some("remote".to_string()),
                replacement: None,
                path_patterns: Vec::new(),
                allowlist_patterns: Vec::new(),
            }],
            vec![],
        )
        .unwrap()
        .with_remote_verifier_config(RemoteVerifierConfig {
            github_user_url: format!("{server}/github/user"),
            slack_auth_test_url: "http://127.0.0.1/unused".to_string(),
            anthropic_models_url: "http://127.0.0.1/unused".to_string(),
            timeout: Duration::from_millis(250),
        });

        let github_pat = github_pat_example();
        let body = format!(r#"{{"messages":[{{"role":"user","content":"token {github_pat}"}}]}}"#);
        let mut ctx = make_ctx(Some(body.as_bytes()));
        match filter.on_request(&mut ctx).await {
            Action::Continue => {
                assert!(ctx.extensions.get::<SafetyAudit>().is_none());
            }
            Action::Respond(_) => panic!("remote github verifier should reject invalid token"),
        }
    }

    #[tokio::test]
    async fn remote_slack_verifier_confirms_valid_token() {
        let server = start_test_server(|req| {
            assert_eq!(req.uri().path(), "/slack/auth.test");
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                .unwrap()
        })
        .await;

        let filter = ContentFilter::new(
            ContentFilterAction::Log,
            vec![FilterRule {
                pattern: None,
                description: None,
                detector_class: Some("slack_bot_token".to_string()),
                action: Some("block".to_string()),
                verification: Some("remote".to_string()),
                replacement: None,
                path_patterns: Vec::new(),
                allowlist_patterns: Vec::new(),
            }],
            vec![],
        )
        .unwrap()
        .with_remote_verifier_config(RemoteVerifierConfig {
            github_user_url: "http://127.0.0.1/unused".to_string(),
            slack_auth_test_url: format!("{server}/slack/auth.test"),
            anthropic_models_url: "http://127.0.0.1/unused".to_string(),
            timeout: Duration::from_millis(250),
        });

        let slack_bot_token = slack_bot_token_example();
        let body =
            format!(r#"{{"messages":[{{"role":"user","content":"token {slack_bot_token}"}}]}}"#);
        let mut ctx = make_ctx(Some(body.as_bytes()));
        match filter.on_request(&mut ctx).await {
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
                let audit = ctx.extensions.get::<SafetyAudit>().unwrap();
                assert_eq!(audit.mode, "block");
            }
            Action::Continue => panic!("remote slack verifier should keep valid token match"),
        }
    }

    #[tokio::test]
    async fn remote_anthropic_verifier_rejects_invalid_key() {
        let server = start_test_server(|req| {
            assert_eq!(req.uri().path(), "/anthropic/models");
            Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Full::new(Bytes::new()))
                .unwrap()
        })
        .await;

        let filter = ContentFilter::new(
            ContentFilterAction::Log,
            vec![
                FilterRule {
                    pattern: None,
                    description: None,
                    detector_class: Some("anthropic_api_key".to_string()),
                    action: Some("block".to_string()),
                    verification: Some("remote".to_string()),
                    replacement: None,
                    path_patterns: Vec::new(),
                    allowlist_patterns: Vec::new(),
                },
                FilterRule {
                    pattern: None,
                    description: None,
                    detector_class: Some("api_secret".to_string()),
                    action: Some("allow".to_string()),
                    verification: None,
                    replacement: None,
                    path_patterns: Vec::new(),
                    allowlist_patterns: Vec::new(),
                },
            ],
            vec![],
        )
        .unwrap()
        .with_remote_verifier_config(RemoteVerifierConfig {
            github_user_url: "http://127.0.0.1/unused".to_string(),
            slack_auth_test_url: "http://127.0.0.1/unused".to_string(),
            anthropic_models_url: format!("{server}/anthropic/models"),
            timeout: Duration::from_millis(250),
        });

        let body =
            br#"{"messages":[{"role":"user","content":"token sk-ant-api03-abcdefghijklmnop"}]}"#;
        let mut ctx = make_ctx(Some(body));
        match filter.on_request(&mut ctx).await {
            Action::Continue => {
                assert!(ctx.extensions.get::<SafetyAudit>().is_none());
            }
            Action::Respond(_) => panic!("remote anthropic verifier should reject invalid key"),
        }
    }

    #[tokio::test]
    async fn detector_catalog_reports_effective_project_overrides() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_safety_policy(SafetyPolicyRecord {
                project_id: "project-a".to_string(),
                mode: "observe_only".to_string(),
                rules_json: Some(
                    r#"[
                        {
                            "detector_class":"aws_access_key",
                            "action":"block",
                            "verification":"disabled"
                        }
                    ]"#
                    .to_string(),
                ),
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let detectors = effective_detector_catalog(Some(&governance), Some("project-a")).unwrap();
        let aws = detectors
            .iter()
            .find(|detector| detector.detector_class == "aws_access_key")
            .unwrap();
        let email = detectors
            .iter()
            .find(|detector| detector.detector_class == "email")
            .unwrap();

        assert_eq!(aws.effective_action, "block");
        assert_eq!(aws.action_source, "rule_override");
        assert_eq!(aws.verification_mode, "disabled");
        assert_eq!(aws.verifier_kind.as_deref(), Some("aws_example_guard"));
        let github = detectors
            .iter()
            .find(|detector| detector.detector_class == "github_pat_classic")
            .unwrap();
        let slack = detectors
            .iter()
            .find(|detector| detector.detector_class == "slack_bot_token")
            .unwrap();
        assert!(detectors
            .iter()
            .any(|detector| detector.detector_class == "github_pat_classic"));
        assert!(detectors
            .iter()
            .any(|detector| detector.detector_class == "slack_bot_token"));
        assert_eq!(
            github.remote_verifier_kind.as_deref(),
            Some("github_user_api")
        );
        assert_eq!(
            slack.remote_verifier_kind.as_deref(),
            Some("slack_auth_test_api")
        );
        assert_eq!(email.effective_action, "observe_only");
        assert_eq!(email.action_source, "project_default");
    }
}
