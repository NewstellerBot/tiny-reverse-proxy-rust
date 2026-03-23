use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::HeaderValue;
use hyper::{Method, Response, StatusCode};
use proxy_core::plugin::{Action, Plugin, RequestContext};
use regex::Regex;
use serde_json::{json, Value};

use crate::governance::GovernanceState;
use crate::virtual_keys::VirtualKeyMeta;
use crate::{cached_request_json_mut, sync_cached_request_json_body};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptResolutionAudit {
    pub prompt_name: String,
    pub prompt_version: String,
    pub prompt_environment: String,
    pub prompt_rollout_id: Option<String>,
    pub prompt_rollout_mode: Option<String>,
}

#[derive(Clone)]
pub struct PromptRegistry {
    governance: Arc<GovernanceState>,
    default_environment: String,
}

#[derive(Debug, Clone)]
struct PromptReference {
    name: String,
    version: Option<String>,
    environment: String,
    variables: HashMap<String, String>,
}

enum PromptProtocol {
    OpenAiChat,
    AnthropicMessages,
}

const SESSION_ID_HEADER: &str = "x-trp-session-id";

impl PromptRegistry {
    pub fn new(
        config: &toml::Value,
        governance: Arc<GovernanceState>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let default_environment = config
            .get("default_environment")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("prod")
            .to_string();
        Ok(Self {
            governance,
            default_environment,
        })
    }
}

#[async_trait]
impl Plugin for PromptRegistry {
    fn name(&self) -> &str {
        "prompt_registry"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        if ctx.method != Method::POST {
            return Action::Continue;
        }

        if ctx.body.is_none() {
            return Action::Continue;
        }
        let project_id = match ctx
            .auth
            .as_ref()
            .and_then(|auth| auth.resolved_project())
            .map(|project| project.0.clone())
            .or_else(|| {
                ctx.extensions
                    .get::<VirtualKeyMeta>()
                    .map(|meta| meta.project_id.clone())
            }) {
            Some(project_id) => project_id,
            None => {
                return Action::Respond(json_error(
                    StatusCode::BAD_REQUEST,
                    "trp_prompt_ref requires a managed runtime key",
                ))
            }
        };

        let rollout_seed = stable_rollout_seed(ctx);
        let request_path = ctx.uri.path().to_string();
        let Some(request_json) = cached_request_json_mut(ctx) else {
            return Action::Continue;
        };
        let prompt_ref = match extract_prompt_reference(request_json, &self.default_environment) {
            Ok(prompt_ref) => prompt_ref,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };
        let Some(prompt_ref) = prompt_ref else {
            return Action::Continue;
        };
        let resolved = match self.governance.resolve_project_prompt(
            &project_id,
            &prompt_ref.name,
            prompt_ref.version.as_deref(),
            &prompt_ref.environment,
            rollout_seed.as_deref(),
        ) {
            Some(record) => record,
            None => {
                let message = if let Some(version) = prompt_ref.version.as_deref() {
                    format!(
                        "prompt reference not found for project '{}': {}/{}/{}",
                        project_id, prompt_ref.name, prompt_ref.environment, version
                    )
                } else {
                    format!(
                        "active prompt reference not found for project '{}': {}/{}",
                        project_id, prompt_ref.name, prompt_ref.environment
                    )
                };
                return Action::Respond(json_error(StatusCode::BAD_REQUEST, &message));
            }
        };
        let record = resolved.record;

        if !record.target.eq_ignore_ascii_case("system") {
            return Action::Respond(json_error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "prompt '{}' version '{}' uses unsupported target '{}'",
                    record.prompt_name, record.version, record.target
                ),
            ));
        }

        let rendered = match render_prompt_template(&record.template_text, &prompt_ref.variables) {
            Ok(rendered) => rendered,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };

        let protocol = infer_protocol(&request_path, request_json);
        let apply_result = match protocol {
            Some(PromptProtocol::OpenAiChat) => apply_openai_system_prompt(request_json, &rendered),
            Some(PromptProtocol::AnthropicMessages) => {
                apply_anthropic_system_prompt(request_json, &rendered)
            }
            None => Err(
                "trp_prompt_ref currently supports OpenAI chat completions and Anthropic messages requests"
                    .to_string(),
            ),
        };
        if let Err(error) = apply_result {
            return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error));
        }

        if let Err(error) = sync_cached_request_json_body(ctx) {
            return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
        }
        ctx.extensions.insert(PromptResolutionAudit {
            prompt_name: record.prompt_name,
            prompt_version: record.version,
            prompt_environment: record.environment,
            prompt_rollout_id: resolved.rollout_id,
            prompt_rollout_mode: resolved.rollout_mode,
        });
        Action::Continue
    }
}

pub fn create_plugin(
    config: &toml::Value,
    governance: Arc<GovernanceState>,
) -> Result<PromptRegistry, Box<dyn std::error::Error>> {
    PromptRegistry::new(config, governance)
}

pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    Ok(Box::new(create_plugin(
        config,
        Arc::new(GovernanceState::new(None)),
    )?))
}

fn prompt_variable_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}").expect("valid prompt variable regex")
    })
}

fn extract_prompt_reference(
    request_json: &mut Value,
    default_environment: &str,
) -> Result<Option<PromptReference>, String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "prompt registry request body must be a JSON object".to_string())?;
    let Some(prompt_ref_value) = object.remove("trp_prompt_ref") else {
        return Ok(None);
    };
    let prompt_ref = prompt_ref_value
        .as_object()
        .ok_or_else(|| "trp_prompt_ref must be an object".to_string())?;

    let name = prompt_ref
        .get("name")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| "trp_prompt_ref.name is required".to_string())?;

    let version = prompt_ref
        .get("version")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let environment = prompt_ref
        .get("environment")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default_environment)
        .to_string();

    let variables = match prompt_ref.get("variables") {
        Some(Value::Object(object)) => object
            .iter()
            .map(|(name, value)| (name.clone(), render_variable_value(value)))
            .collect(),
        Some(_) => return Err("trp_prompt_ref.variables must be an object".to_string()),
        None => HashMap::new(),
    };

    Ok(Some(PromptReference {
        name,
        version,
        environment,
        variables,
    }))
}

fn render_variable_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn stable_rollout_seed(ctx: &RequestContext) -> Option<String> {
    if let Some(seed) = ctx
        .headers
        .get(SESSION_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(seed.to_string());
    }
    ctx.extensions
        .get::<VirtualKeyMeta>()
        .map(|meta| meta.key_hash.clone())
}

fn render_prompt_template(
    template_text: &str,
    variables: &HashMap<String, String>,
) -> Result<String, String> {
    let pattern = prompt_variable_pattern();
    let mut missing = Vec::new();
    for captures in pattern.captures_iter(template_text) {
        let variable = captures
            .get(1)
            .map(|capture| capture.as_str())
            .unwrap_or_default();
        if !variables.contains_key(variable) && !missing.iter().any(|entry| entry == variable) {
            missing.push(variable.to_string());
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "prompt template is missing variables: {}",
            missing.join(", ")
        ));
    }

    Ok(pattern
        .replace_all(template_text, |captures: &regex::Captures<'_>| {
            let variable = captures
                .get(1)
                .map(|capture| capture.as_str())
                .unwrap_or_default();
            variables.get(variable).cloned().unwrap_or_default()
        })
        .into_owned())
}

fn infer_protocol(path: &str, request_json: &Value) -> Option<PromptProtocol> {
    if path.ends_with("/v1/messages") {
        Some(PromptProtocol::AnthropicMessages)
    } else if request_json
        .get("messages")
        .map(|value| value.is_array())
        .unwrap_or(false)
    {
        Some(PromptProtocol::OpenAiChat)
    } else {
        None
    }
}

fn apply_openai_system_prompt(request_json: &mut Value, prompt: &str) -> Result<(), String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "OpenAI prompt request must be a JSON object".to_string())?;
    let messages = object
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "OpenAI prompt request must contain a messages array".to_string())?;
    messages.insert(
        0,
        json!({
            "role": "system",
            "content": prompt,
        }),
    );
    Ok(())
}

fn apply_anthropic_system_prompt(request_json: &mut Value, prompt: &str) -> Result<(), String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "Anthropic prompt request must be a JSON object".to_string())?;
    match object.get_mut("system") {
        Some(Value::String(existing)) => {
            if existing.is_empty() {
                *existing = prompt.to_string();
            } else {
                *existing = format!("{prompt}\n\n{existing}");
            }
        }
        Some(Value::Array(blocks)) => {
            blocks.insert(
                0,
                json!({
                    "type": "text",
                    "text": prompt,
                }),
            );
        }
        Some(Value::Null) | None => {
            object.insert("system".to_string(), Value::String(prompt.to_string()));
        }
        Some(_) => {
            return Err(
                "Anthropic prompt request system must be a string or array of blocks".to_string(),
            )
        }
    }
    Ok(())
}

fn json_error(
    status: StatusCode,
    message: &str,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    let body = json!({ "error": message }).to_string();
    let mut response = Response::builder()
        .status(status)
        .body(
            Full::new(Bytes::from(body))
                .map_err(|never| match never {})
                .boxed(),
        )
        .expect("prompt registry error response");
    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prompt_template_with_scalars_and_json_values() {
        let rendered = render_prompt_template(
            "Hello {{name}}, flags={{flags}}",
            &HashMap::from([
                ("name".to_string(), "Ada".to_string()),
                ("flags".to_string(), "[\"one\",\"two\"]".to_string()),
            ]),
        )
        .expect("rendered");
        assert_eq!(rendered, "Hello Ada, flags=[\"one\",\"two\"]");
    }

    #[test]
    fn render_prompt_template_rejects_missing_variables() {
        let error = render_prompt_template("Hello {{name}} from {{company}}", &HashMap::new())
            .expect_err("missing variables should fail");
        assert!(error.contains("name"));
        assert!(error.contains("company"));
    }
}
