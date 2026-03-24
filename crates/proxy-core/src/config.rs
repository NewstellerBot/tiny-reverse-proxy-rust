use glob::Pattern;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;

/// Resolve a string value: if it starts with `$`, look up the env var.
fn resolve_env(value: &str) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(var_name) = value.strip_prefix('$') {
        std::env::var(var_name)
            .map_err(|_| format!("environment variable {} not set", var_name).into())
    } else {
        Ok(value.to_string())
    }
}

fn parse_u16(value: i64, field: &str) -> Result<u16, Box<dyn std::error::Error>> {
    u16::try_from(value)
        .map_err(|_| format!("{field} must be in range 0..=65535, got {value}").into())
}

fn parse_u64(value: i64, field: &str) -> Result<u64, Box<dyn std::error::Error>> {
    u64::try_from(value).map_err(|_| format!("{field} must be >= 0, got {value}").into())
}

fn parse_u32(value: i64, field: &str) -> Result<u32, Box<dyn std::error::Error>> {
    u32::try_from(value).map_err(|_| format!("{field} must be >= 0, got {value}").into())
}

#[derive(Debug, Clone, PartialEq)]
pub struct HealthCheckConfig {
    pub interval_secs: u64,
    pub timeout_secs: u64,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitConfig {
    pub requests_per_second: f64,
    pub burst: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub cooldown_secs: u64,
    pub window_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CacheConfig {
    pub max_size_mb: u64,
    pub default_ttl_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityConfig {
    pub max_inflight_requests: Option<u32>,
    pub brownout_inflight_requests: Option<u32>,
    pub retry_budget_per_request: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginConfig {
    pub name: String,
    pub enabled: bool,
    pub config: toml::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PreviewFeature {
    #[serde(rename = "responses_composed_streaming")]
    ResponsesComposedStreaming,
    #[serde(rename = "control_plane_import")]
    ControlPlaneImport,
    #[serde(rename = "provider_surface_translations")]
    ProviderSurfaceTranslations,
}

impl PreviewFeature {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "responses_composed_streaming" => Some(Self::ResponsesComposedStreaming),
            "control_plane_import" => Some(Self::ControlPlaneImport),
            "provider_surface_translations" => Some(Self::ProviderSurfaceTranslations),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ResponsesComposedStreaming => "responses_composed_streaming",
            Self::ControlPlaneImport => "control_plane_import",
            Self::ProviderSurfaceTranslations => "provider_surface_translations",
        }
    }

    pub fn enforcement(&self) -> PreviewFeatureEnforcement {
        match self {
            Self::ResponsesComposedStreaming | Self::ControlPlaneImport => {
                PreviewFeatureEnforcement::HardGate
            }
            Self::ProviderSurfaceTranslations => PreviewFeatureEnforcement::VisibilityOnly,
        }
    }

    pub fn all() -> &'static [Self] {
        const ALL: &[PreviewFeature] = &[
            PreviewFeature::ResponsesComposedStreaming,
            PreviewFeature::ControlPlaneImport,
            PreviewFeature::ProviderSurfaceTranslations,
        ];
        ALL
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewFeatureEnforcement {
    #[serde(rename = "hard_gate")]
    HardGate,
    #[serde(rename = "visibility_only")]
    VisibilityOnly,
}

impl PreviewFeatureEnforcement {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HardGate => "hard_gate",
            Self::VisibilityOnly => "visibility_only",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviewFeatureStatus {
    pub name: String,
    pub enabled: bool,
    pub enforcement: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PreviewFeatureSet {
    features: BTreeSet<PreviewFeature>,
}

impl PreviewFeatureSet {
    pub fn new(features: impl IntoIterator<Item = PreviewFeature>) -> Self {
        Self {
            features: features.into_iter().collect(),
        }
    }

    pub fn contains(&self, feature: PreviewFeature) -> bool {
        self.features.contains(&feature)
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn names(&self) -> Vec<String> {
        self.features
            .iter()
            .map(|feature| feature.as_str().to_string())
            .collect()
    }

    pub fn statuses(&self) -> Vec<PreviewFeatureStatus> {
        PreviewFeature::all()
            .iter()
            .map(|feature| PreviewFeatureStatus {
                name: feature.as_str().to_string(),
                enabled: self.contains(*feature),
                enforcement: feature.enforcement().as_str().to_string(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityLevel {
    #[serde(rename = "stable")]
    Stable,
    #[serde(rename = "preview")]
    Preview,
    #[serde(rename = "experimental")]
    Experimental,
}

impl StabilityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Preview => "preview",
            Self::Experimental => "experimental",
        }
    }
}

/// Configuration for an LLM provider (e.g. OpenAI, Anthropic).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderToolProtocol {
    None,
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderFamily {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "openrouter")]
    OpenRouter,
    #[serde(rename = "custom")]
    Custom,
}

impl ProviderFamily {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "gemini" => Some(Self::Gemini),
            "openrouter" => Some(Self::OpenRouter),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::OpenRouter => "openrouter",
            Self::Custom => "custom",
        }
    }

    pub fn stability(&self) -> StabilityLevel {
        match self {
            Self::OpenAi => StabilityLevel::Stable,
            Self::Anthropic | Self::Gemini | Self::OpenRouter => StabilityLevel::Preview,
            Self::Custom => StabilityLevel::Experimental,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolSurface {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "anthropic")]
    Anthropic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponsesSurface {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileSurface {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchSurface {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptCacheProtocol {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "anthropic")]
    Anthropic,
}

impl PromptCacheProtocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheSurface {
    pub protocol: PromptCacheProtocol,
    #[serde(default)]
    pub request_controls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageSurfaceProtocol {
    #[serde(rename = "openai_images")]
    OpenAiImages,
    #[serde(rename = "openrouter_chat_images")]
    OpenRouterChatImages,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSurface {
    pub protocol: ImageSurfaceProtocol,
    #[serde(default)]
    pub input: bool,
    #[serde(default)]
    pub generations: bool,
    #[serde(default)]
    pub edits: bool,
    #[serde(default)]
    pub variations: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioSurfaceProtocol {
    #[serde(rename = "openai_audio")]
    OpenAiAudio,
    #[serde(rename = "openrouter_chat_audio")]
    OpenRouterChatAudio,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSurface {
    pub protocol: AudioSurfaceProtocol,
    #[serde(default)]
    pub input: bool,
    #[serde(default)]
    pub output: bool,
    #[serde(default)]
    pub transcription: bool,
    #[serde(default)]
    pub translation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingSurfaceProtocol {
    #[serde(rename = "openai_embeddings")]
    OpenAiEmbeddings,
    #[serde(rename = "gemini_embed_content")]
    GeminiEmbedContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSurface {
    pub protocol: EmbeddingSurfaceProtocol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealtimeSurface {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManagedToolRequestShape {
    #[serde(rename = "openai_chat_completions")]
    OpenAiChatCompletions,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

impl ManagedToolRequestShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiChatCompletions => "openai_chat_completions",
            Self::OpenAiResponses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProviderCapabilityFlags {
    pub supports_responses_api: bool,
    pub supports_reasoning: bool,
    pub supports_structured_output_json_mode: bool,
    pub supports_structured_output_json_schema: bool,
    pub supports_files: bool,
    pub supports_batches: bool,
    pub supports_image_input: bool,
    pub supports_images_generations: bool,
    pub supports_images_edits: bool,
    pub supports_images_variations: bool,
    pub supports_audio_input: bool,
    pub supports_audio_output: bool,
    pub supports_audio_transcription: bool,
    pub supports_audio_translation: bool,
    pub supports_embeddings: bool,
    pub supports_realtime: bool,
    pub supports_prompt_cache_openai: bool,
    pub supports_prompt_cache_anthropic: bool,
    pub supports_prompt_cache_request_controls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRuntimeSemantics {
    pub managed_tool_request_shapes: Vec<String>,
    pub surface_endpoints: Vec<String>,
    pub tool_protocol: String,
    pub image_protocol: String,
    pub audio_protocol: String,
    pub embedding_protocol: String,
    pub supports_managed_tools: bool,
    pub capabilities: Vec<String>,
    #[serde(flatten)]
    pub capability_flags: ProviderCapabilityFlags,
}

impl Default for ProviderRuntimeSemantics {
    fn default() -> Self {
        Self {
            managed_tool_request_shapes: Vec::new(),
            surface_endpoints: Vec::new(),
            tool_protocol: ProviderToolProtocol::None.as_str().to_string(),
            image_protocol: ProviderImageProtocol::None.as_str().to_string(),
            audio_protocol: ProviderAudioProtocol::None.as_str().to_string(),
            embedding_protocol: ProviderEmbeddingProtocol::None.as_str().to_string(),
            supports_managed_tools: false,
            capabilities: Vec::new(),
            capability_flags: ProviderCapabilityFlags::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderPromptCacheSemantics {
    pub prompt_cache_protocol: String,
    pub supports_prompt_cache: bool,
    pub request_controls_supported: bool,
}

impl Default for ProviderPromptCacheSemantics {
    fn default() -> Self {
        Self {
            prompt_cache_protocol: "none".to_string(),
            supports_prompt_cache: false,
            request_controls_supported: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProviderSurfaceCatalog {
    #[serde(default)]
    pub tools: Option<ToolSurface>,
    #[serde(default)]
    pub responses: Option<ResponsesSurface>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub structured_output_json_mode: bool,
    #[serde(default)]
    pub structured_output_json_schema: bool,
    #[serde(default)]
    pub files: Option<FileSurface>,
    #[serde(default)]
    pub batches: Option<BatchSurface>,
    #[serde(default)]
    pub images: Option<ImageSurface>,
    #[serde(default)]
    pub audio: Option<AudioSurface>,
    #[serde(default)]
    pub embeddings: Option<EmbeddingSurface>,
    #[serde(default)]
    pub realtime: Option<RealtimeSurface>,
    #[serde(default)]
    pub prompt_cache: Option<PromptCacheSurface>,
}

impl ProviderSurfaceCatalog {
    pub fn is_empty(&self) -> bool {
        self.tools.is_none()
            && self.responses.is_none()
            && !self.reasoning
            && !self.structured_output_json_mode
            && !self.structured_output_json_schema
            && self.files.is_none()
            && self.batches.is_none()
            && self.images.is_none()
            && self.audio.is_none()
            && self.embeddings.is_none()
            && self.realtime.is_none()
            && self.prompt_cache.is_none()
    }

    pub fn has_surface(&self, capability: ProviderExtraCapability) -> bool {
        match capability {
            ProviderExtraCapability::ResponsesApi => self.responses.is_some(),
            ProviderExtraCapability::Reasoning => self.reasoning,
            ProviderExtraCapability::StructuredOutputJsonMode => self.structured_output_json_mode,
            ProviderExtraCapability::StructuredOutputJsonSchema => {
                self.structured_output_json_schema
            }
            ProviderExtraCapability::Files => self.files.is_some(),
            ProviderExtraCapability::Batches => self.batches.is_some(),
            ProviderExtraCapability::ImageInput => self
                .images
                .as_ref()
                .map(|surface| surface.input)
                .unwrap_or(false),
            ProviderExtraCapability::ImagesGenerations => self
                .images
                .as_ref()
                .map(|surface| surface.generations)
                .unwrap_or(false),
            ProviderExtraCapability::ImagesEdits => self
                .images
                .as_ref()
                .map(|surface| surface.edits)
                .unwrap_or(false),
            ProviderExtraCapability::ImagesVariations => self
                .images
                .as_ref()
                .map(|surface| surface.variations)
                .unwrap_or(false),
            ProviderExtraCapability::AudioInput => self
                .audio
                .as_ref()
                .map(|surface| surface.input)
                .unwrap_or(false),
            ProviderExtraCapability::AudioOutput => self
                .audio
                .as_ref()
                .map(|surface| surface.output)
                .unwrap_or(false),
            ProviderExtraCapability::AudioTranscription => self
                .audio
                .as_ref()
                .map(|surface| surface.transcription)
                .unwrap_or(false),
            ProviderExtraCapability::AudioTranslation => self
                .audio
                .as_ref()
                .map(|surface| surface.translation)
                .unwrap_or(false),
            ProviderExtraCapability::Embeddings => self.embeddings.is_some(),
            ProviderExtraCapability::Realtime => self.realtime.is_some(),
            ProviderExtraCapability::PromptCacheOpenAi => matches!(
                self.prompt_cache.as_ref().map(|surface| &surface.protocol),
                Some(PromptCacheProtocol::OpenAi)
            ),
            ProviderExtraCapability::PromptCacheAnthropic => matches!(
                self.prompt_cache.as_ref().map(|surface| &surface.protocol),
                Some(PromptCacheProtocol::Anthropic)
            ),
            ProviderExtraCapability::PromptCacheRequestControls => self
                .prompt_cache
                .as_ref()
                .map(|surface| surface.request_controls)
                .unwrap_or(false),
        }
    }

    pub fn derived_capabilities(&self) -> ProviderCapabilityConfig {
        let mut capabilities = ProviderCapabilityConfig::default();
        for capability in [
            ProviderExtraCapability::ResponsesApi,
            ProviderExtraCapability::Reasoning,
            ProviderExtraCapability::StructuredOutputJsonMode,
            ProviderExtraCapability::StructuredOutputJsonSchema,
            ProviderExtraCapability::Files,
            ProviderExtraCapability::Batches,
            ProviderExtraCapability::ImageInput,
            ProviderExtraCapability::ImagesGenerations,
            ProviderExtraCapability::ImagesEdits,
            ProviderExtraCapability::ImagesVariations,
            ProviderExtraCapability::AudioInput,
            ProviderExtraCapability::AudioOutput,
            ProviderExtraCapability::AudioTranscription,
            ProviderExtraCapability::AudioTranslation,
            ProviderExtraCapability::Embeddings,
            ProviderExtraCapability::Realtime,
            ProviderExtraCapability::PromptCacheOpenAi,
            ProviderExtraCapability::PromptCacheAnthropic,
            ProviderExtraCapability::PromptCacheRequestControls,
        ] {
            if self.has_surface(capability) {
                capabilities.enable(capability);
            }
        }
        capabilities
    }

    pub fn derived_tool_protocol(&self) -> ProviderToolProtocol {
        match self.tools {
            Some(ToolSurface::OpenAi) => ProviderToolProtocol::OpenAi,
            Some(ToolSurface::Anthropic) => ProviderToolProtocol::Anthropic,
            None => ProviderToolProtocol::None,
        }
    }

    pub fn derived_image_protocol(&self) -> ProviderImageProtocol {
        match self
            .images
            .as_ref()
            .filter(|surface| surface.generations || surface.edits || surface.variations)
            .map(|surface| &surface.protocol)
        {
            Some(ImageSurfaceProtocol::OpenAiImages) => ProviderImageProtocol::OpenAiImages,
            Some(ImageSurfaceProtocol::OpenRouterChatImages) => {
                ProviderImageProtocol::OpenRouterChatImages
            }
            None => ProviderImageProtocol::None,
        }
    }

    pub fn derived_audio_protocol(&self) -> ProviderAudioProtocol {
        match self
            .audio
            .as_ref()
            .filter(|surface| surface.output || surface.transcription || surface.translation)
            .map(|surface| &surface.protocol)
        {
            Some(AudioSurfaceProtocol::OpenAiAudio) => ProviderAudioProtocol::OpenAiAudio,
            Some(AudioSurfaceProtocol::OpenRouterChatAudio) => {
                ProviderAudioProtocol::OpenRouterChatAudio
            }
            None => ProviderAudioProtocol::None,
        }
    }

    pub fn derived_embedding_protocol(&self) -> ProviderEmbeddingProtocol {
        match self.embeddings.as_ref().map(|surface| &surface.protocol) {
            Some(EmbeddingSurfaceProtocol::OpenAiEmbeddings) => {
                ProviderEmbeddingProtocol::OpenAiEmbeddings
            }
            Some(EmbeddingSurfaceProtocol::GeminiEmbedContent) => {
                ProviderEmbeddingProtocol::GeminiEmbedContent
            }
            None => ProviderEmbeddingProtocol::None,
        }
    }

    pub fn supports_native_batch_endpoint(&self, endpoint: &str) -> bool {
        match endpoint {
            "/v1/responses" => self.responses.is_some(),
            "/v1/files" => self.files.is_some(),
            "/v1/images/generations" => matches!(
                self.images.as_ref(),
                Some(ImageSurface {
                    protocol: ImageSurfaceProtocol::OpenAiImages,
                    generations: true,
                    ..
                })
            ),
            "/v1/images/edits" => matches!(
                self.images.as_ref(),
                Some(ImageSurface {
                    protocol: ImageSurfaceProtocol::OpenAiImages,
                    edits: true,
                    ..
                })
            ),
            "/v1/images/variations" => matches!(
                self.images.as_ref(),
                Some(ImageSurface {
                    protocol: ImageSurfaceProtocol::OpenAiImages,
                    variations: true,
                    ..
                })
            ),
            "/v1/audio/speech" => matches!(
                self.audio.as_ref(),
                Some(AudioSurface {
                    protocol: AudioSurfaceProtocol::OpenAiAudio,
                    output: true,
                    ..
                })
            ),
            "/v1/audio/transcriptions" => matches!(
                self.audio.as_ref(),
                Some(AudioSurface {
                    protocol: AudioSurfaceProtocol::OpenAiAudio,
                    transcription: true,
                    ..
                })
            ),
            "/v1/audio/translations" => matches!(
                self.audio.as_ref(),
                Some(AudioSurface {
                    protocol: AudioSurfaceProtocol::OpenAiAudio,
                    translation: true,
                    ..
                })
            ),
            "/v1/embeddings" => matches!(
                self.embeddings.as_ref(),
                Some(EmbeddingSurface {
                    protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                })
            ),
            _ => true,
        }
    }

    pub fn supports_file_surface(&self) -> bool {
        self.files.is_some()
    }

    pub fn supports_responses_surface(&self) -> bool {
        self.responses.is_some()
    }

    pub fn surface_endpoint_paths(&self) -> Vec<&'static str> {
        let mut endpoints = Vec::new();
        if self.responses.is_some() {
            endpoints.push("/v1/responses");
        }
        if self.files.is_some() {
            endpoints.push("/v1/files");
        }
        if self.batches.is_some() {
            endpoints.push("/v1/batches");
        }
        if self
            .images
            .as_ref()
            .map(|surface| surface.generations)
            .unwrap_or(false)
        {
            endpoints.push("/v1/images/generations");
        }
        if self
            .images
            .as_ref()
            .map(|surface| surface.edits)
            .unwrap_or(false)
        {
            endpoints.push("/v1/images/edits");
        }
        if self
            .images
            .as_ref()
            .map(|surface| surface.variations)
            .unwrap_or(false)
        {
            endpoints.push("/v1/images/variations");
        }
        if self
            .audio
            .as_ref()
            .map(|surface| surface.output)
            .unwrap_or(false)
        {
            endpoints.push("/v1/audio/speech");
        }
        if self
            .audio
            .as_ref()
            .map(|surface| surface.transcription)
            .unwrap_or(false)
        {
            endpoints.push("/v1/audio/transcriptions");
        }
        if self
            .audio
            .as_ref()
            .map(|surface| surface.translation)
            .unwrap_or(false)
        {
            endpoints.push("/v1/audio/translations");
        }
        if self.embeddings.is_some() {
            endpoints.push("/v1/embeddings");
        }
        if self.realtime.is_some() {
            endpoints.push("/v1/realtime");
        }
        endpoints
    }

    pub fn managed_tool_request_shapes(&self) -> Vec<ManagedToolRequestShape> {
        match self.tools.as_ref() {
            Some(ToolSurface::OpenAi) => {
                let mut shapes = vec![ManagedToolRequestShape::OpenAiChatCompletions];
                if self.supports_responses_surface() {
                    shapes.push(ManagedToolRequestShape::OpenAiResponses);
                }
                shapes
            }
            Some(ToolSurface::Anthropic) => vec![ManagedToolRequestShape::AnthropicMessages],
            None => Vec::new(),
        }
    }

    pub fn managed_tool_request_shape_for_path(
        &self,
        path: &str,
    ) -> Option<ManagedToolRequestShape> {
        match self.tools.as_ref() {
            Some(ToolSurface::OpenAi)
                if (path == "/v1/responses" || path.starts_with("/v1/responses/"))
                    && self.supports_responses_surface() =>
            {
                Some(ManagedToolRequestShape::OpenAiResponses)
            }
            Some(ToolSurface::OpenAi) => Some(ManagedToolRequestShape::OpenAiChatCompletions),
            Some(ToolSurface::Anthropic) => Some(ManagedToolRequestShape::AnthropicMessages),
            None => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCommonConfig {
    pub name: String,
    pub api_key: String,
    pub base_url: String,
    pub models: Vec<String>,
    pub api_key_header: String,
    pub timeout_secs: Option<u64>,
    pub routing_metadata: ProviderRoutingMetadataConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderFamilyConfig {
    OpenAi { surfaces: ProviderSurfaceCatalog },
    Anthropic { surfaces: ProviderSurfaceCatalog },
    Gemini { surfaces: ProviderSurfaceCatalog },
    OpenRouter { surfaces: ProviderSurfaceCatalog },
    Custom { surfaces: ProviderSurfaceCatalog },
}

impl ProviderFamilyConfig {
    pub fn family(&self) -> ProviderFamily {
        match self {
            Self::OpenAi { .. } => ProviderFamily::OpenAi,
            Self::Anthropic { .. } => ProviderFamily::Anthropic,
            Self::Gemini { .. } => ProviderFamily::Gemini,
            Self::OpenRouter { .. } => ProviderFamily::OpenRouter,
            Self::Custom { .. } => ProviderFamily::Custom,
        }
    }

    pub fn surfaces(&self) -> &ProviderSurfaceCatalog {
        match self {
            Self::OpenAi { surfaces }
            | Self::Anthropic { surfaces }
            | Self::Gemini { surfaces }
            | Self::OpenRouter { surfaces }
            | Self::Custom { surfaces } => surfaces,
        }
    }

    pub fn from_parts(
        family: ProviderFamily,
        surfaces: ProviderSurfaceCatalog,
    ) -> Result<Self, String> {
        validate_provider_family(family, &surfaces, "provider")?;
        Ok(provider_family_config(family, surfaces))
    }

    pub fn from_optional_parts(
        name: &str,
        family: Option<ProviderFamily>,
        surfaces: ProviderSurfaceCatalog,
    ) -> Result<Self, String> {
        let family = family.unwrap_or_else(|| infer_provider_family(name, &surfaces));
        Self::from_parts(family, surfaces)
    }
}

impl Default for ProviderFamilyConfig {
    fn default() -> Self {
        Self::Custom {
            surfaces: ProviderSurfaceCatalog::default(),
        }
    }
}

impl ProviderToolProtocol {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderImageProtocol {
    None,
    OpenAiImages,
    OpenRouterChatImages,
}

impl ProviderImageProtocol {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "openai_images" => Some(Self::OpenAiImages),
            "openrouter_chat_images" => Some(Self::OpenRouterChatImages),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OpenAiImages => "openai_images",
            Self::OpenRouterChatImages => "openrouter_chat_images",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderAudioProtocol {
    None,
    OpenAiAudio,
    OpenRouterChatAudio,
}

impl ProviderAudioProtocol {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "openai_audio" => Some(Self::OpenAiAudio),
            "openrouter_chat_audio" => Some(Self::OpenRouterChatAudio),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OpenAiAudio => "openai_audio",
            Self::OpenRouterChatAudio => "openrouter_chat_audio",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderEmbeddingProtocol {
    None,
    OpenAiEmbeddings,
    GeminiEmbedContent,
}

impl ProviderEmbeddingProtocol {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "openai_embeddings" => Some(Self::OpenAiEmbeddings),
            "gemini_embed_content" => Some(Self::GeminiEmbedContent),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OpenAiEmbeddings => "openai_embeddings",
            Self::GeminiEmbedContent => "gemini_embed_content",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderDataCollectionMode {
    Allow,
    Deny,
}

impl ProviderDataCollectionMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "allow" => Some(Self::Allow),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderExtraCapability {
    ResponsesApi,
    Reasoning,
    StructuredOutputJsonMode,
    StructuredOutputJsonSchema,
    Files,
    Batches,
    ImageInput,
    ImagesGenerations,
    ImagesEdits,
    ImagesVariations,
    AudioInput,
    AudioOutput,
    AudioTranscription,
    AudioTranslation,
    Embeddings,
    Realtime,
    PromptCacheOpenAi,
    PromptCacheAnthropic,
    PromptCacheRequestControls,
}

impl ProviderExtraCapability {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "responses_api" => Some(Self::ResponsesApi),
            "reasoning" => Some(Self::Reasoning),
            "structured_output_json_mode" => Some(Self::StructuredOutputJsonMode),
            "structured_output_json_schema" => Some(Self::StructuredOutputJsonSchema),
            "files" => Some(Self::Files),
            "batches" => Some(Self::Batches),
            "image_input" => Some(Self::ImageInput),
            "images_generations" => Some(Self::ImagesGenerations),
            "images_edits" => Some(Self::ImagesEdits),
            "images_variations" => Some(Self::ImagesVariations),
            "audio_input" => Some(Self::AudioInput),
            "audio_output" => Some(Self::AudioOutput),
            "audio_transcription" => Some(Self::AudioTranscription),
            "audio_translation" => Some(Self::AudioTranslation),
            "embeddings" => Some(Self::Embeddings),
            "realtime" => Some(Self::Realtime),
            "prompt_cache_openai" => Some(Self::PromptCacheOpenAi),
            "prompt_cache_anthropic" => Some(Self::PromptCacheAnthropic),
            "prompt_cache_request_controls" => Some(Self::PromptCacheRequestControls),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ResponsesApi => "responses_api",
            Self::Reasoning => "reasoning",
            Self::StructuredOutputJsonMode => "structured_output_json_mode",
            Self::StructuredOutputJsonSchema => "structured_output_json_schema",
            Self::Files => "files",
            Self::Batches => "batches",
            Self::ImageInput => "image_input",
            Self::ImagesGenerations => "images_generations",
            Self::ImagesEdits => "images_edits",
            Self::ImagesVariations => "images_variations",
            Self::AudioInput => "audio_input",
            Self::AudioOutput => "audio_output",
            Self::AudioTranscription => "audio_transcription",
            Self::AudioTranslation => "audio_translation",
            Self::Embeddings => "embeddings",
            Self::Realtime => "realtime",
            Self::PromptCacheOpenAi => "prompt_cache_openai",
            Self::PromptCacheAnthropic => "prompt_cache_anthropic",
            Self::PromptCacheRequestControls => "prompt_cache_request_controls",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProviderRoutingMetadataConfig {
    pub data_collection: Option<ProviderDataCollectionMode>,
    pub zdr: bool,
    pub distillable_text: bool,
    pub quantizations: Vec<String>,
    pub supported_parameter_families: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProviderCapabilityConfig {
    pub responses_api: bool,
    pub reasoning: bool,
    pub structured_output_json_mode: bool,
    pub structured_output_json_schema: bool,
    pub files: bool,
    pub batches: bool,
    pub image_input: bool,
    pub images_generations: bool,
    pub images_edits: bool,
    pub images_variations: bool,
    pub audio_input: bool,
    pub audio_output: bool,
    pub audio_transcription: bool,
    pub audio_translation: bool,
    pub embeddings: bool,
    pub realtime: bool,
    pub prompt_cache_openai: bool,
    pub prompt_cache_anthropic: bool,
    pub prompt_cache_request_controls: bool,
}

impl ProviderCapabilityConfig {
    pub fn enable(&mut self, capability: ProviderExtraCapability) {
        match capability {
            ProviderExtraCapability::ResponsesApi => self.responses_api = true,
            ProviderExtraCapability::Reasoning => self.reasoning = true,
            ProviderExtraCapability::StructuredOutputJsonMode => {
                self.structured_output_json_mode = true
            }
            ProviderExtraCapability::StructuredOutputJsonSchema => {
                self.structured_output_json_schema = true
            }
            ProviderExtraCapability::Files => self.files = true,
            ProviderExtraCapability::Batches => self.batches = true,
            ProviderExtraCapability::ImageInput => self.image_input = true,
            ProviderExtraCapability::ImagesGenerations => self.images_generations = true,
            ProviderExtraCapability::ImagesEdits => self.images_edits = true,
            ProviderExtraCapability::ImagesVariations => self.images_variations = true,
            ProviderExtraCapability::AudioInput => self.audio_input = true,
            ProviderExtraCapability::AudioOutput => self.audio_output = true,
            ProviderExtraCapability::AudioTranscription => self.audio_transcription = true,
            ProviderExtraCapability::AudioTranslation => self.audio_translation = true,
            ProviderExtraCapability::Embeddings => self.embeddings = true,
            ProviderExtraCapability::Realtime => self.realtime = true,
            ProviderExtraCapability::PromptCacheOpenAi => self.prompt_cache_openai = true,
            ProviderExtraCapability::PromptCacheAnthropic => self.prompt_cache_anthropic = true,
            ProviderExtraCapability::PromptCacheRequestControls => {
                self.prompt_cache_request_controls = true
            }
        }
    }

    pub fn supports(&self, capability: ProviderExtraCapability) -> bool {
        match capability {
            ProviderExtraCapability::ResponsesApi => self.responses_api,
            ProviderExtraCapability::Reasoning => self.reasoning,
            ProviderExtraCapability::StructuredOutputJsonMode => self.structured_output_json_mode,
            ProviderExtraCapability::StructuredOutputJsonSchema => {
                self.structured_output_json_schema
            }
            ProviderExtraCapability::Files => self.files,
            ProviderExtraCapability::Batches => self.batches,
            ProviderExtraCapability::ImageInput => self.image_input,
            ProviderExtraCapability::ImagesGenerations => self.images_generations,
            ProviderExtraCapability::ImagesEdits => self.images_edits,
            ProviderExtraCapability::ImagesVariations => self.images_variations,
            ProviderExtraCapability::AudioInput => self.audio_input,
            ProviderExtraCapability::AudioOutput => self.audio_output,
            ProviderExtraCapability::AudioTranscription => self.audio_transcription,
            ProviderExtraCapability::AudioTranslation => self.audio_translation,
            ProviderExtraCapability::Embeddings => self.embeddings,
            ProviderExtraCapability::Realtime => self.realtime,
            ProviderExtraCapability::PromptCacheOpenAi => self.prompt_cache_openai,
            ProviderExtraCapability::PromptCacheAnthropic => self.prompt_cache_anthropic,
            ProviderExtraCapability::PromptCacheRequestControls => {
                self.prompt_cache_request_controls
            }
        }
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        for capability in [
            ProviderExtraCapability::ResponsesApi,
            ProviderExtraCapability::Reasoning,
            ProviderExtraCapability::StructuredOutputJsonMode,
            ProviderExtraCapability::StructuredOutputJsonSchema,
            ProviderExtraCapability::Files,
            ProviderExtraCapability::Batches,
            ProviderExtraCapability::ImageInput,
            ProviderExtraCapability::ImagesGenerations,
            ProviderExtraCapability::ImagesEdits,
            ProviderExtraCapability::ImagesVariations,
            ProviderExtraCapability::AudioInput,
            ProviderExtraCapability::AudioOutput,
            ProviderExtraCapability::AudioTranscription,
            ProviderExtraCapability::AudioTranslation,
            ProviderExtraCapability::Embeddings,
            ProviderExtraCapability::Realtime,
            ProviderExtraCapability::PromptCacheOpenAi,
            ProviderExtraCapability::PromptCacheAnthropic,
            ProviderExtraCapability::PromptCacheRequestControls,
        ] {
            if self.supports(capability) {
                names.push(capability.as_str());
            }
        }
        names
    }
}

/// Configuration for an LLM provider (e.g. OpenAI, Anthropic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderKeyConfig {
    pub name: String,
    pub api_key: String,
    pub base_url: String,
    pub models: Vec<String>,
    pub api_key_header: String,
    pub timeout_secs: Option<u64>,
    pub family: ProviderFamilyConfig,
    /// Tool request/response protocol for this provider.
    pub tool_protocol: ProviderToolProtocol,
    /// Image request/response protocol for this provider.
    pub image_protocol: ProviderImageProtocol,
    /// Audio request/response protocol for this provider.
    pub audio_protocol: ProviderAudioProtocol,
    /// Embeddings request/response protocol for this provider.
    pub embedding_protocol: ProviderEmbeddingProtocol,
    /// Optional provider routing metadata used for request-level provider policy.
    #[serde(skip)]
    pub routing_metadata: ProviderRoutingMetadataConfig,
    /// Additional provider-specific request capabilities.
    #[serde(skip)]
    pub capabilities: ProviderCapabilityConfig,
}

impl ProviderKeyConfig {
    pub fn new(common: ProviderCommonConfig, family: ProviderFamilyConfig) -> Self {
        let routing_metadata = common.routing_metadata.clone();
        let surfaces = family.surfaces().clone();
        Self {
            name: common.name,
            api_key: common.api_key,
            base_url: common.base_url,
            models: common.models,
            api_key_header: common.api_key_header,
            timeout_secs: common.timeout_secs,
            family,
            tool_protocol: surfaces.derived_tool_protocol(),
            image_protocol: surfaces.derived_image_protocol(),
            audio_protocol: surfaces.derived_audio_protocol(),
            embedding_protocol: surfaces.derived_embedding_protocol(),
            routing_metadata,
            capabilities: surfaces.derived_capabilities(),
        }
    }

    pub fn family_kind(&self) -> ProviderFamily {
        self.family.family()
    }

    pub fn stability(&self) -> StabilityLevel {
        self.family_kind().stability()
    }

    pub fn surfaces(&self) -> &ProviderSurfaceCatalog {
        self.family.surfaces()
    }

    pub fn tool_protocol_kind(&self) -> ProviderToolProtocol {
        self.surfaces().derived_tool_protocol()
    }

    pub fn image_protocol_kind(&self) -> ProviderImageProtocol {
        self.surfaces().derived_image_protocol()
    }

    pub fn audio_protocol_kind(&self) -> ProviderAudioProtocol {
        self.surfaces().derived_audio_protocol()
    }

    pub fn embedding_protocol_kind(&self) -> ProviderEmbeddingProtocol {
        self.surfaces().derived_embedding_protocol()
    }

    pub fn supports_capability(&self, capability: ProviderExtraCapability) -> bool {
        self.surfaces().has_surface(capability)
    }

    pub fn supports_managed_tools(&self) -> bool {
        self.surfaces().tools.is_some()
    }

    pub fn managed_tool_request_shapes(&self) -> Vec<ManagedToolRequestShape> {
        self.surfaces().managed_tool_request_shapes()
    }

    pub fn capability_names(&self) -> Vec<String> {
        self.surfaces()
            .derived_capabilities()
            .names()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    pub fn capability_flags(&self) -> ProviderCapabilityFlags {
        let surfaces = self.surfaces();
        ProviderCapabilityFlags {
            supports_responses_api: surfaces.has_surface(ProviderExtraCapability::ResponsesApi),
            supports_reasoning: surfaces.has_surface(ProviderExtraCapability::Reasoning),
            supports_structured_output_json_mode: surfaces
                .has_surface(ProviderExtraCapability::StructuredOutputJsonMode),
            supports_structured_output_json_schema: surfaces
                .has_surface(ProviderExtraCapability::StructuredOutputJsonSchema),
            supports_files: surfaces.has_surface(ProviderExtraCapability::Files),
            supports_batches: surfaces.has_surface(ProviderExtraCapability::Batches),
            supports_image_input: surfaces.has_surface(ProviderExtraCapability::ImageInput),
            supports_images_generations: surfaces
                .has_surface(ProviderExtraCapability::ImagesGenerations),
            supports_images_edits: surfaces.has_surface(ProviderExtraCapability::ImagesEdits),
            supports_images_variations: surfaces
                .has_surface(ProviderExtraCapability::ImagesVariations),
            supports_audio_input: surfaces.has_surface(ProviderExtraCapability::AudioInput),
            supports_audio_output: surfaces.has_surface(ProviderExtraCapability::AudioOutput),
            supports_audio_transcription: surfaces
                .has_surface(ProviderExtraCapability::AudioTranscription),
            supports_audio_translation: surfaces
                .has_surface(ProviderExtraCapability::AudioTranslation),
            supports_embeddings: surfaces.has_surface(ProviderExtraCapability::Embeddings),
            supports_realtime: surfaces.has_surface(ProviderExtraCapability::Realtime),
            supports_prompt_cache_openai: surfaces
                .has_surface(ProviderExtraCapability::PromptCacheOpenAi),
            supports_prompt_cache_anthropic: surfaces
                .has_surface(ProviderExtraCapability::PromptCacheAnthropic),
            supports_prompt_cache_request_controls: surfaces
                .has_surface(ProviderExtraCapability::PromptCacheRequestControls),
        }
    }

    pub fn runtime_semantics(&self) -> ProviderRuntimeSemantics {
        let surfaces = self.surfaces();
        ProviderRuntimeSemantics {
            managed_tool_request_shapes: surfaces
                .managed_tool_request_shapes()
                .into_iter()
                .map(|shape| shape.as_str().to_string())
                .collect(),
            surface_endpoints: surfaces
                .surface_endpoint_paths()
                .into_iter()
                .map(str::to_string)
                .collect(),
            tool_protocol: surfaces.derived_tool_protocol().as_str().to_string(),
            image_protocol: surfaces.derived_image_protocol().as_str().to_string(),
            audio_protocol: surfaces.derived_audio_protocol().as_str().to_string(),
            embedding_protocol: surfaces.derived_embedding_protocol().as_str().to_string(),
            supports_managed_tools: surfaces.tools.is_some(),
            capabilities: surfaces
                .derived_capabilities()
                .names()
                .into_iter()
                .map(str::to_string)
                .collect(),
            capability_flags: self.capability_flags(),
        }
    }

    pub fn prompt_cache_semantics(&self) -> ProviderPromptCacheSemantics {
        if let Some(surface) = self.surfaces().prompt_cache.as_ref() {
            return ProviderPromptCacheSemantics {
                prompt_cache_protocol: surface.protocol.as_str().to_string(),
                supports_prompt_cache: true,
                request_controls_supported: surface.request_controls,
            };
        }

        match self.family_kind() {
            ProviderFamily::Anthropic => ProviderPromptCacheSemantics {
                prompt_cache_protocol: PromptCacheProtocol::Anthropic.as_str().to_string(),
                supports_prompt_cache: true,
                request_controls_supported: true,
            },
            ProviderFamily::OpenAi | ProviderFamily::OpenRouter => ProviderPromptCacheSemantics {
                prompt_cache_protocol: PromptCacheProtocol::OpenAi.as_str().to_string(),
                supports_prompt_cache: true,
                request_controls_supported: true,
            },
            ProviderFamily::Gemini | ProviderFamily::Custom => {
                ProviderPromptCacheSemantics::default()
            }
        }
    }

    pub fn refresh_derived_semantics(&mut self) {
        let surfaces = self.family.surfaces();
        self.tool_protocol = surfaces.derived_tool_protocol();
        self.image_protocol = surfaces.derived_image_protocol();
        self.audio_protocol = surfaces.derived_audio_protocol();
        self.embedding_protocol = surfaces.derived_embedding_protocol();
        self.capabilities = surfaces.derived_capabilities();
    }
}

fn parse_bool_field(
    table: &toml::value::Table,
    field: &str,
    context: &str,
) -> Result<bool, String> {
    table
        .get(field)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| format!("{context}.{field} must be a boolean"))
        })
        .transpose()
        .map(|value| value.unwrap_or(false))
}

fn parse_provider_surface_catalog(
    entry: &toml::Value,
    idx: usize,
) -> Result<Option<ProviderSurfaceCatalog>, String> {
    let Some(surfaces) = entry.get("surfaces") else {
        return Ok(None);
    };
    let surfaces = surfaces
        .as_table()
        .ok_or_else(|| format!("providers[{idx}].surfaces must be a table"))?;
    let parse_protocol = |field: &str| surfaces.get(field).and_then(|value| value.as_str());

    let tools = parse_protocol("tools")
        .map(|value| match value {
            "openai" => Ok(ToolSurface::OpenAi),
            "anthropic" => Ok(ToolSurface::Anthropic),
            _ => Err(format!(
                "providers[{idx}].surfaces.tools must be one of: openai, anthropic"
            )),
        })
        .transpose()?;
    let responses = parse_protocol("responses")
        .map(|value| match value {
            "openai_compatible" => Ok(ResponsesSurface::OpenAiCompatible),
            _ => Err(format!(
                "providers[{idx}].surfaces.responses must be 'openai_compatible'"
            )),
        })
        .transpose()?;
    let files = parse_protocol("files")
        .map(|value| match value {
            "openai_compatible" => Ok(FileSurface::OpenAiCompatible),
            _ => Err(format!(
                "providers[{idx}].surfaces.files must be 'openai_compatible'"
            )),
        })
        .transpose()?;
    let batches = parse_protocol("batches")
        .map(|value| match value {
            "openai_compatible" => Ok(BatchSurface::OpenAiCompatible),
            _ => Err(format!(
                "providers[{idx}].surfaces.batches must be 'openai_compatible'"
            )),
        })
        .transpose()?;
    let realtime = parse_protocol("realtime")
        .map(|value| match value {
            "openai_compatible" => Ok(RealtimeSurface::OpenAiCompatible),
            _ => Err(format!(
                "providers[{idx}].surfaces.realtime must be 'openai_compatible'"
            )),
        })
        .transpose()?;
    let images = surfaces
        .get("images")
        .map(|value| {
            let image = value
                .as_table()
                .ok_or_else(|| format!("providers[{idx}].surfaces.images must be a table"))?;
            let protocol = image
                .get("protocol")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("providers[{idx}].surfaces.images.protocol is required")
                })?;
            let protocol = match protocol {
                "openai_images" => ImageSurfaceProtocol::OpenAiImages,
                "openrouter_chat_images" => ImageSurfaceProtocol::OpenRouterChatImages,
                _ => {
                    return Err(format!(
                        "providers[{idx}].surfaces.images.protocol must be one of: openai_images, openrouter_chat_images"
                    ))
                }
            };
            Ok(ImageSurface {
                protocol,
                input: parse_bool_field(
                    image,
                    "input",
                    &format!("providers[{idx}].surfaces.images"),
                )?,
                generations: parse_bool_field(
                    image,
                    "generations",
                    &format!("providers[{idx}].surfaces.images"),
                )?,
                edits: parse_bool_field(
                    image,
                    "edits",
                    &format!("providers[{idx}].surfaces.images"),
                )?,
                variations: parse_bool_field(
                    image,
                    "variations",
                    &format!("providers[{idx}].surfaces.images"),
                )?,
            })
        })
        .transpose()?;
    let audio = surfaces
        .get("audio")
        .map(|value| {
            let audio = value
                .as_table()
                .ok_or_else(|| format!("providers[{idx}].surfaces.audio must be a table"))?;
            let protocol = audio
                .get("protocol")
                .and_then(|value| value.as_str())
                .ok_or_else(|| format!("providers[{idx}].surfaces.audio.protocol is required"))?;
            let protocol = match protocol {
                "openai_audio" => AudioSurfaceProtocol::OpenAiAudio,
                "openrouter_chat_audio" => AudioSurfaceProtocol::OpenRouterChatAudio,
                _ => {
                    return Err(format!(
                        "providers[{idx}].surfaces.audio.protocol must be one of: openai_audio, openrouter_chat_audio"
                    ))
                }
            };
            Ok(AudioSurface {
                protocol,
                input: parse_bool_field(
                    audio,
                    "input",
                    &format!("providers[{idx}].surfaces.audio"),
                )?,
                output: parse_bool_field(
                    audio,
                    "output",
                    &format!("providers[{idx}].surfaces.audio"),
                )?,
                transcription: parse_bool_field(
                    audio,
                    "transcription",
                    &format!("providers[{idx}].surfaces.audio"),
                )?,
                translation: parse_bool_field(
                    audio,
                    "translation",
                    &format!("providers[{idx}].surfaces.audio"),
                )?,
            })
        })
        .transpose()?;
    let embeddings = surfaces
        .get("embeddings")
        .map(|value| {
            let embedding = value.as_table().ok_or_else(|| {
                format!("providers[{idx}].surfaces.embeddings must be a table")
            })?;
            let protocol = embedding
                .get("protocol")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("providers[{idx}].surfaces.embeddings.protocol is required")
                })?;
            let protocol = match protocol {
                "openai_embeddings" => EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                "gemini_embed_content" => EmbeddingSurfaceProtocol::GeminiEmbedContent,
                _ => {
                    return Err(format!(
                        "providers[{idx}].surfaces.embeddings.protocol must be one of: openai_embeddings, gemini_embed_content"
                    ))
                }
            };
            Ok(EmbeddingSurface { protocol })
        })
        .transpose()?;
    let prompt_cache = surfaces
        .get("prompt_cache")
        .map(|value| {
            let prompt_cache = value.as_table().ok_or_else(|| {
                format!("providers[{idx}].surfaces.prompt_cache must be a table")
            })?;
            let protocol = prompt_cache
                .get("protocol")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("providers[{idx}].surfaces.prompt_cache.protocol is required")
                })?;
            let protocol = match protocol {
                "openai" => PromptCacheProtocol::OpenAi,
                "anthropic" => PromptCacheProtocol::Anthropic,
                _ => {
                    return Err(format!(
                        "providers[{idx}].surfaces.prompt_cache.protocol must be one of: openai, anthropic"
                    ))
                }
            };
            Ok(PromptCacheSurface {
                protocol,
                request_controls: parse_bool_field(
                    prompt_cache,
                    "request_controls",
                    &format!("providers[{idx}].surfaces.prompt_cache"),
                )?,
            })
        })
        .transpose()?;

    Ok(Some(ProviderSurfaceCatalog {
        tools,
        responses,
        reasoning: parse_bool_field(surfaces, "reasoning", &format!("providers[{idx}].surfaces"))?,
        structured_output_json_mode: parse_bool_field(
            surfaces,
            "structured_output_json_mode",
            &format!("providers[{idx}].surfaces"),
        )?,
        structured_output_json_schema: parse_bool_field(
            surfaces,
            "structured_output_json_schema",
            &format!("providers[{idx}].surfaces"),
        )?,
        files,
        batches,
        images,
        audio,
        embeddings,
        realtime,
        prompt_cache,
    }))
}

fn has_legacy_provider_semantics(entry: &toml::Value) -> bool {
    [
        "tool_protocol",
        "image_protocol",
        "audio_protocol",
        "embedding_protocol",
        "capabilities",
    ]
    .into_iter()
    .any(|field| entry.get(field).is_some())
}

fn infer_provider_family(name: &str, surfaces: &ProviderSurfaceCatalog) -> ProviderFamily {
    if matches!(surfaces.tools, Some(ToolSurface::Anthropic)) {
        return ProviderFamily::Anthropic;
    }
    if matches!(
        surfaces
            .embeddings
            .as_ref()
            .map(|surface| &surface.protocol),
        Some(EmbeddingSurfaceProtocol::GeminiEmbedContent)
    ) {
        return ProviderFamily::Gemini;
    }
    if matches!(
        surfaces.images.as_ref().map(|surface| &surface.protocol),
        Some(ImageSurfaceProtocol::OpenRouterChatImages)
    ) || matches!(
        surfaces.audio.as_ref().map(|surface| &surface.protocol),
        Some(AudioSurfaceProtocol::OpenRouterChatAudio)
    ) {
        return ProviderFamily::OpenRouter;
    }
    match name {
        "openai" => ProviderFamily::OpenAi,
        "anthropic" => ProviderFamily::Anthropic,
        "gemini" => ProviderFamily::Gemini,
        "openrouter" => ProviderFamily::OpenRouter,
        _ => {
            if surfaces.tools.is_some()
                || surfaces.responses.is_some()
                || surfaces.files.is_some()
                || surfaces.batches.is_some()
                || surfaces.realtime.is_some()
                || matches!(
                    surfaces
                        .prompt_cache
                        .as_ref()
                        .map(|surface| &surface.protocol),
                    Some(PromptCacheProtocol::OpenAi)
                )
            {
                ProviderFamily::OpenAi
            } else {
                ProviderFamily::Custom
            }
        }
    }
}

fn validate_provider_family(
    family: ProviderFamily,
    surfaces: &ProviderSurfaceCatalog,
    context: &str,
) -> Result<(), String> {
    match family {
        ProviderFamily::OpenAi => {
            if matches!(surfaces.tools, Some(ToolSurface::Anthropic)) {
                return Err(format!(
                    "{context}.surfaces.tools='anthropic' is invalid for family='openai'"
                ));
            }
        }
        ProviderFamily::Anthropic => {
            if matches!(surfaces.tools, Some(ToolSurface::OpenAi)) {
                return Err(format!(
                    "{context}.surfaces.tools='openai' is invalid for family='anthropic'"
                ));
            }
            if matches!(
                surfaces
                    .prompt_cache
                    .as_ref()
                    .map(|surface| &surface.protocol),
                Some(PromptCacheProtocol::OpenAi)
            ) {
                return Err(format!("{context}.surfaces.prompt_cache.protocol='openai' is invalid for family='anthropic'"));
            }
        }
        ProviderFamily::Gemini => {
            if matches!(
                surfaces
                    .embeddings
                    .as_ref()
                    .map(|surface| &surface.protocol),
                Some(EmbeddingSurfaceProtocol::OpenAiEmbeddings)
            ) {
                return Err(format!("{context}.surfaces.embeddings.protocol='openai_embeddings' is invalid for family='gemini'"));
            }
        }
        ProviderFamily::OpenRouter => {
            if matches!(
                surfaces.images.as_ref().map(|surface| &surface.protocol),
                Some(ImageSurfaceProtocol::OpenAiImages)
            ) || matches!(
                surfaces.audio.as_ref().map(|surface| &surface.protocol),
                Some(AudioSurfaceProtocol::OpenAiAudio)
            ) {
                return Err(format!("{context} openrouter family must use openrouter_chat_* protocols for image/audio surfaces"));
            }
        }
        ProviderFamily::Custom => {}
    }
    Ok(())
}

fn provider_family_config(
    family: ProviderFamily,
    surfaces: ProviderSurfaceCatalog,
) -> ProviderFamilyConfig {
    match family {
        ProviderFamily::OpenAi => ProviderFamilyConfig::OpenAi { surfaces },
        ProviderFamily::Anthropic => ProviderFamilyConfig::Anthropic { surfaces },
        ProviderFamily::Gemini => ProviderFamilyConfig::Gemini { surfaces },
        ProviderFamily::OpenRouter => ProviderFamilyConfig::OpenRouter { surfaces },
        ProviderFamily::Custom => ProviderFamilyConfig::Custom { surfaces },
    }
}

/// Model alias mapping (friendly name → real model ID).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelAliasConfig {
    pub alias: String,
    pub model: String,
}

/// Load-balancing strategy for a route.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LbStrategy {
    #[default]
    RoundRobin,
    LeastConnections,
    ConsistentHash,
    WeightedRoundRobin,
}

/// A single route entry, supporting both simple arrays and extended table format.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteConfig {
    pub servers: Vec<String>,
    pub lb: LbStrategy,
    pub weights: Option<Vec<u32>>,
}

pub struct Config {
    pub port: u16,
    pub routes: Vec<(Pattern, RouteConfig)>,
    pub health_check: Option<HealthCheckConfig>,
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
    pub tls_auto: bool,
    pub metrics_port: Option<u16>,
    pub management_api_port: Option<u16>,
    pub management_api_token: Option<String>,
    pub allow_direct_provider_keys: bool,
    pub store_url: Option<String>,
    pub max_request_body_bytes: u64,
    pub header_read_timeout_secs: u64,
    pub upstream_timeout_secs: u64,
    pub rate_limit: Option<RateLimitConfig>,
    pub compression_enabled: bool,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    pub cache: Option<CacheConfig>,
    pub reliability: ReliabilityConfig,
    pub preview_features: PreviewFeatureSet,
    pub proxy_protocol: bool,
    pub plugins: Vec<PluginConfig>,
    pub providers: Vec<ProviderKeyConfig>,
    pub model_aliases: Vec<ModelAliasConfig>,
    pub opentelemetry_enabled: bool,
}

impl Config {
    /// Validate the configuration, returning a list of errors if any.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.port == 0 {
            errors.push("port must not be 0".to_string());
        }

        if let Some(mp) = self.metrics_port {
            if mp == 0 {
                errors.push("metrics_port must not be 0".to_string());
            }
            if mp == self.port {
                errors.push("metrics_port must differ from port".to_string());
            }
        }

        if let Some(ap) = self.management_api_port {
            if ap == 0 {
                errors.push("management_api_port must not be 0".to_string());
            }
            if ap == self.port {
                errors.push("management_api_port must differ from port".to_string());
            }
            if self.metrics_port == Some(ap) {
                errors.push("management_api_port must differ from metrics_port".to_string());
            }
        }
        if self
            .management_api_token
            .as_ref()
            .map(|s| s.trim().is_empty())
            .unwrap_or(false)
        {
            errors.push("management_api_token must not be empty when provided".to_string());
        }

        if self.routes.is_empty() {
            errors.push("at least one route must be defined in [paths]".to_string());
        }

        for (pattern, route) in &self.routes {
            let p = pattern.as_str();
            if route.servers.is_empty() {
                errors.push(format!("route '{}': servers must not be empty", p));
            }
            for (i, s) in route.servers.iter().enumerate() {
                if s.is_empty() {
                    errors.push(format!("route '{}': server[{}] must not be empty", p, i));
                }
            }
            if let Some(ref weights) = route.weights {
                if weights.len() != route.servers.len() {
                    errors.push(format!(
                        "route '{}': weights count ({}) must match servers count ({})",
                        p,
                        weights.len(),
                        route.servers.len()
                    ));
                }
            }
        }

        if let Some(ref rl) = self.rate_limit {
            if rl.requests_per_second <= 0.0 {
                errors.push("rate_limit.requests_per_second must be > 0".to_string());
            }
            if rl.burst == 0 {
                errors.push("rate_limit.burst must be > 0".to_string());
            }
        }

        if let Some(ref cb) = self.circuit_breaker {
            if cb.failure_threshold == 0 {
                errors.push("circuit_breaker.failure_threshold must be > 0".to_string());
            }
        }

        if let Some(ref c) = self.cache {
            if c.max_size_mb == 0 {
                errors.push("cache.max_size_mb must be > 0".to_string());
            }
        }

        if let Some(max) = self.reliability.max_inflight_requests {
            if max == 0 {
                errors.push("reliability.max_inflight_requests must be > 0".to_string());
            }
        }

        if let Some(brownout) = self.reliability.brownout_inflight_requests {
            if brownout == 0 {
                errors.push("reliability.brownout_inflight_requests must be > 0".to_string());
            }
            if let Some(max) = self.reliability.max_inflight_requests {
                if brownout > max {
                    errors.push(
                        "reliability.brownout_inflight_requests must be <= reliability.max_inflight_requests"
                            .to_string(),
                    );
                }
            }
        }

        if let Some(ref hc) = self.health_check {
            if hc.interval_secs == 0 {
                errors.push("health_check.interval_secs must be > 0".to_string());
            }
        }

        if self.max_request_body_bytes == 0 {
            errors.push("max_request_body_bytes must be > 0".to_string());
        }

        // TLS file warnings (non-fatal).
        if let Some(ref cert) = self.tls_cert {
            if !std::path::Path::new(cert).exists() {
                tracing::warn!("tls_cert path does not exist: {}", cert);
            }
        }
        if let Some(ref key) = self.tls_key {
            if !std::path::Path::new(key).exists() {
                tracing::warn!("tls_key path does not exist: {}", key);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)?;
        Self::parse(&content)
    }

    pub fn parse(content: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let parsed: toml::Value = toml::from_str(content)?;

        let port = parse_u16(
            parsed
                .get("port")
                .and_then(|p| p.as_integer())
                .ok_or("Port not found in configuration file.")?,
            "port",
        )?;

        let mut routes = Vec::new();
        if let Some(paths) = parsed.get("paths").and_then(|p| p.as_table()) {
            for (path, value) in paths {
                let route_config = if let Some(servers_array) = value.as_array() {
                    // Simple format: "/path" = ["server1", "server2"]
                    let server_list = servers_array
                        .iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect();
                    RouteConfig {
                        servers: server_list,
                        lb: LbStrategy::RoundRobin,
                        weights: None,
                    }
                } else if let Some(table) = value.as_table() {
                    // Extended format: "/path" = { servers = [...], lb = "least-connections" }
                    let servers = table
                        .get("servers")
                        .and_then(|s| s.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|s| s.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();

                    let lb = table
                        .get("lb")
                        .and_then(|v| v.as_str())
                        .map(|s| match s {
                            "least-connections" => LbStrategy::LeastConnections,
                            "consistent-hash" => LbStrategy::ConsistentHash,
                            "weighted-round-robin" => LbStrategy::WeightedRoundRobin,
                            _ => LbStrategy::RoundRobin,
                        })
                        .unwrap_or_default();

                    let weights = if let Some(arr) = table.get("weights").and_then(|w| w.as_array())
                    {
                        let mut parsed_weights = Vec::with_capacity(arr.len());
                        for (idx, v) in arr.iter().enumerate() {
                            let raw = v.as_integer().ok_or_else(|| {
                                format!("paths.{path}.weights[{idx}] must be an integer")
                            })?;
                            parsed_weights
                                .push(parse_u32(raw, &format!("paths.{path}.weights[{idx}]"))?);
                        }
                        Some(parsed_weights)
                    } else {
                        None
                    };

                    RouteConfig {
                        servers,
                        lb,
                        weights,
                    }
                } else {
                    continue;
                };

                let pattern = Pattern::new(path)?;
                routes.push((pattern, route_config));
            }
        }

        routes.sort_by(|a, b| b.0.as_str().len().cmp(&a.0.as_str().len()));

        let health_check = if let Some(hc) = parsed.get("health_check") {
            Some(HealthCheckConfig {
                interval_secs: parse_u64(
                    hc.get("interval_secs")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(10),
                    "health_check.interval_secs",
                )?,
                timeout_secs: parse_u64(
                    hc.get("timeout_secs")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(5),
                    "health_check.timeout_secs",
                )?,
                path: hc
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("/health")
                    .to_string(),
            })
        } else {
            None
        };

        let tls_cert = parsed
            .get("tls_cert")
            .and_then(|v| v.as_str())
            .map(String::from);

        let tls_key = parsed
            .get("tls_key")
            .and_then(|v| v.as_str())
            .map(String::from);

        let tls_auto = parsed.get("tls").and_then(|v| v.as_bool()).unwrap_or(false);

        let metrics_port = match parsed.get("metrics_port").and_then(|v| v.as_integer()) {
            Some(v) => Some(parse_u16(v, "metrics_port")?),
            None => None,
        };

        let management_api_port = match parsed
            .get("management_api_port")
            .and_then(|v| v.as_integer())
        {
            Some(v) => Some(parse_u16(v, "management_api_port")?),
            None => None,
        };
        let management_api_token = match parsed.get("management_api_token").and_then(|v| v.as_str())
        {
            Some(raw) => Some(resolve_env(raw)?),
            None => None,
        };
        let allow_direct_provider_keys = parsed
            .get("allow_direct_provider_keys")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let store_url = parsed
            .get("store_url")
            .and_then(|v| v.as_str())
            .map(String::from);

        let max_request_body_bytes = parse_u64(
            parsed
                .get("max_request_body_bytes")
                .and_then(|v| v.as_integer())
                .unwrap_or(10 * 1024 * 1024),
            "max_request_body_bytes",
        )?; // 10MB default

        let header_read_timeout_secs = parse_u64(
            parsed
                .get("header_read_timeout_secs")
                .and_then(|v| v.as_integer())
                .unwrap_or(30),
            "header_read_timeout_secs",
        )?;

        let upstream_timeout_secs = parse_u64(
            parsed
                .get("upstream_timeout_secs")
                .and_then(|v| v.as_integer())
                .unwrap_or(600),
            "upstream_timeout_secs",
        )?;

        let rate_limit = if let Some(rl) = parsed.get("rate_limit") {
            Some(RateLimitConfig {
                requests_per_second: rl
                    .get("requests_per_second")
                    .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
                    .unwrap_or(100.0),
                burst: parse_u32(
                    rl.get("burst").and_then(|v| v.as_integer()).unwrap_or(50),
                    "rate_limit.burst",
                )?,
            })
        } else {
            None
        };

        let compression_enabled = parsed
            .get("compression_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let circuit_breaker = if let Some(cb) = parsed.get("circuit_breaker") {
            Some(CircuitBreakerConfig {
                failure_threshold: parse_u32(
                    cb.get("failure_threshold")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(5),
                    "circuit_breaker.failure_threshold",
                )?,
                cooldown_secs: parse_u64(
                    cb.get("cooldown_secs")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(30),
                    "circuit_breaker.cooldown_secs",
                )?,
                window_secs: parse_u64(
                    cb.get("window_secs")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(60),
                    "circuit_breaker.window_secs",
                )?,
            })
        } else {
            None
        };

        let cache = if let Some(c) = parsed.get("cache") {
            Some(CacheConfig {
                max_size_mb: parse_u64(
                    c.get("max_size_mb")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(256),
                    "cache.max_size_mb",
                )?,
                default_ttl_secs: parse_u64(
                    c.get("default_ttl_secs")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(300),
                    "cache.default_ttl_secs",
                )?,
            })
        } else {
            None
        };

        let reliability = if let Some(section) = parsed.get("reliability") {
            ReliabilityConfig {
                max_inflight_requests: section
                    .get("max_inflight_requests")
                    .and_then(|v| v.as_integer())
                    .map(|v| parse_u32(v, "reliability.max_inflight_requests"))
                    .transpose()?,
                brownout_inflight_requests: section
                    .get("brownout_inflight_requests")
                    .and_then(|v| v.as_integer())
                    .map(|v| parse_u32(v, "reliability.brownout_inflight_requests"))
                    .transpose()?,
                retry_budget_per_request: parse_u32(
                    section
                        .get("retry_budget_per_request")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(2),
                    "reliability.retry_budget_per_request",
                )?,
            }
        } else {
            ReliabilityConfig {
                max_inflight_requests: None,
                brownout_inflight_requests: None,
                retry_budget_per_request: 2,
            }
        };

        let preview_features = if let Some(values) = parsed.get("preview_features") {
            let array = values
                .as_array()
                .ok_or("preview_features must be an array of strings")?;
            let mut features = Vec::with_capacity(array.len());
            for (idx, value) in array.iter().enumerate() {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("preview_features[{idx}] must be a string"))?;
                let feature = PreviewFeature::parse(raw).ok_or_else(|| {
                    format!("preview_features[{idx}] has unknown feature '{raw}'")
                })?;
                features.push(feature);
            }
            PreviewFeatureSet::new(features)
        } else {
            PreviewFeatureSet::default()
        };

        let proxy_protocol = parsed
            .get("proxy_protocol")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let plugins = parsed
            .get("plugins")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|entry| {
                        let name = entry.get("name")?.as_str()?.to_string();
                        let enabled = entry
                            .get("enabled")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        let config = entry
                            .get("config")
                            .cloned()
                            .unwrap_or(toml::Value::Table(toml::value::Table::new()));
                        Some(PluginConfig {
                            name,
                            enabled,
                            config,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut providers = Vec::new();
        if let Some(arr) = parsed.get("providers").and_then(|v| v.as_array()) {
            for (idx, entry) in arr.iter().enumerate() {
                let name = entry
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("providers[{idx}].name is required"))?
                    .to_string();
                let raw_key = entry
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("providers[{idx}].api_key is required"))?;
                let api_key = resolve_env(raw_key)
                    .map_err(|e| format!("providers[{idx}].api_key could not be resolved: {e}"))?;
                let base_url = entry
                    .get("base_url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("providers[{idx}].base_url is required"))?
                    .to_string();
                let models = entry
                    .get("models")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|m| m.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let api_key_header = entry
                    .get("api_key_header")
                    .and_then(|v| v.as_str())
                    .unwrap_or("authorization")
                    .to_string();
                let timeout_secs = entry
                    .get("timeout_secs")
                    .and_then(|v| v.as_integer())
                    .map(|v| v as u64);
                let routing_metadata = if let Some(metadata) = entry.get("routing_metadata") {
                    let metadata = metadata.as_table().ok_or_else(|| {
                        format!("providers[{idx}].routing_metadata must be a table")
                    })?;
                    let data_collection = metadata
                        .get("data_collection")
                        .map(|value| {
                            let raw = value.as_str().ok_or_else(|| {
                                format!(
                                    "providers[{idx}].routing_metadata.data_collection must be a string"
                                )
                            })?;
                            ProviderDataCollectionMode::parse(raw).ok_or_else(|| {
                                format!(
                                    "providers[{idx}].routing_metadata.data_collection must be one of: allow, deny"
                                )
                            })
                        })
                        .transpose()
                        .map_err(|e| e.to_string())?;
                    let zdr = metadata
                        .get("zdr")
                        .map(|value| {
                            value.as_bool().ok_or_else(|| {
                                format!("providers[{idx}].routing_metadata.zdr must be a boolean")
                            })
                        })
                        .transpose()
                        .map_err(|e| e.to_string())?
                        .unwrap_or(false);
                    let distillable_text = metadata
                        .get("distillable_text")
                        .map(|value| {
                            value.as_bool().ok_or_else(|| {
                                format!(
                                    "providers[{idx}].routing_metadata.distillable_text must be a boolean"
                                )
                            })
                        })
                        .transpose()
                        .map_err(|e| e.to_string())?
                        .unwrap_or(false);
                    let quantizations = metadata
                        .get("quantizations")
                        .map(|value| {
                            value.as_array()
                                .ok_or_else(|| {
                                    format!(
                                        "providers[{idx}].routing_metadata.quantizations must be an array of strings"
                                    )
                                })?
                                .iter()
                                .map(|entry| {
                                    entry.as_str().map(str::to_string).ok_or_else(|| {
                                        format!(
                                            "providers[{idx}].routing_metadata.quantizations entries must be strings"
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()
                        .map_err(|e| e.to_string())?
                        .unwrap_or_default();
                    let supported_parameter_families = metadata
                        .get("supported_parameter_families")
                        .map(|value| {
                            value.as_array()
                                .ok_or_else(|| {
                                    format!(
                                        "providers[{idx}].routing_metadata.supported_parameter_families must be an array of strings"
                                    )
                                })?
                                .iter()
                                .map(|entry| {
                                    entry.as_str().map(str::to_string).ok_or_else(|| {
                                        format!(
                                            "providers[{idx}].routing_metadata.supported_parameter_families entries must be strings"
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()
                        .map_err(|e| e.to_string())?
                        .unwrap_or_default();
                    ProviderRoutingMetadataConfig {
                        data_collection,
                        zdr,
                        distillable_text,
                        quantizations,
                        supported_parameter_families,
                    }
                } else {
                    ProviderRoutingMetadataConfig::default()
                };
                if has_legacy_provider_semantics(entry) {
                    return Err(format!(
                        "providers[{idx}] legacy protocol/capability fields are no longer supported; use family + surfaces"
                    )
                    .into());
                }
                let surfaces = parse_provider_surface_catalog(entry, idx)?.unwrap_or_default();
                let family = entry
                    .get("family")
                    .and_then(|v| v.as_str())
                    .map(|value| {
                        ProviderFamily::parse(value).ok_or_else(|| {
                            format!(
                                "providers[{idx}].family must be one of: openai, anthropic, gemini, openrouter, custom"
                            )
                        })
                    })
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or_else(|| infer_provider_family(name.as_str(), &surfaces));
                validate_provider_family(family, &surfaces, &format!("providers[{idx}]"))
                    .map_err(|e| e.to_string())?;
                providers.push(ProviderKeyConfig::new(
                    ProviderCommonConfig {
                        name,
                        api_key,
                        base_url,
                        models,
                        api_key_header,
                        timeout_secs,
                        routing_metadata,
                    },
                    provider_family_config(family, surfaces),
                ));
            }
        }

        let model_aliases = parsed
            .get("model_aliases")
            .and_then(|v| v.as_table())
            .map(|table| {
                table
                    .iter()
                    .filter_map(|(alias, value)| {
                        let model = value.as_str()?.to_string();
                        Some(ModelAliasConfig {
                            alias: alias.clone(),
                            model,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Config {
            port,
            routes,
            health_check,
            tls_cert,
            tls_key,
            tls_auto,
            metrics_port,
            management_api_port,
            management_api_token,
            allow_direct_provider_keys,
            store_url,
            max_request_body_bytes,
            header_read_timeout_secs,
            upstream_timeout_secs,
            rate_limit,
            compression_enabled,
            circuit_breaker,
            cache,
            reliability,
            preview_features,
            proxy_protocol,
            plugins,
            providers,
            model_aliases,
            opentelemetry_enabled: parsed
                .get("opentelemetry_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_valid_config() -> Config {
        Config {
            port: 8080,
            routes: vec![(
                Pattern::new("/*").unwrap(),
                RouteConfig {
                    servers: vec!["127.0.0.1:3000".to_string()],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            )],
            health_check: None,
            tls_cert: None,
            tls_key: None,
            tls_auto: false,
            metrics_port: None,
            management_api_port: None,
            management_api_token: None,
            allow_direct_provider_keys: false,
            store_url: None,
            max_request_body_bytes: 10 * 1024 * 1024,
            header_read_timeout_secs: 30,
            upstream_timeout_secs: 600,
            rate_limit: None,
            compression_enabled: true,
            circuit_breaker: None,
            cache: None,
            reliability: ReliabilityConfig {
                max_inflight_requests: None,
                brownout_inflight_requests: None,
                retry_budget_per_request: 2,
            },
            preview_features: PreviewFeatureSet::default(),
            proxy_protocol: false,
            plugins: Vec::new(),
            providers: Vec::new(),
            model_aliases: Vec::new(),
            opentelemetry_enabled: false,
        }
    }

    #[test]
    fn valid_config_passes() {
        let config = minimal_valid_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn port_zero_rejected() {
        let mut config = minimal_valid_config();
        config.port = 0;
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("port must not be 0")));
    }

    #[test]
    fn metrics_port_zero_rejected() {
        let mut config = minimal_valid_config();
        config.metrics_port = Some(0);
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("metrics_port must not be 0")));
    }

    #[test]
    fn metrics_port_same_as_port_rejected() {
        let mut config = minimal_valid_config();
        config.metrics_port = Some(8080);
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("metrics_port must differ from port")));
    }

    #[test]
    fn empty_routes_rejected() {
        let mut config = minimal_valid_config();
        config.routes.clear();
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("at least one route")));
    }

    #[test]
    fn empty_servers_rejected() {
        let mut config = minimal_valid_config();
        config.routes[0].1.servers.clear();
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("servers must not be empty")));
    }

    #[test]
    fn empty_server_address_rejected() {
        let mut config = minimal_valid_config();
        config.routes[0].1.servers.push(String::new());
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("server[1] must not be empty")));
    }

    #[test]
    fn weights_count_mismatch_rejected() {
        let mut config = minimal_valid_config();
        config.routes[0].1.weights = Some(vec![1, 2, 3]);
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("weights count (3) must match servers count (1)")));
    }

    #[test]
    fn rate_limit_zero_rps_rejected() {
        let mut config = minimal_valid_config();
        config.rate_limit = Some(RateLimitConfig {
            requests_per_second: 0.0,
            burst: 10,
        });
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("requests_per_second must be > 0")));
    }

    #[test]
    fn rate_limit_zero_burst_rejected() {
        let mut config = minimal_valid_config();
        config.rate_limit = Some(RateLimitConfig {
            requests_per_second: 10.0,
            burst: 0,
        });
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("burst must be > 0")));
    }

    #[test]
    fn circuit_breaker_zero_threshold_rejected() {
        let mut config = minimal_valid_config();
        config.circuit_breaker = Some(CircuitBreakerConfig {
            failure_threshold: 0,
            cooldown_secs: 30,
            window_secs: 60,
        });
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("failure_threshold must be > 0")));
    }

    #[test]
    fn cache_zero_size_rejected() {
        let mut config = minimal_valid_config();
        config.cache = Some(CacheConfig {
            max_size_mb: 0,
            default_ttl_secs: 300,
        });
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("max_size_mb must be > 0")));
    }

    #[test]
    fn health_check_zero_interval_rejected() {
        let mut config = minimal_valid_config();
        config.health_check = Some(HealthCheckConfig {
            interval_secs: 0,
            timeout_secs: 5,
            path: "/health".to_string(),
        });
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("interval_secs must be > 0")));
    }

    #[test]
    fn max_request_body_zero_rejected() {
        let mut config = minimal_valid_config();
        config.max_request_body_bytes = 0;
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("max_request_body_bytes must be > 0")));
    }

    #[test]
    fn empty_management_api_token_rejected() {
        let mut config = minimal_valid_config();
        config.management_api_token = Some("".to_string());
        let errors = config.validate().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("management_api_token must not be empty")));
    }

    #[test]
    fn multiple_errors_collected() {
        let mut config = minimal_valid_config();
        config.port = 0;
        config.routes.clear();
        config.max_request_body_bytes = 0;
        let errors = config.validate().unwrap_err();
        assert!(errors.len() >= 3);
    }

    #[test]
    fn providers_parsed() {
        std::env::set_var("TEST_OPENAI_KEY", "sk-test-123");
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "$TEST_OPENAI_KEY"
base_url = "https://api.openai.com"
models = ["gpt-4o", "gpt-4o-mini"]

[[providers]]
name = "anthropic"
api_key = "sk-ant-literal"
base_url = "https://api.anthropic.com"
models = ["claude-sonnet-4-20250514"]
api_key_header = "x-api-key"
"#;
        let config = Config::parse(config_str).unwrap();
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].name, "openai");
        assert_eq!(config.providers[0].api_key, "sk-test-123");
        assert_eq!(config.providers[0].api_key_header, "authorization");
        assert_eq!(config.providers[0].models, vec!["gpt-4o", "gpt-4o-mini"]);
        assert_eq!(config.providers[1].name, "anthropic");
        assert_eq!(config.providers[1].api_key, "sk-ant-literal");
        assert_eq!(config.providers[1].api_key_header, "x-api-key");
        std::env::remove_var("TEST_OPENAI_KEY");
    }

    #[test]
    fn provider_capabilities_parsed() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "sk-test"
base_url = "https://api.openai.com"
models = ["gpt-4o"]
family = "openai"

[providers.surfaces]
responses = "openai_compatible"
files = "openai_compatible"
batches = "openai_compatible"
structured_output_json_mode = true
structured_output_json_schema = true

[providers.surfaces.images]
protocol = "openai_images"
generations = true
edits = true
variations = true

[providers.surfaces.audio]
protocol = "openai_audio"
transcription = true
translation = true

[providers.surfaces.embeddings]
protocol = "gemini_embed_content"
"#;
        let config = Config::parse(config_str).unwrap();
        let capabilities = &config.providers[0].capabilities;
        assert_eq!(
            config.providers[0].image_protocol,
            ProviderImageProtocol::OpenAiImages
        );
        assert_eq!(
            config.providers[0].audio_protocol,
            ProviderAudioProtocol::OpenAiAudio
        );
        assert_eq!(
            config.providers[0].embedding_protocol,
            ProviderEmbeddingProtocol::GeminiEmbedContent
        );
        assert!(capabilities.responses_api);
        assert!(capabilities.files);
        assert!(capabilities.batches);
        assert!(capabilities.structured_output_json_mode);
        assert!(capabilities.structured_output_json_schema);
        assert!(capabilities.images_generations);
        assert!(capabilities.images_edits);
        assert!(capabilities.images_variations);
        assert!(capabilities.audio_transcription);
        assert!(capabilities.audio_translation);
        assert!(capabilities.embeddings);
        assert!(capabilities
            .names()
            .contains(&"structured_output_json_schema"));
        assert!(capabilities.names().contains(&"files"));
        assert!(capabilities.names().contains(&"batches"));
        assert!(capabilities.names().contains(&"images_generations"));
        assert!(capabilities.names().contains(&"audio_translation"));
        assert!(capabilities.names().contains(&"embeddings"));
    }

    #[test]
    fn preview_features_parsed() {
        let config = Config::parse(
            r#"
port = 8080
preview_features = ["responses_composed_streaming", "control_plane_import"]

[paths]
"/*" = ["http://127.0.0.1:3000"]
"#,
        )
        .unwrap();

        assert!(config
            .preview_features
            .contains(PreviewFeature::ResponsesComposedStreaming));
        assert!(config
            .preview_features
            .contains(PreviewFeature::ControlPlaneImport));
        assert!(!config
            .preview_features
            .contains(PreviewFeature::ProviderSurfaceTranslations));
    }

    #[test]
    fn unknown_preview_feature_rejected() {
        let error = Config::parse(
            r#"
port = 8080
preview_features = ["totally_unknown"]

[paths]
"/*" = ["http://127.0.0.1:3000"]
"#,
        )
        .err()
        .unwrap()
        .to_string();

        assert!(error.contains("preview_features[0] has unknown feature"));
    }

    #[test]
    fn provider_family_and_surfaces_parsed() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "sk-test"
base_url = "https://api.openai.com"
models = ["gpt-4.1-mini"]
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"
reasoning = true
structured_output_json_mode = true
structured_output_json_schema = true
files = "openai_compatible"
batches = "openai_compatible"

[providers.surfaces.prompt_cache]
protocol = "openai"
request_controls = true
"#;

        let config = Config::parse(config_str).unwrap();
        let provider = &config.providers[0];
        assert_eq!(provider.family_kind(), ProviderFamily::OpenAi);
        assert_eq!(provider.surfaces().tools, Some(ToolSurface::OpenAi));
        assert_eq!(
            provider.surfaces().responses,
            Some(ResponsesSurface::OpenAiCompatible)
        );
        assert_eq!(
            provider.surfaces().files,
            Some(FileSurface::OpenAiCompatible)
        );
        assert_eq!(
            provider.surfaces().batches,
            Some(BatchSurface::OpenAiCompatible)
        );
        assert!(provider.capabilities.responses_api);
        assert!(provider.capabilities.batches);
        assert!(provider.capabilities.prompt_cache_openai);
        assert!(provider.capabilities.prompt_cache_request_controls);
    }

    #[test]
    fn native_batch_surface_support_rejects_translated_only_protocols() {
        let translated_images = ProviderSurfaceCatalog {
            batches: Some(BatchSurface::OpenAiCompatible),
            images: Some(ImageSurface {
                protocol: ImageSurfaceProtocol::OpenRouterChatImages,
                input: false,
                generations: true,
                edits: false,
                variations: false,
            }),
            ..ProviderSurfaceCatalog::default()
        };
        assert!(!translated_images.supports_native_batch_endpoint("/v1/images/generations"));

        let native_images = ProviderSurfaceCatalog {
            batches: Some(BatchSurface::OpenAiCompatible),
            images: Some(ImageSurface {
                protocol: ImageSurfaceProtocol::OpenAiImages,
                input: false,
                generations: true,
                edits: false,
                variations: false,
            }),
            ..ProviderSurfaceCatalog::default()
        };
        assert!(native_images.supports_native_batch_endpoint("/v1/images/generations"));

        let translated_audio = ProviderSurfaceCatalog {
            batches: Some(BatchSurface::OpenAiCompatible),
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenRouterChatAudio,
                input: false,
                output: true,
                transcription: false,
                translation: false,
            }),
            ..ProviderSurfaceCatalog::default()
        };
        assert!(!translated_audio.supports_native_batch_endpoint("/v1/audio/speech"));

        let translated_embeddings = ProviderSurfaceCatalog {
            batches: Some(BatchSurface::OpenAiCompatible),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::GeminiEmbedContent,
            }),
            ..ProviderSurfaceCatalog::default()
        };
        assert!(!translated_embeddings.supports_native_batch_endpoint("/v1/embeddings"));
    }

    #[test]
    fn managed_tool_request_shapes_follow_surfaces() {
        let openai_chat_only = ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            ..ProviderSurfaceCatalog::default()
        };
        assert_eq!(
            openai_chat_only
                .managed_tool_request_shapes()
                .into_iter()
                .map(|shape| shape.as_str())
                .collect::<Vec<_>>(),
            vec!["openai_chat_completions"]
        );
        assert_eq!(
            openai_chat_only
                .managed_tool_request_shape_for_path("/v1/responses")
                .map(|shape| shape.as_str()),
            Some("openai_chat_completions")
        );

        let openai_with_responses = ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            responses: Some(ResponsesSurface::OpenAiCompatible),
            ..ProviderSurfaceCatalog::default()
        };
        assert_eq!(
            openai_with_responses
                .managed_tool_request_shapes()
                .into_iter()
                .map(|shape| shape.as_str())
                .collect::<Vec<_>>(),
            vec!["openai_chat_completions", "openai_responses"]
        );
        assert_eq!(
            openai_with_responses
                .managed_tool_request_shape_for_path("/v1/responses")
                .map(|shape| shape.as_str()),
            Some("openai_responses")
        );

        let anthropic = ProviderSurfaceCatalog {
            tools: Some(ToolSurface::Anthropic),
            ..ProviderSurfaceCatalog::default()
        };
        assert_eq!(
            anthropic
                .managed_tool_request_shapes()
                .into_iter()
                .map(|shape| shape.as_str())
                .collect::<Vec<_>>(),
            vec!["anthropic_messages"]
        );
    }

    #[test]
    fn runtime_semantics_follow_surface_inventory() {
        let provider = ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: "openai".to_string(),
                api_key: "sk-test".to_string(),
                base_url: "https://api.openai.test".to_string(),
                models: vec!["gpt-4o".to_string()],
                api_key_header: "authorization".to_string(),
                timeout_secs: Some(30),
                routing_metadata: ProviderRoutingMetadataConfig::default(),
            },
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    responses: Some(ResponsesSurface::OpenAiCompatible),
                    files: Some(FileSurface::OpenAiCompatible),
                    audio: Some(AudioSurface {
                        protocol: AudioSurfaceProtocol::OpenAiAudio,
                        input: false,
                        output: true,
                        transcription: false,
                        translation: true,
                    }),
                    embeddings: Some(EmbeddingSurface {
                        protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                    }),
                    realtime: Some(RealtimeSurface::OpenAiCompatible),
                    prompt_cache: Some(PromptCacheSurface {
                        protocol: PromptCacheProtocol::OpenAi,
                        request_controls: true,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        );

        let semantics = provider.runtime_semantics();
        assert_eq!(semantics.tool_protocol, "openai");
        assert_eq!(semantics.audio_protocol, "openai_audio");
        assert_eq!(semantics.embedding_protocol, "openai_embeddings");
        assert!(semantics.supports_managed_tools);
        assert!(semantics
            .managed_tool_request_shapes
            .iter()
            .any(|shape| shape == "openai_chat_completions"));
        assert!(semantics
            .managed_tool_request_shapes
            .iter()
            .any(|shape| shape == "openai_responses"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/responses"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/files"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/audio/speech"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/audio/translations"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/embeddings"));
        assert!(semantics
            .surface_endpoints
            .iter()
            .any(|path| path == "/v1/realtime"));
        assert!(semantics.capability_flags.supports_responses_api);
        assert!(semantics.capability_flags.supports_files);
        assert!(semantics.capability_flags.supports_audio_output);
        assert!(semantics.capability_flags.supports_audio_translation);
        assert!(semantics.capability_flags.supports_embeddings);
        assert!(semantics.capability_flags.supports_realtime);
        assert!(semantics.capability_flags.supports_prompt_cache_openai);
        assert!(
            semantics
                .capability_flags
                .supports_prompt_cache_request_controls
        );
    }

    #[test]
    fn prompt_cache_semantics_fall_back_to_provider_family() {
        let openai = ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: "openai".to_string(),
                api_key: "sk-test".to_string(),
                base_url: "https://api.openai.test".to_string(),
                models: vec!["gpt-4o".to_string()],
                api_key_header: "authorization".to_string(),
                timeout_secs: None,
                routing_metadata: ProviderRoutingMetadataConfig::default(),
            },
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog::default(),
            },
        );
        let anthropic = ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: "anthropic".to_string(),
                api_key: "sk-test".to_string(),
                base_url: "https://api.anthropic.test".to_string(),
                models: vec!["claude-sonnet-4".to_string()],
                api_key_header: "x-api-key".to_string(),
                timeout_secs: None,
                routing_metadata: ProviderRoutingMetadataConfig::default(),
            },
            ProviderFamilyConfig::Anthropic {
                surfaces: ProviderSurfaceCatalog::default(),
            },
        );

        let openai_semantics = openai.prompt_cache_semantics();
        assert_eq!(openai_semantics.prompt_cache_protocol, "openai");
        assert!(openai_semantics.supports_prompt_cache);
        assert!(openai_semantics.request_controls_supported);

        let anthropic_semantics = anthropic.prompt_cache_semantics();
        assert_eq!(anthropic_semantics.prompt_cache_protocol, "anthropic");
        assert!(anthropic_semantics.supports_prompt_cache);
        assert!(anthropic_semantics.request_controls_supported);
    }

    #[test]
    fn provider_family_rejects_invalid_surface_combination() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openrouter"
api_key = "sk-test"
base_url = "https://openrouter.ai/api/v1"
models = ["openai/gpt-4.1-mini"]
family = "openrouter"

[providers.surfaces.images]
protocol = "openai_images"
generations = true
"#;

        let error = Config::parse(config_str).err().unwrap().to_string();
        assert!(error.contains("openrouter family must use openrouter_chat_* protocols"));
    }

    #[test]
    fn provider_family_rejects_mixing_surfaces_with_legacy_semantics() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "sk-test"
base_url = "https://api.openai.com"
models = ["gpt-4.1-mini"]
family = "openai"
tool_protocol = "openai"

[providers.surfaces]
responses = "openai_compatible"
"#;

        let error = Config::parse(config_str).err().unwrap().to_string();
        assert!(error.contains("legacy protocol/capability fields are no longer supported"));
    }

    #[test]
    fn provider_routing_metadata_parsed() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "sk-test"
base_url = "https://api.openai.com"
models = ["gpt-4o"]

[providers.routing_metadata]
data_collection = "deny"
zdr = true
distillable_text = true
quantizations = ["fp8", "int4"]
supported_parameter_families = ["tools", "prompt_cache_controls"]
"#;
        let config = Config::parse(config_str).unwrap();
        let metadata = &config.providers[0].routing_metadata;
        assert_eq!(
            metadata.data_collection,
            Some(ProviderDataCollectionMode::Deny)
        );
        assert!(metadata.zdr);
        assert!(metadata.distillable_text);
        assert_eq!(metadata.quantizations, vec!["fp8", "int4"]);
        assert_eq!(
            metadata.supported_parameter_families,
            vec!["tools", "prompt_cache_controls"]
        );
    }

    #[test]
    fn model_aliases_parsed() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[model_aliases]
"gpt4" = "gpt-4o"
"claude" = "claude-sonnet-4-20250514"
"#;
        let config = Config::parse(config_str).unwrap();
        assert_eq!(config.model_aliases.len(), 2);
        let aliases: Vec<(&str, &str)> = config
            .model_aliases
            .iter()
            .map(|a| (a.alias.as_str(), a.model.as_str()))
            .collect();
        assert!(aliases.contains(&("gpt4", "gpt-4o")));
        assert!(aliases.contains(&("claude", "claude-sonnet-4-20250514")));
    }

    #[test]
    fn provider_timeout_secs_parsed() {
        std::env::set_var("TEST_TIMEOUT_KEY", "sk-test-timeout");
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "anthropic"
api_key = "$TEST_TIMEOUT_KEY"
base_url = "https://api.anthropic.com"
models = ["claude-sonnet-4-20250514"]
api_key_header = "x-api-key"
timeout_secs = 120
"#;
        let config = Config::parse(config_str).unwrap();
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].timeout_secs, Some(120));
        std::env::remove_var("TEST_TIMEOUT_KEY");
    }

    #[test]
    fn provider_timeout_secs_defaults_to_none() {
        std::env::set_var("TEST_TIMEOUT_KEY2", "sk-test-timeout2");
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "$TEST_TIMEOUT_KEY2"
base_url = "https://api.openai.com"
models = ["gpt-4o"]
"#;
        let config = Config::parse(config_str).unwrap();
        assert_eq!(config.providers[0].timeout_secs, None);
        std::env::remove_var("TEST_TIMEOUT_KEY2");
    }

    #[test]
    fn no_providers_is_empty_vec() {
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]
"#;
        let config = Config::parse(config_str).unwrap();
        assert!(config.providers.is_empty());
        assert!(config.model_aliases.is_empty());
    }

    #[test]
    fn provider_missing_env_var_rejected() {
        std::env::remove_var("MISSING_PROVIDER_KEY_FOR_TEST");
        let config_str = r#"
port = 8080

[paths]
"/*" = ["127.0.0.1:3000"]

[[providers]]
name = "openai"
api_key = "$MISSING_PROVIDER_KEY_FOR_TEST"
base_url = "https://api.openai.com"
models = ["gpt-4o"]
"#;
        let err = match Config::parse(config_str) {
            Ok(_) => panic!("expected missing provider env var to be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("could not be resolved"));
    }

    #[test]
    fn management_api_token_env_resolved() {
        std::env::set_var("TRP_MGMT_TOKEN_TEST", "secret-token");
        let config_str = r#"
port = 8080
management_api_token = "$TRP_MGMT_TOKEN_TEST"

[paths]
"/*" = ["127.0.0.1:3000"]
"#;
        let config = Config::parse(config_str).unwrap();
        assert_eq!(config.management_api_token.as_deref(), Some("secret-token"));
        std::env::remove_var("TRP_MGMT_TOKEN_TEST");
    }

    #[test]
    fn negative_ports_rejected_during_parse() {
        let config_str = r#"
port = -1

[paths]
"/*" = ["127.0.0.1:3000"]
"#;
        assert!(Config::parse(config_str).is_err());
    }

    /// #23: TOML rejects duplicate keys in [paths].
    #[test]
    fn duplicate_route_keys_in_toml_rejected() {
        let config_str = r#"
port = 8080

[paths]
"/api/*" = ["server1:80"]
"/api/*" = ["server2:80"]
"#;
        let result = Config::parse(config_str);
        assert!(
            result.is_err(),
            "TOML parser should reject duplicate keys in [paths]"
        );
    }
}
