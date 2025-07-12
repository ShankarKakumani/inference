pub mod error;
pub mod tensor;
pub mod session;
pub mod preprocessing;

pub use error::InferenceError;
pub use tensor::{Tensor, TensorSpec, DataType};

pub use preprocessing::Preprocessor;

/// Supported model architectures from candle-transformers
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    /// BERT for NLP tasks (text classification, question answering)
    Bert,
    /// ResNet for image classification
    ResNet { variant: ResNetVariant },
    /// MobileNet for mobile-optimized image classification
    MobileNet { variant: MobileNetVariant },
    /// Mistral for text generation
    Mistral,
    /// Llama for text generation
    Llama,
    /// Whisper for speech recognition
    Whisper,
    /// Generic SafeTensors model (user-defined architecture)
    Generic,
}

/// ResNet model variants
#[derive(Debug, Clone, PartialEq)]
pub enum ResNetVariant {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
}

/// MobileNet model variants
#[derive(Debug, Clone, PartialEq)]
pub enum MobileNetVariant {
    V1,
    V2,
    V3Small,
    V3Large,
}

/// Model configuration for loading pre-trained models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// HuggingFace repository ID (e.g., "bert-base-uncased")
    pub repo_id: Option<String>,
    /// Model filename (defaults to architecture-specific name)
    pub filename: Option<String>,
    /// Model revision/branch (defaults to "main")
    pub revision: Option<String>,
    /// Whether to use authentication token
    pub use_auth_token: bool,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(architecture: ModelArchitecture) -> Self {
        Self {
            architecture,
            repo_id: None,
            filename: None,
            revision: Some("main".to_string()),
            use_auth_token: false,
        }
    }
    
    /// Set HuggingFace repository ID
    pub fn with_repo_id(mut self, repo_id: &str) -> Self {
        self.repo_id = Some(repo_id.to_string());
        self
    }
    
    /// Set model filename
    pub fn with_filename(mut self, filename: &str) -> Self {
        self.filename = Some(filename.to_string());
        self
    }
    
    /// Set revision/branch
    pub fn with_revision(mut self, revision: &str) -> Self {
        self.revision = Some(revision.to_string());
        self
    }
    
    /// Enable authentication token
    pub fn with_auth_token(mut self) -> Self {
        self.use_auth_token = true;
        self
    }
    
    /// Get default filename for architecture
    pub fn default_filename(&self) -> &str {
        match &self.architecture {
            ModelArchitecture::Bert => "model.safetensors",
            ModelArchitecture::ResNet { .. } => "model.safetensors",
            ModelArchitecture::MobileNet { .. } => "model.safetensors",
            ModelArchitecture::Mistral => "model.safetensors",
            ModelArchitecture::Llama => "model.safetensors",
            ModelArchitecture::Whisper => "model.safetensors",
            ModelArchitecture::Generic => "model.safetensors",
        }
    }
    
    /// Get default repository ID for well-known models
    pub fn default_repo_id(&self) -> Option<&str> {
        match &self.architecture {
            ModelArchitecture::Bert => Some("bert-base-uncased"),
            ModelArchitecture::ResNet { variant } => match variant {
                ResNetVariant::ResNet18 => Some("microsoft/resnet-18"),
                ResNetVariant::ResNet50 => Some("microsoft/resnet-50"),
                _ => None,
            },
            ModelArchitecture::MobileNet { variant } => match variant {
                MobileNetVariant::V2 => Some("google/mobilenet_v2_1.0_224"),
                _ => None,
            },
            _ => None,
        }
    }
}

 