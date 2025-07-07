use crate::engines::{InferenceEngine, Model, EngineType};
use crate::models::{InferenceError, Tensor, TensorSpec};


/// Unified session interface for all ML engines
/// 
/// This provides a common abstraction over different engine implementations,
/// allowing for consistent API regardless of the underlying engine.
#[derive(Debug)]
pub struct Session {
    /// The underlying model implementation
    model: Box<dyn Model>,
    /// The engine that loaded this model
    engine_type: EngineType,
    /// Optional session metadata
    metadata: Option<SessionMetadata>,
}

impl Session {
    /// Create a new session with a loaded model
    pub fn new(model: Box<dyn Model>, engine_type: EngineType) -> Self {
        Self {
            model,
            engine_type,
            metadata: None,
        }
    }
    
    /// Create a session with metadata
    pub fn with_metadata(mut self, metadata: SessionMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Make a prediction with the model
    pub async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        self.model.predict(input).await
    }
    
    /// Make batch predictions
    pub async fn predict_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError> {
        self.model.predict_batch(inputs).await
    }
    
    /// Get input tensor specifications
    pub fn input_specs(&self) -> &[TensorSpec] {
        self.model.input_specs()
    }
    
    /// Get output tensor specifications  
    pub fn output_specs(&self) -> &[TensorSpec] {
        self.model.output_specs()
    }
    
    /// Get the engine type
    pub fn engine_type(&self) -> EngineType {
        self.engine_type.clone()
    }
    
    /// Get session metadata
    pub fn metadata(&self) -> Option<&SessionMetadata> {
        self.metadata.as_ref()
    }
    
    /// Get the underlying model (for engine-specific operations)
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }
}

/// Session metadata and configuration
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Model file path or identifier
    pub model_path: Option<String>,
    /// Model name or description
    pub model_name: Option<String>,
    /// Model version
    pub model_version: Option<String>,
    /// Session creation timestamp
    pub created_at: std::time::SystemTime,
    /// Additional custom metadata
    pub custom_metadata: std::collections::HashMap<String, String>,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_name: None,
            model_version: None,
            created_at: std::time::SystemTime::now(),
            custom_metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Set model path
    pub fn with_model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.model_path = Some(path.into());
        self
    }
    
    /// Set model name
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.model_name = Some(name.into());
        self
    }
    
    /// Set model version
    pub fn with_model_version<S: Into<String>>(mut self, version: S) -> Self {
        self.model_version = Some(version.into());
        self
    }
    
    /// Add custom metadata
    pub fn with_custom<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.custom_metadata.insert(key.into(), value.into());
        self
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Session builder for creating sessions with various configurations
#[derive(Debug)]
pub struct SessionBuilder {
    engine: Option<Box<dyn InferenceEngine>>,
    metadata: SessionMetadata,
}

impl SessionBuilder {
    /// Create a new session builder
    pub fn new() -> Self {
        Self {
            engine: None,
            metadata: SessionMetadata::new(),
        }
    }
    
    /// Set the engine to use
    pub fn with_engine(mut self, engine: Box<dyn InferenceEngine>) -> Self {
        self.engine = Some(engine);
        self
    }
    
    /// Set metadata
    pub fn with_metadata(mut self, metadata: SessionMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// Load a model from path and create session
    pub async fn load_from_path<S: AsRef<str>>(self, path: S) -> Result<Session, InferenceError> {
        let engine = self.engine.ok_or_else(|| {
            InferenceError::configuration("No engine specified for session builder")
        })?;
        
        let model = engine.load_model(path.as_ref()).await?;
        let engine_type = engine.engine_type();
        
        let metadata = self.metadata.with_model_path(path.as_ref());
        
        Ok(Session::new(model, engine_type).with_metadata(metadata))
    }
    
    /// Load a model from bytes and create session
    pub async fn load_from_bytes(self, bytes: &[u8]) -> Result<Session, InferenceError> {
        let engine = self.engine.ok_or_else(|| {
            InferenceError::configuration("No engine specified for session builder")
        })?;
        
        let model = engine.load_from_bytes(bytes).await?;
        let engine_type = engine.engine_type();
        
        Ok(Session::new(model, engine_type).with_metadata(self.metadata))
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
} 