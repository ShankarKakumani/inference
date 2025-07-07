use crate::models::{InferenceError, Tensor, TensorSpec};
use async_trait::async_trait;
use std::fmt::Debug;
use std::any::Any;

/// Represents the supported ML engine types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EngineType {
    Candle,
    Ort,
    Linfa,
}

/// Represents supported model formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    SafeTensors,
    PyTorch,
    Onnx,
    Linfa,
}

/// Core trait for ML inference engines
/// 
/// This trait provides a unified interface for loading and managing ML models
/// across different backend engines (Candle, ONNX Runtime, Linfa).
#[async_trait]
pub trait InferenceEngine: Send + Sync + Debug {
    /// Load a model from a file path
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError>;
    
    /// Load a model from raw bytes
    async fn load_from_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Model>, InferenceError>;
    
    /// Check if this engine supports a specific model format
    fn supports_format(&self, format: &ModelFormat) -> bool;
    
    /// Get the name of this engine
    fn engine_name(&self) -> &'static str;
    
    /// Get the engine type
    fn engine_type(&self) -> EngineType;
}

/// Core trait for loaded ML models
/// 
/// This trait provides a unified interface for making predictions with loaded models,
/// regardless of the underlying engine.
#[async_trait]
pub trait Model: Send + Sync + Debug {
    /// Make a single prediction
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError>;
    
    /// Make batch predictions
    async fn predict_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError>;
    
    /// Get input tensor specifications
    fn input_specs(&self) -> &[TensorSpec];
    
    /// Get output tensor specifications
    fn output_specs(&self) -> &[TensorSpec];
    
    /// Get the engine type that loaded this model
    fn engine_type(&self) -> EngineType;
    
    /// Get model metadata if available
    fn metadata(&self) -> Option<&ModelMetadata> {
        None
    }
    
    /// Get reference to the underlying model as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Optional metadata for models
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
}

// Re-export engine implementations
#[cfg(feature = "candle")]
pub mod candle_engine;
#[cfg(feature = "candle")]
pub use candle_engine::CandleEngine;

#[cfg(feature = "ort")]
pub mod ort_engine;
#[cfg(feature = "ort")]
pub use ort_engine::OrtEngine;

#[cfg(feature = "linfa")]
pub mod linfa_engine;
#[cfg(feature = "linfa")]
pub use linfa_engine::LinfaEngine;

// Engine factory for auto-selection
pub mod factory;
pub use factory::EngineFactory; 