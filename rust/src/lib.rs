pub mod api;
pub mod engines;
pub mod models;
pub mod utils;
mod frb_generated;

// Re-export core types for convenience
pub use engines::{InferenceEngine, Model, EngineType, ModelFormat};
pub use models::{InferenceError, Tensor, TensorSpec, DataType, Preprocessor};
pub use models::tensor::TensorInfo;
pub use models::session::{Session, SessionMetadata, SessionBuilder};
pub use models::preprocessing::{
    ImagePreprocessConfig, TextPreprocessConfig, AudioPreprocessConfig,
    Normalization, ImageFormat
};
pub use utils::{ModelDetector, TensorConverter, detect_engine, detect_format};
