use thiserror::Error;

/// Main error type for the inference system
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    
    #[error("Prediction failed: {0}")]
    Prediction(String),
    
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
    
    #[error("Engine error: {0}")]
    Engine(String),
    
    #[error("IO error: {0}")]
    Io(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
    
    #[error("Thread pool error: {0}")]
    ThreadPool(String),
    
    #[error("GPU error: {0}")]
    Gpu(String),
    
    #[error("Model format detection failed: {0}")]
    FormatDetection(String),
}

impl InferenceError {
    /// Create a model loading error
    pub fn model_load<T: Into<String>>(msg: T) -> Self {
        Self::ModelLoad(msg.into())
    }
    
    /// Create a prediction error
    pub fn prediction<T: Into<String>>(msg: T) -> Self {
        Self::Prediction(msg.into())
    }
    
    /// Create an unsupported format error
    pub fn unsupported_format<T: Into<String>>(format: T) -> Self {
        Self::UnsupportedFormat(format.into())
    }
    
    /// Create an invalid shape error
    pub fn invalid_shape(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::InvalidShape { expected, actual }
    }
    
    /// Create an invalid tensor data error
    pub fn invalid_tensor_data<T: Into<String>>(msg: T) -> Self {
        Self::InvalidTensorData(msg.into())
    }
    
    /// Create a simple invalid shape error with just a message
    pub fn invalid_shape_msg<T: Into<String>>(msg: T) -> Self {
        Self::InvalidTensorData(msg.into())
    }
    
    /// Create a configuration error
    pub fn configuration<T: Into<String>>(msg: T) -> Self {
        Self::Configuration(msg.into())
    }
    
    /// Create a resource not found error
    pub fn resource_not_found<T: Into<String>>(resource: T) -> Self {
        Self::ResourceNotFound(resource.into())
    }
    
    /// Create a memory allocation error
    pub fn memory_allocation<T: Into<String>>(msg: T) -> Self {
        Self::MemoryAllocation(msg.into())
    }
    
    /// Create a GPU error
    pub fn gpu<T: Into<String>>(msg: T) -> Self {
        Self::Gpu(msg.into())
    }
    
    /// Create a format detection error
    pub fn format_detection<T: Into<String>>(msg: T) -> Self {
        Self::FormatDetection(msg.into())
    }
}

// Manual conversion implementations to handle the error types that FRB can't serialize
impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        Self::Engine(err.to_string())
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl From<serde_json::Error> for InferenceError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}