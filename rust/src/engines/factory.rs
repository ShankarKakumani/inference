use crate::engines::{InferenceEngine, EngineType, ModelFormat};
use crate::models::InferenceError;
use std::path::Path;

#[cfg(feature = "candle")]
use crate::engines::CandleEngine;
#[cfg(feature = "linfa")]
use crate::engines::LinfaEngine;

/// Engine factory for automatic engine selection
/// 
/// This factory can automatically detect the appropriate engine based on:
/// - File extension
/// - File content magic bytes
/// - Explicit engine type specification
pub struct EngineFactory;

impl EngineFactory {
    /// Create an engine based on file path auto-detection
    pub fn from_file(path: &str) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        let format = Self::detect_format_from_path(path)?;
        Self::create_engine(format)
    }
    
    /// Create an engine based on file content auto-detection
    pub fn from_bytes(bytes: &[u8]) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        let format = Self::detect_format_from_bytes(bytes)?;
        Self::create_engine(format)
    }
    
    /// Create an engine for a specific format
    pub fn create_engine(format: ModelFormat) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        match format {
            ModelFormat::SafeTensors | ModelFormat::PyTorch => {
                #[cfg(feature = "candle")]
                {
                    let engine = CandleEngine::new()?;
                    Ok(Box::new(engine))
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(InferenceError::configuration(
                        "Candle engine not available - compile with 'candle' feature"
                    ))
                }
            }
            ModelFormat::Linfa => {
                #[cfg(feature = "linfa")]
                {
                    let engine = LinfaEngine::default();
                    Ok(Box::new(engine))
                }
                #[cfg(not(feature = "linfa"))]
                {
                    Err(InferenceError::configuration(
                        "Linfa engine not available - compile with 'linfa' feature"
                    ))
                }
            }
        }
    }
    
    /// Create an engine for a specific engine type
    pub fn create_engine_by_type(engine_type: EngineType) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        match engine_type {
            EngineType::Candle => {
                #[cfg(feature = "candle")]
                {
                    let engine = CandleEngine::new()?;
                    Ok(Box::new(engine))
                }
                #[cfg(not(feature = "candle"))]
                {
                    Err(InferenceError::configuration(
                        "Candle engine not available - compile with 'candle' feature"
                    ))
                }
            }
            EngineType::Linfa => {
                #[cfg(feature = "linfa")]
                {
                    let engine = LinfaEngine::default();
                    Ok(Box::new(engine))
                }
                #[cfg(not(feature = "linfa"))]
                {
                    Err(InferenceError::configuration(
                        "Linfa engine not available - compile with 'linfa' feature"
                    ))
                }
            }
        }
    }
    
    /// Detect model format from file path
    pub fn detect_format_from_path(path: &str) -> Result<ModelFormat, InferenceError> {
        let path_obj = Path::new(path);
        let extension = path_obj.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| InferenceError::unsupported_format(format!("No file extension found for {}", path)))?;
        
        match extension.to_lowercase().as_str() {
            "safetensors" => Ok(ModelFormat::SafeTensors),
            "pt" | "pth" => Ok(ModelFormat::PyTorch),
            "linfa" | "lfa" => Ok(ModelFormat::Linfa),
            _ => {
                // Try to read file and detect from content
                let bytes = std::fs::read(path)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read file {}: {}", path, e)))?;
                Self::detect_format_from_bytes(&bytes)
            }
        }
    }
    
    /// Detect model format from file content
    pub fn detect_format_from_bytes(bytes: &[u8]) -> Result<ModelFormat, InferenceError> {
        if bytes.is_empty() {
            return Err(InferenceError::unsupported_format("Empty file content"));
        }
        
        
        // SafeTensors format detection (starts with JSON metadata)
        if bytes.starts_with(b"{") {
            // Try to parse as JSON to confirm it's SafeTensors
            if let Ok(json_str) = std::str::from_utf8(&bytes[..std::cmp::min(1024, bytes.len())]) {
                if json_str.contains("__metadata__") || json_str.contains("dtype") {
                    return Ok(ModelFormat::SafeTensors);
                }
            }
        }
        
        // PyTorch format detection (pickle format)
        if bytes.starts_with(b"\x80") {
            return Ok(ModelFormat::PyTorch);
        }
        
        // Linfa format detection (bincode serialized)
        if bytes.len() > 8 {
            // Try to deserialize as bincode - if it works, likely Linfa
            if let Ok(_) = bincode::deserialize::<serde_json::Value>(bytes) {
                return Ok(ModelFormat::Linfa);
            }
        }
        
        // Default fallback to SafeTensors for unknown formats
        Ok(ModelFormat::SafeTensors)
    }
    
    /// Get all available engines
    pub fn available_engines() -> Vec<EngineType> {
        let mut engines = Vec::new();
        
        #[cfg(feature = "candle")]
        engines.push(EngineType::Candle);
        
        
        #[cfg(feature = "linfa")]
        engines.push(EngineType::Linfa);
        
        engines
    }
    
    /// Check if a specific engine is available
    pub fn is_engine_available(engine_type: EngineType) -> bool {
        match engine_type {
            EngineType::Candle => cfg!(feature = "candle"),
            EngineType::Linfa => cfg!(feature = "linfa"),
        }
    }
    
    /// Get the preferred engine for a given format
    pub fn preferred_engine_for_format(format: ModelFormat) -> EngineType {
        match format {
            ModelFormat::SafeTensors | ModelFormat::PyTorch => EngineType::Candle,
            ModelFormat::Linfa => EngineType::Linfa,
        }
    }
    
    /// Create the best available engine for a format
    pub fn create_best_engine_for_format(format: ModelFormat) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        let preferred = Self::preferred_engine_for_format(format);
        
        // Try preferred engine first
        if Self::is_engine_available(preferred) {
            return Self::create_engine_by_type(preferred);
        }
        
        // Fall back to any available engine
        let available = Self::available_engines();
        if available.is_empty() {
            return Err(InferenceError::configuration(
                "No inference engines available - compile with at least one engine feature (candle, ort, linfa)"
            ));
        }
        
        Self::create_engine_by_type(available[0])
    }
}

/// Configuration for engine selection
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Preferred engine type (None for auto-detection)
    pub preferred_engine: Option<EngineType>,
    /// Whether to allow fallback to other engines
    pub allow_fallback: bool,
    /// Whether to enable GPU acceleration when available
    pub gpu_acceleration: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            preferred_engine: None,
            allow_fallback: true,
            gpu_acceleration: true,
        }
    }
}

impl EngineConfig {
    /// Create a new engine configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set preferred engine type
    pub fn with_preferred_engine(mut self, engine_type: EngineType) -> Self {
        self.preferred_engine = Some(engine_type);
        self
    }
    
    /// Set fallback behavior
    pub fn with_fallback(mut self, allow_fallback: bool) -> Self {
        self.allow_fallback = allow_fallback;
        self
    }
    
    /// Set GPU acceleration preference
    pub fn with_gpu_acceleration(mut self, gpu_acceleration: bool) -> Self {
        self.gpu_acceleration = gpu_acceleration;
        self
    }
    
    /// Create an engine using this configuration
    pub fn create_engine(&self, format: ModelFormat) -> Result<Box<dyn InferenceEngine>, InferenceError> {
        // Try preferred engine first
        if let Some(preferred) = self.preferred_engine {
            if EngineFactory::is_engine_available(preferred) {
                return EngineFactory::create_engine_by_type(preferred);
            } else if !self.allow_fallback {
                return Err(InferenceError::configuration(
                    format!("Preferred engine {:?} not available and fallback disabled", preferred)
                ));
            }
        }
        
        // Fall back to best engine for format
        if self.allow_fallback {
            EngineFactory::create_best_engine_for_format(format)
        } else {
            Err(InferenceError::configuration(
                "No suitable engine available and fallback disabled"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(
            EngineFactory::detect_format_from_path("model.safetensors").unwrap(),
            ModelFormat::SafeTensors
        );
        assert_eq!(
            EngineFactory::detect_format_from_path("model.pt").unwrap(),
            ModelFormat::PyTorch
        );
        assert_eq!(
            EngineFactory::detect_format_from_path("model.pth").unwrap(),
            ModelFormat::PyTorch
        );
        assert_eq!(
            EngineFactory::detect_format_from_path("model.linfa").unwrap(),
            ModelFormat::Linfa
        );
    }
    
    #[test]
    fn test_format_detection_from_bytes() {
        
        // SafeTensors format (starts with JSON)
        let safetensors_bytes = b"{\"__metadata__\": {\"format\": \"pt\"}}";
        assert_eq!(
            EngineFactory::detect_format_from_bytes(safetensors_bytes).unwrap(),
            ModelFormat::SafeTensors
        );
        
        // PyTorch format (pickle, starts with 0x80)
        let pytorch_bytes = b"\x80\x02}q\x00";
        assert_eq!(
            EngineFactory::detect_format_from_bytes(pytorch_bytes).unwrap(),
            ModelFormat::PyTorch
        );
        
        // Unknown format defaults to SafeTensors
        let unknown_bytes = b"unknown format";
        assert_eq!(
            EngineFactory::detect_format_from_bytes(unknown_bytes).unwrap(),
            ModelFormat::SafeTensors
        );
    }
    
    #[test]
    fn test_available_engines() {
        let engines = EngineFactory::available_engines();
        assert!(!engines.is_empty(), "At least one engine should be available");
        
        // Check that available engines can be created
        for engine_type in engines {
            assert!(EngineFactory::is_engine_available(engine_type));
        }
    }
    
    #[test]
    fn test_preferred_engine_for_format() {
        assert_eq!(
            EngineFactory::preferred_engine_for_format(ModelFormat::SafeTensors),
            EngineType::Candle
        );
        assert_eq!(
            EngineFactory::preferred_engine_for_format(ModelFormat::PyTorch),
            EngineType::Candle
        );
        assert_eq!(
            EngineFactory::preferred_engine_for_format(ModelFormat::Linfa),
            EngineType::Linfa
        );
    }
    
    #[test]
    fn test_engine_config() {
        let config = EngineConfig::new()
            .with_preferred_engine(EngineType::Candle)
            .with_fallback(false)
            .with_gpu_acceleration(true);
        
        assert_eq!(config.preferred_engine, Some(EngineType::Candle));
        assert!(!config.allow_fallback);
        assert!(config.gpu_acceleration);
    }
    
    #[test]
    fn test_create_engine_for_available_formats() {
        let available_engines = EngineFactory::available_engines();
        
        for engine_type in available_engines {
            let result = EngineFactory::create_engine_by_type(engine_type);
            assert!(result.is_ok(), "Should be able to create available engine {:?}", engine_type);
            
            let engine = result.unwrap();
            assert_eq!(engine.engine_type(), engine_type);
        }
    }
} 