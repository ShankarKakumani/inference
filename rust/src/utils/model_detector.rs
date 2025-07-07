use crate::engines::{EngineType, ModelFormat};
use crate::models::InferenceError;
use std::path::Path;
use std::fs;

/// Model detector for automatic engine selection
pub struct ModelDetector;

impl ModelDetector {
    /// Detect engine type from file path
    pub fn detect_engine_from_path(path: &str) -> EngineType {
        let path = Path::new(path);
        
        // Check file extension first
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "safetensors" => return EngineType::Candle,
                "pt" | "pth" => return EngineType::Candle,

                _ => {}
            }
        }
        
        // If extension doesn't match, try content detection
        if let Ok(engine) = Self::detect_engine_from_content(path) {
            return engine;
        }
        
        // Default fallback to Candle
        EngineType::Candle
    }
    
    /// Detect engine type from file content
    pub fn detect_engine_from_content(path: &Path) -> Result<EngineType, InferenceError> {
        let bytes = fs::read(path)
            .map_err(|e| InferenceError::format_detection(format!("Failed to read file: {}", e)))?;
        
        Self::detect_engine_from_bytes(&bytes)
    }
    
    /// Detect engine type from raw bytes
    pub fn detect_engine_from_bytes(bytes: &[u8]) -> Result<EngineType, InferenceError> {
        // Check for SafeTensors format
        if Self::is_safetensors_format(bytes) {
            return Ok(EngineType::Candle);
        }
        
        // Check for PyTorch format
        if Self::is_pytorch_format(bytes) {
            return Ok(EngineType::Candle);
        }
        
        Err(InferenceError::format_detection(
            "Could not detect model format from content".to_string()
        ))
    }
    
    /// Detect model format from file path
    pub fn detect_format_from_path(path: &str) -> Result<ModelFormat, InferenceError> {
        let path = Path::new(path);
        
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "safetensors" => return Ok(ModelFormat::SafeTensors),
                "pt" | "pth" => return Ok(ModelFormat::PyTorch),

                _ => {}
            }
        }
        
        // Try content detection
        Self::detect_format_from_content(path)
    }
    
    /// Detect model format from file content
    pub fn detect_format_from_content(path: &Path) -> Result<ModelFormat, InferenceError> {
        let bytes = fs::read(path)
            .map_err(|e| InferenceError::format_detection(format!("Failed to read file: {}", e)))?;
        
        Self::detect_format_from_bytes(&bytes)
    }
    
    /// Detect model format from raw bytes
    pub fn detect_format_from_bytes(bytes: &[u8]) -> Result<ModelFormat, InferenceError> {
        if Self::is_safetensors_format(bytes) {
            return Ok(ModelFormat::SafeTensors);
        }
        
        if Self::is_pytorch_format(bytes) {
            return Ok(ModelFormat::PyTorch);
        }
        
        Err(InferenceError::format_detection(
            "Could not detect model format from content".to_string()
        ))
    }
    

    
    /// Check if bytes represent SafeTensors format
    fn is_safetensors_format(bytes: &[u8]) -> bool {
        if bytes.len() < 8 {
            return false;
        }
        
        // SafeTensors files start with JSON metadata length (8 bytes little-endian)
        // followed by JSON metadata that typically starts with '{'
        let json_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]
        ]);
        
        // Sanity check: JSON length should be reasonable
        if json_len > (bytes.len() - 8) as u64 || json_len < 1 {
            return false;
        }
        
        // Check if the JSON starts with '{'
        if bytes.len() > 8 {
            bytes[8] == b'{'
        } else {
            false
        }
    }
    
    /// Check if bytes represent PyTorch format
    fn is_pytorch_format(bytes: &[u8]) -> bool {
        if bytes.len() < 4 {
            return false;
        }
        
        // PyTorch files often start with pickle protocol markers
        // Common patterns include:
        // - Pickle protocol 2: 0x80 0x02
        // - Pickle protocol 3: 0x80 0x03
        // - Pickle protocol 4: 0x80 0x04
        // - ZIP file signature (for newer PyTorch files): 0x50 0x4B
        bytes.starts_with(&[0x80, 0x02]) ||
        bytes.starts_with(&[0x80, 0x03]) ||
        bytes.starts_with(&[0x80, 0x04]) ||
        bytes.starts_with(&[0x50, 0x4B]) || // ZIP signature
        bytes.starts_with(b"PK") // ZIP signature (alternative)
    }
}

/// Convenience function for detecting engine type from path
pub fn detect_engine(path: &str) -> EngineType {
    ModelDetector::detect_engine_from_path(path)
}

/// Convenience function for detecting model format from path
pub fn detect_format(path: &str) -> Result<ModelFormat, InferenceError> {
    ModelDetector::detect_format_from_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_engine_from_extension() {
        assert_eq!(detect_engine("model.safetensors"), EngineType::Candle);
        assert_eq!(detect_engine("model.pt"), EngineType::Candle);
        assert_eq!(detect_engine("model.pth"), EngineType::Candle);
        assert_eq!(detect_engine("model.unknown"), EngineType::Candle); // fallback
    }
    
    #[test]
    fn test_detect_format_from_extension() {
        assert_eq!(detect_format("model.safetensors").unwrap(), ModelFormat::SafeTensors);
        assert_eq!(detect_format("model.pt").unwrap(), ModelFormat::PyTorch);
        assert_eq!(detect_format("model.pth").unwrap(), ModelFormat::PyTorch);
    }
    
    
    #[test]
    fn test_safetensors_format_detection() {
        // Mock SafeTensors format: 8-byte length + JSON starting with '{'
        let mut safetensors_bytes = vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // 1 byte length
        safetensors_bytes.push(b'{'); // JSON start
        assert!(ModelDetector::is_safetensors_format(&safetensors_bytes));
        
        let not_safetensors = vec![0x08, 0x01, 0x12, 0x04];
        assert!(!ModelDetector::is_safetensors_format(&not_safetensors));
    }
    
    #[test]
    fn test_pytorch_format_detection() {
        let pytorch_bytes = vec![0x80, 0x02, 0x00, 0x00];
        assert!(ModelDetector::is_pytorch_format(&pytorch_bytes));
        
        let zip_pytorch = vec![0x50, 0x4B, 0x03, 0x04];
        assert!(ModelDetector::is_pytorch_format(&zip_pytorch));
        
        let not_pytorch = vec![0x08, 0x01, 0x12, 0x04];
        assert!(!ModelDetector::is_pytorch_format(&not_pytorch));
    }
} 