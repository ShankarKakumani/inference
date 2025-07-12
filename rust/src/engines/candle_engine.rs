use crate::engines::{InferenceEngine, Model, EngineType, ModelFormat};
use crate::models::{InferenceError, Tensor, TensorSpec, DataType, ModelArchitecture, ModelConfig, ResNetVariant};
use async_trait::async_trait;
use std::path::Path;
use std::any::Any;
use std::collections::HashMap;

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor as CandleTensor, DType};

#[cfg(feature = "candle")]
use safetensors::SafeTensors;

// Import real model wrappers
pub mod model_wrappers;
#[cfg(feature = "candle")]
use model_wrappers::{BertModelWrapper, ResNetModelWrapper};

/// Candle ML engine implementation
/// 
/// This engine provides a thin wrapper around Candle's SafeTensors loading capability.
/// It does NOT implement specific model architectures - that should be done by the user
/// or by using Candle's built-in model architectures from candle-transformers.
#[derive(Debug)]
pub struct CandleEngine {
    /// Device to use for computations (CPU or CUDA)
    device: Device,
    /// Whether GPU acceleration is available
    gpu_available: bool,
}

impl CandleEngine {
    /// Create a new Candle engine with automatic device detection
    pub fn new() -> Result<Self, InferenceError> {
        #[cfg(feature = "candle")]
        {
            let (device, gpu_available) = Self::detect_device()?;
            Ok(Self {
                device,
                gpu_available,
            })
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(InferenceError::configuration(
                "Candle engine not available - compile with 'candle' feature"
            ))
        }
    }
    
    /// Create a new Candle engine with explicit device selection
    pub fn with_device(device: Device) -> Result<Self, InferenceError> {
        #[cfg(feature = "candle")]
        {
            let gpu_available = matches!(device, Device::Cuda(_));
            Ok(Self {
                device,
                gpu_available,
            })
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(InferenceError::configuration(
                "Candle engine not available - compile with 'candle' feature"
            ))
        }
    }
    
    /// Create a CPU-only Candle engine
    pub fn cpu() -> Result<Self, InferenceError> {
        #[cfg(feature = "candle")]
        {
            Ok(Self {
                device: Device::Cpu,
                gpu_available: false,
            })
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(InferenceError::configuration(
                "Candle engine not available - compile with 'candle' feature"
            ))
        }
    }
    
    /// Create a CUDA-enabled Candle engine
    pub fn cuda(_device_id: usize) -> Result<Self, InferenceError> {
        #[cfg(all(feature = "candle", feature = "candle-cuda"))]
        {
            let device = Device::new_cuda(_device_id)
                .map_err(|e| InferenceError::gpu(format!("Failed to create CUDA device {}: {}", _device_id, e)))?;
            Ok(Self {
                device,
                gpu_available: true,
            })
        }
        #[cfg(all(feature = "candle", not(feature = "candle-cuda")))]
        {
            Err(InferenceError::configuration(
                "CUDA support not available - compile with 'candle-cuda' feature"
            ))
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(InferenceError::configuration(
                "Candle engine not available - compile with 'candle' feature"
            ))
        }
    }
    
    /// Detect the best available device
    #[cfg(feature = "candle")]
    fn detect_device() -> Result<(Device, bool), InferenceError> {
        // Try CUDA first if compiled with CUDA support
        #[cfg(feature = "candle-cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return Ok((device, true));
            }
        }
        
        // Fall back to CPU
        Ok((Device::Cpu, false))
    }
    
    /// Get the current device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Check if GPU acceleration is available
    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }
    
    /// Load a SafeTensors model - provides raw tensor access
    /// 
    /// NOTE: This is a low-level interface that just loads the tensors.
    /// Users need to implement their own model architecture on top of this.
    /// For proper model architectures, users should use candle-transformers
    /// or implement their own model structures using these tensors.
    #[cfg(feature = "candle")]
    async fn load_safetensors(&self, path: &str) -> Result<Box<dyn Model>, InferenceError> {
        println!("🔧 Loading SafeTensors file: {}", path);
        
        // Read the SafeTensors file
        let bytes = tokio::fs::read(path).await
            .map_err(|e| InferenceError::model_load(format!("Failed to read SafeTensors file {}: {}", path, e)))?;
        
        self.load_safetensors_from_bytes(&bytes).await
    }
    
    /// Load a SafeTensors model from bytes
    #[cfg(feature = "candle")]
    async fn load_safetensors_from_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Model>, InferenceError> {
        println!("🔧 Parsing SafeTensors from {} bytes", bytes.len());
        
        // Parse the SafeTensors file
        let safetensors = SafeTensors::deserialize(bytes)
            .map_err(|e| InferenceError::model_load(format!("Failed to parse SafeTensors: {}", e)))?;
        
        // Load all tensors into a HashMap
        let mut tensors = HashMap::new();
        let mut input_specs = Vec::new();
        let mut output_specs = Vec::new();
        
        println!("📊 Found {} tensors in SafeTensors file", safetensors.tensors().len());
        
        for (name, info) in safetensors.tensors() {
            let tensor_data = safetensors.tensor(&name)
                .map_err(|e| InferenceError::model_load(format!("Failed to get tensor {}: {}", name, e)))?;
            
            // Convert safetensors dtype to candle dtype
            let dtype = match info.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I32 => DType::I64, // Map to closest available
                safetensors::Dtype::I64 => DType::I64,
                _ => return Err(InferenceError::unsupported_format(
                    format!("Unsupported SafeTensors dtype: {:?}", info.dtype())
                )),
            };
            
            // Create Candle tensor from raw data
            let shape = info.shape();
            let tensor = CandleTensor::from_raw_buffer(tensor_data.data(), dtype, shape, &self.device)
                .map_err(|e| InferenceError::model_load(format!("Failed to create tensor {}: {}", name, e)))?;
            
            println!("📦 Loaded tensor '{}': shape {:?}, dtype {:?}", name, shape, dtype);
            tensors.insert(name.to_string(), tensor);
            
            // Create tensor specs for inputs/outputs based on naming convention
            let tensor_spec = TensorSpec::new(
                name.to_string(),
                info.shape().iter().map(|&s| Some(s)).collect(),
                match dtype {
                    DType::F32 => DataType::F32,
                    DType::F64 => DataType::F64,
                    DType::I64 => DataType::I64,
                    _ => DataType::F32, // Default fallback
                }
            );
            
            // Simple heuristic: tensors with "input" in name are inputs, others are weights/outputs
            if name.to_lowercase().contains("input") {
                input_specs.push(tensor_spec);
            } else if name.to_lowercase().contains("output") || name.to_lowercase().contains("classifier") {
                output_specs.push(tensor_spec);
            }
        }
        
        // If no explicit input/output specs found, create default ones
        if input_specs.is_empty() {
            input_specs.push(TensorSpec::new(
                "input".to_string(),
                vec![None, None, None, None], // Dynamic shape for images
                DataType::F32,
            ));
        }
        
        if output_specs.is_empty() {
            output_specs.push(TensorSpec::new(
                "output".to_string(),
                vec![None, None], // Dynamic shape for classification
                DataType::F32,
            ));
        }
        
        println!("✅ Successfully loaded SafeTensors model with {} tensors", tensors.len());
        println!("📋 Input specs: {} tensors", input_specs.len());
        println!("📋 Output specs: {} tensors", output_specs.len());
        
        let model = GenericSafeTensorsModel::new(tensors, input_specs, output_specs, self.device.clone())?;
        Ok(Box::new(model))
    }
    
    /// Check if bytes represent SafeTensors format
    fn is_safetensors_format(&self, bytes: &[u8]) -> bool {
        // SafeTensors files start with an 8-byte header containing the JSON length
        if bytes.len() < 8 {
            return false;
        }
        
        // Try to parse the header
        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;
        
        // Check if we have enough bytes for header + JSON
        if bytes.len() < 8 + header_len {
            return false;
        }
        
        // Try to parse the JSON metadata
        let json_bytes = &bytes[8..8 + header_len];
        serde_json::from_slice::<serde_json::Value>(json_bytes).is_ok()
    }
    
    /// Load a PyTorch model (.pt, .pth)
    #[cfg(feature = "candle")]
    async fn load_pytorch(&self, _path: &str) -> Result<Box<dyn Model>, InferenceError> {
        // PyTorch loading would require additional dependencies
        Err(InferenceError::unsupported_format(
            "PyTorch model loading not yet implemented - use SafeTensors format instead"
        ))
    }
    
    /// Load a model with specific architecture from HuggingFace
    #[cfg(feature = "candle")]
    pub async fn load_from_huggingface(&self, config: &ModelConfig) -> Result<Box<dyn Model>, InferenceError> {
        let repo_id = config.repo_id.as_ref()
            .map(|s| s.as_str())
            .or_else(|| config.default_repo_id())
            .ok_or_else(|| InferenceError::model_load("Repository ID required for HuggingFace loading".to_string()))?;
        
        let filename = config.filename.as_ref()
            .map(|s| s.as_str())
            .or_else(|| Some(config.default_filename()));
        
        match &config.architecture {
            ModelArchitecture::Bert => {
                let model = BertModelWrapper::load_from_huggingface(&self.device, repo_id, filename).await?;
                Ok(Box::new(model))
            }
            ModelArchitecture::ResNet { variant } => {
                // Load ResNet from HuggingFace with real model downloading
                let model = ResNetModelWrapper::load_from_huggingface(&self.device, repo_id, filename, variant.clone()).await?;
                Ok(Box::new(model))
            }
            _ => Err(InferenceError::unsupported_format(
                format!("Model architecture {:?} not yet supported for HuggingFace loading", config.architecture)
            )),
        }
    }
    
    /// Load a BERT model from HuggingFace (convenience method)
    #[cfg(feature = "candle")]
    pub async fn load_bert(&self, repo_id: &str) -> Result<Box<dyn Model>, InferenceError> {
        let config = ModelConfig::new(ModelArchitecture::Bert)
            .with_repo_id(repo_id);
        self.load_from_huggingface(&config).await
    }
    
    /// Load a ResNet model (convenience method)
    #[cfg(feature = "candle")]
    pub async fn load_resnet(&self, variant: ResNetVariant) -> Result<Box<dyn Model>, InferenceError> {
        let config = ModelConfig::new(ModelArchitecture::ResNet { variant });
        self.load_from_huggingface(&config).await
    }
}

impl Default for CandleEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default Candle engine")
    }
}

#[async_trait]
impl InferenceEngine for CandleEngine {
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError> {
        let path_obj = Path::new(path);
        
        match path_obj.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => self.load_safetensors(path).await,
            Some("pt") | Some("pth") => self.load_pytorch(path).await,
            _ => Err(InferenceError::unsupported_format(
                format!("Unsupported file extension for path: {}", path)
            )),
        }
    }
    
    async fn load_from_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Model>, InferenceError> {
        if self.is_safetensors_format(bytes) {
            self.load_safetensors_from_bytes(bytes).await
        } else {
            Err(InferenceError::unsupported_format(
                "Unsupported model format in bytes - only SafeTensors supported"
            ))
        }
    }
    
    fn supports_format(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::SafeTensors | ModelFormat::PyTorch)
    }
    
    fn engine_name(&self) -> &'static str {
        "candle"
    }
    
    fn engine_type(&self) -> EngineType {
        EngineType::Candle
    }
}

/// Generic SafeTensors model that provides raw tensor access
/// 
/// This is a low-level model implementation that just holds the loaded tensors.
/// It does NOT implement any specific model architecture or inference logic.
/// Users should implement their own inference logic or use this as a base
/// for more sophisticated model implementations.
pub struct GenericSafeTensorsModel {
    /// Raw tensors loaded from SafeTensors file
    #[cfg(feature = "candle")]
    tensors: HashMap<String, CandleTensor>,
    /// Device for computations
    device: Device,
    /// Input specifications
    input_specs: Vec<TensorSpec>,
    /// Output specifications
    output_specs: Vec<TensorSpec>,
}

impl std::fmt::Debug for GenericSafeTensorsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericSafeTensorsModel")
            .field("device", &self.device)
            .field("input_specs", &self.input_specs)
            .field("output_specs", &self.output_specs)
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl GenericSafeTensorsModel {
    #[cfg(feature = "candle")]
    pub fn new(
        tensors: HashMap<String, CandleTensor>,
        input_specs: Vec<TensorSpec>,
        output_specs: Vec<TensorSpec>,
        device: Device,
    ) -> Result<Self, InferenceError> {
        Ok(Self {
            tensors,
            device,
            input_specs,
            output_specs,
        })
    }
    
    /// Get access to the raw tensors (for advanced users)
    #[cfg(feature = "candle")]
    pub fn get_tensor(&self, name: &str) -> Option<&CandleTensor> {
        self.tensors.get(name)
    }
    
    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
    
    /// Convert our Tensor to Candle Tensor
    #[cfg(feature = "candle")]
    fn tensor_to_candle(&self, tensor: &Tensor) -> Result<CandleTensor, InferenceError> {
        let dtype = match tensor.data_type() {
            DataType::F32 => DType::F32,
            DataType::F64 => DType::F64,
            DataType::I32 => DType::I64,
            DataType::I64 => DType::I64,
            _ => return Err(InferenceError::prediction("Unsupported tensor data type")),
        };
        
        let shape: Vec<usize> = tensor.shape().iter().map(|&dim| dim as usize).collect();
        
        // Get the raw data as bytes
        let data_bytes = tensor.data();
        
        CandleTensor::from_raw_buffer(data_bytes, dtype, &shape, &self.device)
            .map_err(|e| InferenceError::prediction(format!("Failed to create Candle tensor: {}", e)))
    }
    
    /// Convert Candle Tensor to our Tensor
    #[cfg(feature = "candle")]
    fn candle_to_tensor(&self, candle_tensor: &CandleTensor) -> Result<Tensor, InferenceError> {
        let shape: Vec<i64> = candle_tensor.shape().dims().iter().map(|&dim| dim as i64).collect();
        
        let data_type = match candle_tensor.dtype() {
            DType::F32 => DataType::F32,
            DType::F64 => DataType::F64,
            DType::I64 => DataType::I64,
            _ => return Err(InferenceError::prediction("Unsupported Candle tensor data type")),
        };
        
        // Extract data from Candle tensor using to_vec1
        // This is a simplified approach - real implementations might need more sophisticated conversion
        let data: Vec<f32> = match data_type {
            DataType::F32 => {
                candle_tensor.to_vec1()
                    .map_err(|e| InferenceError::prediction(format!("Failed to extract F32 data: {}", e)))?
            }
            DataType::F64 => {
                let f64_data: Vec<f64> = candle_tensor.to_vec1()
                    .map_err(|e| InferenceError::prediction(format!("Failed to extract F64 data: {}", e)))?;
                f64_data.into_iter().map(|x| x as f32).collect()
            }
            DataType::I64 => {
                let i64_data: Vec<i64> = candle_tensor.to_vec1()
                    .map_err(|e| InferenceError::prediction(format!("Failed to extract I64 data: {}", e)))?;
                i64_data.into_iter().map(|x| x as f32).collect()
            }
            _ => return Err(InferenceError::prediction("Unsupported data type conversion")),
        };
        
        // Convert f32 data to bytes for our tensor format
        let bytes: Vec<u8> = data.iter()
            .flat_map(|&x| x.to_le_bytes().to_vec())
            .collect();
        
        Tensor::new(bytes, shape.iter().map(|&x| x as usize).collect(), data_type)
            .map_err(|e| InferenceError::prediction(format!("Failed to convert Candle tensor: {}", e)))
    }
    
    /// Perform inference using the loaded SafeTensors model
    /// 
    /// This performs a basic linear transformation using the loaded weights.
    /// For complex architectures, users should use the specific model wrappers.
    #[cfg(feature = "candle")]
    fn run_inference(&self, input: &CandleTensor) -> Result<CandleTensor, InferenceError> {
        println!("🔧 Running SafeTensors inference with {} tensors", self.tensors.len());
        println!("📊 Input shape: {:?}", input.shape());
        
        // Try to find a weight tensor that matches the input dimensions
        // This is a generic approach for simple linear models
        let input_dims = input.shape().dims();
        let last_dim = input_dims[input_dims.len() - 1];
        
        // Look for a weight tensor that can be multiplied with the input
        for (name, tensor) in &self.tensors {
            let tensor_dims = tensor.shape().dims();
            
            // Check if this tensor can be used for matrix multiplication
            if tensor_dims.len() == 2 && tensor_dims[0] == last_dim {
                println!("🎯 Using tensor '{}' for inference: {:?}", name, tensor_dims);
                
                // Perform matrix multiplication: input @ weight
                let output = input.matmul(tensor)
                    .map_err(|e| InferenceError::prediction(format!("Matrix multiplication failed: {}", e)))?;
                
                println!("✅ Inference complete, output shape: {:?}", output.shape());
                return Ok(output);
            }
        }
        
        // If no suitable weight tensor found, return error
        Err(InferenceError::prediction(
            "No suitable weight tensor found for inference. This SafeTensors model may require a specific architecture wrapper."
        ))
    }
}

#[async_trait]
impl Model for GenericSafeTensorsModel {
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        #[cfg(feature = "candle")]
        {
            // Convert input to Candle tensor
            let candle_input = self.tensor_to_candle(input)?;
            
            // Run placeholder inference
            let output = self.run_inference(&candle_input)?;
            
            // Convert output back to our tensor format
            self.candle_to_tensor(&output)
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(InferenceError::configuration("Candle not available"))
        }
    }
    
    async fn predict_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.predict(input).await?);
        }
        Ok(results)
    }
    
    fn input_specs(&self) -> &[TensorSpec] {
        &self.input_specs
    }
    
    fn output_specs(&self) -> &[TensorSpec] {
        &self.output_specs
    }
    
    fn engine_type(&self) -> EngineType {
        EngineType::Candle
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// HuggingFace integration utilities
/// These functions help with loading models from HuggingFace Hub
pub struct HuggingFaceIntegration;

impl HuggingFaceIntegration {
    /// Download a model from HuggingFace Hub
    /// This is a placeholder - would need actual HF Hub integration
    pub async fn download_model(_repo_id: &str, _filename: &str) -> Result<Vec<u8>, InferenceError> {
        Err(InferenceError::unsupported_format(
            "HuggingFace Hub integration not yet implemented"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DataType;

    #[test]
    fn test_candle_engine_creation() {
        let engine = CandleEngine::new();
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert_eq!(engine.engine_name(), "candle");
        assert_eq!(engine.engine_type(), EngineType::Candle);
    }

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_safetensors_format_detection() {
        let engine = CandleEngine::new().unwrap();
        
        // Test with invalid data
        let invalid_data = vec![0u8; 10];
        assert!(!engine.is_safetensors_format(&invalid_data));
        
        // Test with too short data
        let short_data = vec![0u8; 5];
        assert!(!engine.is_safetensors_format(&short_data));
    }
} 