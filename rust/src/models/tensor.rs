use crate::models::InferenceError;
use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Supported tensor data types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
    U32,
    Bool,
}

impl DataType {
    /// Get the size in bytes of this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::I32 => 4,
            DataType::I64 => 8,
            DataType::U8 => 1,
            DataType::U32 => 4,
            DataType::Bool => 1,
        }
    }
    
    /// Get the name of this data type
    pub fn name(&self) -> &'static str {
        match self {
            DataType::F32 => "float32",
            DataType::F64 => "float64",
            DataType::I32 => "int32",
            DataType::I64 => "int64",
            DataType::U8 => "uint8",
            DataType::U32 => "uint32",
            DataType::Bool => "bool",
        }
    }
}

/// Unified tensor representation for all engines
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data as bytes
    data: Vec<u8>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    data_type: DataType,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(data: Vec<u8>, shape: Vec<usize>, data_type: DataType) -> Result<Self, InferenceError> {
        let expected_size = shape.iter().product::<usize>() * data_type.size_bytes();
        if data.len() != expected_size {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Data size {} doesn't match expected size {} for shape {:?} and type {:?}",
                data.len(), expected_size, shape, data_type
            )));
        }
        
        Ok(Self {
            data,
            shape,
            data_type,
        })
    }
    
    /// Create tensor from f32 data
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, InferenceError> {
        let bytes = data.into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        Self::new(bytes, shape, DataType::F32)
    }
    
    /// Create tensor from f64 data
    pub fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, InferenceError> {
        let bytes = data.into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        Self::new(bytes, shape, DataType::F64)
    }
    
    /// Create tensor from i32 data
    pub fn from_i32(data: Vec<i32>, shape: Vec<usize>) -> Result<Self, InferenceError> {
        let bytes = data.into_iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        Self::new(bytes, shape, DataType::I32)
    }
    
    /// Create tensor from i64 data
    pub fn from_i64(data: Vec<i64>, shape: Vec<usize>) -> Result<Self, InferenceError> {
        let bytes = data.into_iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        Self::new(bytes, shape, DataType::I64)
    }
    
    /// Create tensor from ndarray
    pub fn from_ndarray_f32(array: ArrayD<f32>) -> Result<Self, InferenceError> {
        let shape = array.shape().to_vec();
        let data = array.into_raw_vec();
        Self::from_f32(data, shape)
    }
    
    /// Create tensor from ndarray
    pub fn from_ndarray_f64(array: ArrayD<f64>) -> Result<Self, InferenceError> {
        let shape = array.shape().to_vec();
        let data = array.into_raw_vec();
        Self::from_f64(data, shape)
    }
    
    /// Get raw data as bytes
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get data type
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
    
    /// Get number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Convert to f32 vector (if compatible)
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, InferenceError> {
        match self.data_type {
            DataType::F32 => {
                let mut result = Vec::with_capacity(self.len());
                for chunk in self.data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            _ => Err(InferenceError::invalid_tensor_data(format!(
                "Cannot convert {:?} to f32 vector", self.data_type
            )))
        }
    }
    
    /// Convert to f64 vector (if compatible)
    pub fn to_f64_vec(&self) -> Result<Vec<f64>, InferenceError> {
        match self.data_type {
            DataType::F64 => {
                let mut result = Vec::with_capacity(self.len());
                for chunk in self.data.chunks_exact(8) {
                    let bytes = [
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ];
                    result.push(f64::from_le_bytes(bytes));
                }
                Ok(result)
            }
            DataType::F32 => {
                // Convert f32 to f64
                let f32_data = self.to_f32_vec()?;
                Ok(f32_data.into_iter().map(|f| f as f64).collect())
            }
            _ => Err(InferenceError::invalid_tensor_data(format!(
                "Cannot convert {:?} to f64 vector", self.data_type
            )))
        }
    }
    
    /// Convert to ndarray
    pub fn to_ndarray_f32(&self) -> Result<ArrayD<f32>, InferenceError> {
        let data = self.to_f32_vec()?;
        Array::from_shape_vec(IxDyn(&self.shape), data)
            .map_err(|e| InferenceError::invalid_tensor_data(format!("Failed to create ndarray: {}", e)))
    }
    
    /// Convert to ndarray
    pub fn to_ndarray_f64(&self) -> Result<ArrayD<f64>, InferenceError> {
        let data = self.to_f64_vec()?;
        Array::from_shape_vec(IxDyn(&self.shape), data)
            .map_err(|e| InferenceError::invalid_tensor_data(format!("Failed to create ndarray: {}", e)))
    }
    
    /// Reshape tensor (creates new tensor with same data)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, InferenceError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(InferenceError::invalid_shape(vec![self.len()], vec![new_len]));
        }
        
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            data_type: self.data_type.clone(),
        })
    }
    
    /// Get a scalar value (for single-element tensors)
    pub fn scalar_f32(&self) -> Result<f32, InferenceError> {
        if self.len() != 1 {
            return Err(InferenceError::invalid_tensor_data(
                "Tensor must have exactly one element to get scalar value".to_string()
            ));
        }
        let vec = self.to_f32_vec()?;
        Ok(vec[0])
    }
    
    /// Get a scalar value (for single-element tensors)
    pub fn scalar_f64(&self) -> Result<f64, InferenceError> {
        if self.len() != 1 {
            return Err(InferenceError::invalid_tensor_data(
                "Tensor must have exactly one element to get scalar value".to_string()
            ));
        }
        let vec = self.to_f64_vec()?;
        Ok(vec[0])
    }
    
    /// Get data as f32 slice (if compatible)
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.data_type != DataType::F32 || self.data.len() % 4 != 0 {
            return None;
        }
        // SAFETY: We've verified the data type and length alignment
        unsafe {
            Some(std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.data.len() / 4,
            ))
        }
    }
    
    /// Get data as f64 slice (if compatible)
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.data_type != DataType::F64 || self.data.len() % 8 != 0 {
            return None;
        }
        // SAFETY: We've verified the data type and length alignment
        unsafe {
            Some(std::slice::from_raw_parts(
                self.data.as_ptr() as *const f64,
                self.data.len() / 8,
            ))
        }
    }
    
    /// Get data as i32 slice (if compatible)
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.data_type != DataType::I32 || self.data.len() % 4 != 0 {
            return None;
        }
        // SAFETY: We've verified the data type and length alignment
        unsafe {
            Some(std::slice::from_raw_parts(
                self.data.as_ptr() as *const i32,
                self.data.len() / 4,
            ))
        }
    }
    
    /// Get data as i64 slice (if compatible)
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if self.data_type != DataType::I64 || self.data.len() % 8 != 0 {
            return None;
        }
        // SAFETY: We've verified the data type and length alignment
        unsafe {
            Some(std::slice::from_raw_parts(
                self.data.as_ptr() as *const i64,
                self.data.len() / 8,
            ))
        }
    }
}

/// Tensor specification for model inputs/outputs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Expected shape (None for dynamic dimensions)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub data_type: DataType,
    /// Optional description
    pub description: Option<String>,
}

impl TensorSpec {
    /// Create a new tensor specification
    pub fn new(name: String, shape: Vec<Option<usize>>, data_type: DataType) -> Self {
        Self {
            name,
            shape,
            data_type,
            description: None,
        }
    }
    
    /// Create with description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
    
    /// Check if a tensor matches this specification
    pub fn matches(&self, tensor: &Tensor) -> bool {
        // Check data type
        if tensor.data_type() != &self.data_type {
            return false;
        }
        
        // Check shape compatibility
        if tensor.shape().len() != self.shape.len() {
            return false;
        }
        
        for (actual, expected) in tensor.shape().iter().zip(self.shape.iter()) {
            if let Some(expected_size) = expected {
                if actual != expected_size {
                    return false;
                }
            }
            // None means dynamic dimension, so any size is acceptable
        }
        
        true
    }
    
    /// Get the fixed size if all dimensions are specified
    pub fn fixed_size(&self) -> Option<Vec<usize>> {
        let mut result = Vec::new();
        for dim in &self.shape {
            match dim {
                Some(size) => result.push(*size),
                None => return None,
            }
        }
        Some(result)
    }
}

/// Type alias for TensorSpec to match BRD naming convention
pub type TensorInfo = TensorSpec; 