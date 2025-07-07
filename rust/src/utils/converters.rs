use crate::models::{InferenceError, Tensor};

/// Conversion utilities for tensors between different formats
pub struct TensorConverter;

impl TensorConverter {
    /// Convert Vec<Vec<f32>> to Tensor
    pub fn from_2d_f32(data: Vec<Vec<f32>>) -> Result<Tensor, InferenceError> {
        if data.is_empty() {
            return Err(InferenceError::invalid_tensor_data("Empty 2D array"));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Validate all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(InferenceError::invalid_tensor_data(format!(
                    "Row {} has length {} but expected {}", i, row.len(), cols
                )));
            }
        }
        
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        Tensor::from_f32(flat_data, vec![rows, cols])
    }
    
    /// Convert Vec<Vec<Vec<f32>>> to Tensor
    pub fn from_3d_f32(data: Vec<Vec<Vec<f32>>>) -> Result<Tensor, InferenceError> {
        if data.is_empty() {
            return Err(InferenceError::invalid_tensor_data("Empty 3D array"));
        }
        
        let dim0 = data.len();
        let dim1 = data[0].len();
        let dim2 = if dim1 > 0 { data[0][0].len() } else { 0 };
        
        // Validate dimensions
        for (i, matrix) in data.iter().enumerate() {
            if matrix.len() != dim1 {
                return Err(InferenceError::invalid_tensor_data(format!(
                    "Matrix {} has {} rows but expected {}", i, matrix.len(), dim1
                )));
            }
            for (j, row) in matrix.iter().enumerate() {
                if row.len() != dim2 {
                    return Err(InferenceError::invalid_tensor_data(format!(
                        "Matrix {} row {} has length {} but expected {}", i, j, row.len(), dim2
                    )));
                }
            }
        }
        
        let flat_data: Vec<f32> = data.into_iter()
            .flatten()
            .flatten()
            .collect();
        
        Tensor::from_f32(flat_data, vec![dim0, dim1, dim2])
    }
    
    /// Convert Tensor to Vec<Vec<f32>>
    pub fn to_2d_f32(tensor: &Tensor) -> Result<Vec<Vec<f32>>, InferenceError> {
        if tensor.ndim() != 2 {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Expected 2D tensor, got {}D", tensor.ndim()
            )));
        }
        
        let shape = tensor.shape();
        let rows = shape[0];
        let cols = shape[1];
        let data = tensor.to_f32_vec()?;
        
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(data[start..end].to_vec());
        }
        
        Ok(result)
    }
    
    /// Convert Tensor to Vec<Vec<Vec<f32>>>
    pub fn to_3d_f32(tensor: &Tensor) -> Result<Vec<Vec<Vec<f32>>>, InferenceError> {
        if tensor.ndim() != 3 {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Expected 3D tensor, got {}D", tensor.ndim()
            )));
        }
        
        let shape = tensor.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];
        let data = tensor.to_f32_vec()?;
        
        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut matrix = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let start = (i * dim1 + j) * dim2;
                let end = start + dim2;
                matrix.push(data[start..end].to_vec());
            }
            result.push(matrix);
        }
        
        Ok(result)
    }
    
    /// Convert image data (HWC format) to tensor
    pub fn from_image_hwc(
        data: Vec<u8>,
        height: usize,
        width: usize,
        channels: usize
    ) -> Result<Tensor, InferenceError> {
        if data.len() != height * width * channels {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Image data size {} doesn't match dimensions {}x{}x{}",
                data.len(), height, width, channels
            )));
        }
        
        // Convert u8 to f32 and normalize to [0, 1]
        let normalized_data: Vec<f32> = data.into_iter()
            .map(|byte| byte as f32 / 255.0)
            .collect();
        
        Tensor::from_f32(normalized_data, vec![height, width, channels])
    }
    
    /// Convert image data (CHW format) to tensor
    pub fn from_image_chw(
        data: Vec<u8>,
        channels: usize,
        height: usize,
        width: usize
    ) -> Result<Tensor, InferenceError> {
        if data.len() != channels * height * width {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Image data size {} doesn't match dimensions {}x{}x{}",
                data.len(), channels, height, width
            )));
        }
        
        // Convert u8 to f32 and normalize to [0, 1]
        let normalized_data: Vec<f32> = data.into_iter()
            .map(|byte| byte as f32 / 255.0)
            .collect();
        
        Tensor::from_f32(normalized_data, vec![channels, height, width])
    }
    
    /// Convert HWC to CHW format
    pub fn hwc_to_chw(tensor: &Tensor) -> Result<Tensor, InferenceError> {
        if tensor.ndim() != 3 {
            return Err(InferenceError::invalid_tensor_data(
                "Expected 3D tensor for HWC to CHW conversion"
            ));
        }
        
        let shape = tensor.shape();
        let height = shape[0];
        let width = shape[1];
        let channels = shape[2];
        
        let hwc_data = tensor.to_f32_vec()?;
        let mut chw_data = vec![0.0; hwc_data.len()];
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let hwc_idx = h * width * channels + w * channels + c;
                    let chw_idx = c * height * width + h * width + w;
                    chw_data[chw_idx] = hwc_data[hwc_idx];
                }
            }
        }
        
        Tensor::from_f32(chw_data, vec![channels, height, width])
    }
    
    /// Convert CHW to HWC format
    pub fn chw_to_hwc(tensor: &Tensor) -> Result<Tensor, InferenceError> {
        if tensor.ndim() != 3 {
            return Err(InferenceError::invalid_tensor_data(
                "Expected 3D tensor for CHW to HWC conversion"
            ));
        }
        
        let shape = tensor.shape();
        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];
        
        let chw_data = tensor.to_f32_vec()?;
        let mut hwc_data = vec![0.0; chw_data.len()];
        
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let chw_idx = c * height * width + h * width + w;
                    let hwc_idx = h * width * channels + w * channels + c;
                    hwc_data[hwc_idx] = chw_data[chw_idx];
                }
            }
        }
        
        Tensor::from_f32(hwc_data, vec![height, width, channels])
    }
    
    /// Normalize tensor values to [0, 1] range
    pub fn normalize_0_1(tensor: &Tensor) -> Result<Tensor, InferenceError> {
        let data = tensor.to_f32_vec()?;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < f32::EPSILON {
            // All values are the same, return zeros
            let normalized = vec![0.0; data.len()];
            return Tensor::from_f32(normalized, tensor.shape().to_vec());
        }
        
        let normalized: Vec<f32> = data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect();
        
        Tensor::from_f32(normalized, tensor.shape().to_vec())
    }
    
    /// Standardize tensor values (zero mean, unit variance)
    pub fn standardize(tensor: &Tensor) -> Result<Tensor, InferenceError> {
        let data = tensor.to_f32_vec()?;
        let len = data.len() as f32;
        
        // Calculate mean
        let mean = data.iter().sum::<f32>() / len;
        
        // Calculate standard deviation
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / len;
        let std_dev = variance.sqrt();
        
        if std_dev < f32::EPSILON {
            // All values are the same, return zeros
            let standardized = vec![0.0; data.len()];
            return Tensor::from_f32(standardized, tensor.shape().to_vec());
        }
        
        let standardized: Vec<f32> = data.iter()
            .map(|&x| (x - mean) / std_dev)
            .collect();
        
        Tensor::from_f32(standardized, tensor.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_from_2d_f32() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let tensor = TensorConverter::from_2d_f32(data).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    
    #[test]
    fn test_to_2d_f32() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = TensorConverter::to_2d_f32(&tensor).unwrap();
        assert_eq!(result, vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
    }
    
    #[test]
    fn test_hwc_to_chw() {
        // 2x2x3 tensor (HWC format)
        let tensor = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 2, 3]
        ).unwrap();
        
        let chw_tensor = TensorConverter::hwc_to_chw(&tensor).unwrap();
        assert_eq!(chw_tensor.shape(), &[3, 2, 2]);
        
        // Convert back to verify
        let hwc_tensor = TensorConverter::chw_to_hwc(&chw_tensor).unwrap();
        assert_eq!(hwc_tensor.to_f32_vec().unwrap(), tensor.to_f32_vec().unwrap());
    }
    
    #[test]
    fn test_normalize_0_1() {
        let tensor = Tensor::from_f32(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let normalized = TensorConverter::normalize_0_1(&tensor).unwrap();
        let result = normalized.to_f32_vec().unwrap();
        
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_standardize() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let standardized = TensorConverter::standardize(&tensor).unwrap();
        let result = standardized.to_f32_vec().unwrap();
        
        // Mean should be approximately 0
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-6);
        
        // Standard deviation should be approximately 1
        let variance = result.iter().map(|&x| x.powi(2)).sum::<f32>() / result.len() as f32;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 1e-6);
    }
} 