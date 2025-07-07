use crate::models::{InferenceError, Tensor};

/// Data preprocessing utilities for different input types
pub struct Preprocessor;

impl Preprocessor {
    /// Preprocess image data for ML models
    pub fn preprocess_image(
        data: &[u8],
        width: usize,
        height: usize,
        channels: usize,
        config: &ImagePreprocessConfig,
    ) -> Result<Tensor, InferenceError> {
        // Validate input dimensions
        let expected_size = width * height * channels;
        if data.len() != expected_size {
            return Err(InferenceError::invalid_tensor_data(format!(
                "Image data size {} doesn't match dimensions {}x{}x{}",
                data.len(), width, height, channels
            )));
        }
        
        // Convert to f32 and apply preprocessing
        let mut processed_data: Vec<f32> = data.iter()
            .map(|&byte| byte as f32)
            .collect();
        
        // Apply normalization
        if let Some(norm) = &config.normalization {
            match norm {
                Normalization::ZeroToOne => {
                    for pixel in processed_data.iter_mut() {
                        *pixel /= 255.0;
                    }
                }
                Normalization::MinusOneToOne => {
                    for pixel in processed_data.iter_mut() {
                        *pixel = (*pixel / 255.0) * 2.0 - 1.0;
                    }
                }
                Normalization::ImageNet => {
                    // Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    let means = [0.485, 0.456, 0.406];
                    let stds = [0.229, 0.224, 0.225];
                    
                    for (i, pixel) in processed_data.iter_mut().enumerate() {
                        let channel = i % channels;
                        *pixel = (*pixel / 255.0 - means[channel]) / stds[channel];
                    }
                }
                Normalization::Custom { mean, std } => {
                    for (i, pixel) in processed_data.iter_mut().enumerate() {
                        let channel = i % channels;
                        *pixel = (*pixel / 255.0 - mean[channel]) / std[channel];
                    }
                }
            }
        }
        
        // Apply format conversion if needed
        let final_shape = match config.format {
            ImageFormat::HWC => vec![height, width, channels],
            ImageFormat::CHW => {
                // Convert HWC to CHW
                let mut chw_data = vec![0.0f32; processed_data.len()];
                for h in 0..height {
                    for w in 0..width {
                        for c in 0..channels {
                            let hwc_idx = h * width * channels + w * channels + c;
                            let chw_idx = c * height * width + h * width + w;
                            chw_data[chw_idx] = processed_data[hwc_idx];
                        }
                    }
                }
                processed_data = chw_data;
                vec![channels, height, width]
            }
        };
        
        Tensor::from_f32(processed_data, final_shape)
    }
    
    /// Preprocess text data for ML models
    pub fn preprocess_text(
        text: &str,
        config: &TextPreprocessConfig,
    ) -> Result<Tensor, InferenceError> {
        let mut processed_text = text.to_string();
        
        // Apply text transformations
        if config.lowercase {
            processed_text = processed_text.to_lowercase();
        }
        
        if config.remove_punctuation {
            processed_text = processed_text
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }
        
        // Tokenization (basic whitespace tokenization)
        let tokens: Vec<&str> = processed_text.split_whitespace().collect();
        
        // Convert to token IDs (this is a simplified version)
        let token_ids: Vec<f32> = if let Some(vocab) = &config.vocabulary {
            tokens.iter()
                .map(|token| vocab.get(*token).copied().unwrap_or(config.unknown_token_id) as f32)
                .collect()
        } else {
            // Simple hash-based tokenization for demo
            tokens.iter()
                .map(|token| (token.len() % 1000) as f32)
                .collect()
        };
        
        // Apply padding or truncation
        let mut final_tokens = token_ids;
        if let Some(max_length) = config.max_length {
            if final_tokens.len() > max_length {
                final_tokens.truncate(max_length);
            } else {
                final_tokens.resize(max_length, config.padding_token_id as f32);
            }
        }
        
        let length = final_tokens.len();
        Tensor::from_f32(final_tokens, vec![length])
    }
    
    /// Preprocess audio data for ML models
    pub fn preprocess_audio(
        samples: &[f32],
        sample_rate: u32,
        config: &AudioPreprocessConfig,
    ) -> Result<Tensor, InferenceError> {
        let mut processed_samples = samples.to_vec();
        
        // Apply normalization
        if config.normalize {
            let max_amplitude = processed_samples.iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);
            
            if max_amplitude > 0.0 {
                for sample in processed_samples.iter_mut() {
                    *sample /= max_amplitude;
                }
            }
        }
        
        // Apply resampling if needed
        if let Some(target_rate) = config.target_sample_rate {
            if target_rate != sample_rate {
                processed_samples = resample_audio(&processed_samples, sample_rate, target_rate)?;
            }
        }
        
        // Apply windowing if needed
        if let Some(window_size) = config.window_size {
            let hop_size = config.hop_size.unwrap_or(window_size / 2);
            processed_samples = apply_windowing(&processed_samples, window_size, hop_size);
        }
        
        let length = processed_samples.len();
        Tensor::from_f32(processed_samples, vec![length])
    }
}

/// Image preprocessing configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    pub normalization: Option<Normalization>,
    pub format: ImageFormat,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            normalization: Some(Normalization::ZeroToOne),
            format: ImageFormat::HWC,
        }
    }
}

/// Normalization strategies for images
#[derive(Debug, Clone)]
pub enum Normalization {
    /// Normalize to [0, 1] range
    ZeroToOne,
    /// Normalize to [-1, 1] range
    MinusOneToOne,
    /// ImageNet normalization
    ImageNet,
    /// Custom normalization with mean and std per channel
    Custom { mean: Vec<f32>, std: Vec<f32> },
}

/// Image format options
#[derive(Debug, Clone)]
pub enum ImageFormat {
    /// Height x Width x Channels
    HWC,
    /// Channels x Height x Width
    CHW,
}

/// Text preprocessing configuration
#[derive(Debug, Clone)]
pub struct TextPreprocessConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub max_length: Option<usize>,
    pub vocabulary: Option<std::collections::HashMap<String, u32>>,
    pub unknown_token_id: u32,
    pub padding_token_id: u32,
}

impl Default for TextPreprocessConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            max_length: None,
            vocabulary: None,
            unknown_token_id: 0,
            padding_token_id: 0,
        }
    }
}

/// Audio preprocessing configuration
#[derive(Debug, Clone)]
pub struct AudioPreprocessConfig {
    pub normalize: bool,
    pub target_sample_rate: Option<u32>,
    pub window_size: Option<usize>,
    pub hop_size: Option<usize>,
}

impl Default for AudioPreprocessConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            target_sample_rate: None,
            window_size: None,
            hop_size: None,
        }
    }
}

/// Simple audio resampling (linear interpolation)
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, InferenceError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    let ratio = to_rate as f64 / from_rate as f64;
    let new_length = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_length);
    
    for i in 0..new_length {
        let src_index = i as f64 / ratio;
        let src_index_floor = src_index.floor() as usize;
        let src_index_ceil = (src_index.ceil() as usize).min(samples.len() - 1);
        
        if src_index_floor == src_index_ceil {
            resampled.push(samples[src_index_floor]);
        } else {
            let frac = src_index - src_index_floor as f64;
            let interpolated = samples[src_index_floor] * (1.0 - frac) as f32 +
                             samples[src_index_ceil] * frac as f32;
            resampled.push(interpolated);
        }
    }
    
    Ok(resampled)
}

/// Apply windowing to audio samples
fn apply_windowing(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<f32> {
    let mut windowed = Vec::new();
    let mut pos = 0;
    
    while pos + window_size <= samples.len() {
        windowed.extend_from_slice(&samples[pos..pos + window_size]);
        pos += hop_size;
    }
    
    windowed
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_image_preprocessing() {
        let data = vec![128u8; 3 * 2 * 2]; // 2x2 RGB image
        let config = ImagePreprocessConfig::default();
        
        let result = Preprocessor::preprocess_image(&data, 2, 2, 3, &config);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 3]);
        
        // Check normalization (128/255 ≈ 0.502)
        let values = tensor.to_f32_vec().unwrap();
        assert!((values[0] - 0.502).abs() < 0.01);
    }
    
    #[test]
    fn test_text_preprocessing() {
        let config = TextPreprocessConfig::default();
        let result = Preprocessor::preprocess_text("Hello World!", &config);
        
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().len(), 1);
    }
    
    #[test]
    fn test_audio_preprocessing() {
        let samples = vec![0.5, -0.5, 0.3, -0.3];
        let config = AudioPreprocessConfig::default();
        
        let result = Preprocessor::preprocess_audio(&samples, 44100, &config);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[4]);
        
        // Check normalization (max amplitude should be 1.0)
        let values = tensor.to_f32_vec().unwrap();
        let max_val = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.01);
    }
} 