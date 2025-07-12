# Inference Library V2: New Implementation Plan

## Executive Summary

This document outlines a complete rebuild of the Inference Flutter package to properly fulfill its mission: **exposing the full power of Rust ML ecosystem to Flutter developers**. Based on research into Candle and Linfa usage patterns, this plan creates a true bridge library rather than a custom ML framework.

## Core Architecture Principles

### 1. Bridge, Not Framework
- **Thin wrappers** around existing Rust ML libraries
- **Direct exposure** of library capabilities
- **Minimal abstraction** layers
- **Zero custom inference logic**

### 2. Real ML Capabilities
- **Actual model architectures** from candle-transformers
- **Production-ready algorithms** from Linfa
- **No placeholder implementations**

### 3. True Zero-Setup Experience
- **One command install**: `flutter pub add inference`
- **Three lines of code**: load → predict → result
- **Automatic engine detection** based on file format
- **Built-in model downloading** from HuggingFace Hub

## Research-Based Implementation

### Candle Usage Patterns (From Research)

Based on real-world Candle usage, the library should expose:

#### 1. Model Architecture Support
```rust
// Real Candle model architectures
pub enum ModelArchitecture {
    Bert,           // BERT for NLP tasks
    ResNet,         // ResNet for image classification
    MobileNet,      // MobileNet for mobile deployment
    Mistral,        // Mistral for text generation
    Llama,          // Llama for text generation
    Whisper,        // Whisper for speech recognition
}

impl CandleEngine {
    pub async fn load_transformers_model(
        &self,
        architecture: ModelArchitecture,
        repo: &str,
    ) -> Result<Box<dyn Model>, InferenceError> {
        match architecture {
            ModelArchitecture::Bert => {
                let model = candle_transformers::models::bert::BertModel::load(
                    &self.device, 
                    repo
                )?;
                Ok(Box::new(BertModelWrapper::new(model)))
            }
            ModelArchitecture::ResNet => {
                let model = candle_transformers::models::resnet::resnet18(&self.device)?;
                Ok(Box::new(ResNetModelWrapper::new(model)))
            }
            // ... other architectures
        }
    }
}
```

#### 2. HuggingFace Integration
```rust
// Direct HuggingFace Hub integration
pub async fn load_from_huggingface(
    repo: &str,
    filename: Option<&str>,
    revision: Option<&str>,
) -> Result<InferenceSession, InferenceError> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(repo.to_string());
    let model_path = repo.get(filename.unwrap_or("model.safetensors"))?;
    
    // Load using actual Candle model loading
    let session = CandleEngine::new()?.load_safetensors(&model_path).await?;
    Ok(InferenceSession::new(session))
}
```

#### 3. Real Inference Implementation
```rust
// No placeholder inference - use actual model forward passes
impl BertModelWrapper {
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        // Convert input to Candle tensor
        let candle_input = self.tensor_to_candle(input)?;
        
        // Run actual BERT forward pass
        let output = self.model.forward(&candle_input)?;
        
        // Convert output back to our tensor format
        self.candle_to_tensor(&output)
    }
}
```

### Linfa Usage Patterns (From Research)

Based on Linfa best practices, the library should expose:

#### 1. Classical ML Algorithms
```rust
// Real Linfa algorithm implementations
pub enum LinfaAlgorithm {
    KMeans { k: usize, max_iterations: usize },
    LinearRegression { fit_intercept: bool },
    LogisticRegression { max_iterations: usize },
    SVM { c: f64, kernel: String },
    DecisionTree { max_depth: Option<usize> },
    RandomForest { n_estimators: usize },
}

impl LinfaEngine {
    pub async fn train_algorithm(
        &self,
        algorithm: LinfaAlgorithm,
        features: Array2<f64>,
        targets: Option<Array1<f64>>,
    ) -> Result<Box<dyn Model>, InferenceError> {
        match algorithm {
            LinfaAlgorithm::KMeans { k, max_iterations } => {
                let dataset = Dataset::new(features, Array1::zeros(features.nrows()));
                let model = KMeans::params(k)
                    .max_n_iterations(max_iterations)
                    .fit(&dataset)?;
                Ok(Box::new(KMeansModelWrapper::new(model)))
            }
            LinfaAlgorithm::LinearRegression { fit_intercept } => {
                let targets = targets.ok_or_else(|| InferenceError::missing_targets())?;
                let dataset = Dataset::new(features, targets);
                let model = LinearRegression::default().fit(&dataset)?;
                Ok(Box::new(LinearRegressionModelWrapper::new(model)))
            }
            // ... other algorithms
        }
    }
}
```

#### 2. Data Preprocessing
```rust
// Expose Linfa's data preprocessing capabilities
pub struct DataPreprocessor;

impl DataPreprocessor {
    pub fn standardize(data: &Array2<f64>) -> Array2<f64> {
        // Use Linfa's standardization
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        (data - &mean) / &std
    }
    
    pub fn normalize(data: &Array2<f64>) -> Array2<f64> {
        // Use Linfa's normalization
        let min = data.fold_axis(Axis(0), f64::INFINITY, |&a, &b| a.min(b));
        let max = data.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));
        (data - &min) / (&max - &min)
    }
}
```

## New Architecture Design

### 1. Engine Structure

```rust
// Core engine trait - simplified
pub trait InferenceEngine: Send + Sync {
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError>;
    async fn load_from_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Model>, InferenceError>;
    fn supports_format(&self, format: &str) -> bool;
    fn engine_name(&self) -> &'static str;
}

// Simplified model trait
pub trait Model: Send + Sync {
    async fn predict(&self, input: &InferenceInput) -> Result<InferenceResult, InferenceError>;
    fn input_specs(&self) -> Vec<TensorSpec>;
    fn output_specs(&self) -> Vec<TensorSpec>;
}
```

### 2. Two Engine Implementation

#### Candle Engine (Deep Learning)
```rust
pub struct CandleEngine {
    device: Device,
    model_cache: Arc<RwLock<HashMap<String, Arc<dyn Model>>>>,
}

impl CandleEngine {
    // Real model architectures
    pub async fn load_bert(&self, repo: &str) -> Result<Box<dyn Model>, InferenceError> {
        let model = candle_transformers::models::bert::BertModel::load(&self.device, repo)?;
        Ok(Box::new(BertModelWrapper::new(model)))
    }
    
    pub async fn load_resnet(&self, variant: ResNetVariant) -> Result<Box<dyn Model>, InferenceError> {
        let model = match variant {
            ResNetVariant::ResNet18 => candle_transformers::models::resnet::resnet18(&self.device)?,
            ResNetVariant::ResNet50 => candle_transformers::models::resnet::resnet50(&self.device)?,
        };
        Ok(Box::new(ResNetModelWrapper::new(model)))
    }
    
    // Direct SafeTensors support
    pub async fn load_safetensors(&self, path: &str) -> Result<Box<dyn Model>, InferenceError> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        // Use actual Candle tensor operations, not placeholders
        Ok(Box::new(SafeTensorsModelWrapper::new(tensors)))
    }
}
```

#### Linfa Engine (Classical ML)
```rust
pub struct LinfaEngine;

impl LinfaEngine {
    // Real algorithm implementations
    pub async fn train_kmeans(
        &self,
        data: Array2<f64>,
        k: usize,
        max_iterations: usize,
    ) -> Result<Box<dyn Model>, InferenceError> {
        let dataset = Dataset::new(data, Array1::zeros(data.nrows()));
        let model = linfa_clustering::KMeans::params(k)
            .max_n_iterations(max_iterations)
            .fit(&dataset)?;
        Ok(Box::new(KMeansModelWrapper::new(model)))
    }
    
    pub async fn train_linear_regression(
        &self,
        features: Array2<f64>,
        targets: Array1<f64>,
    ) -> Result<Box<dyn Model>, InferenceError> {
        let dataset = Dataset::new(features, targets);
        let model = linfa_linear::LinearRegression::default().fit(&dataset)?;
        Ok(Box::new(LinearRegressionModelWrapper::new(model)))
    }
}
```

### 3. Flutter API Design

#### Main Interface
```dart
class InferenceSession {
  // Auto-detection (as specified in BRD)
  static Future<InferenceSession> load(String modelPath) async {
    final engine = await detectEngineFromPath(modelPath);
    return _loadWithEngine(modelPath, engine);
  }
  
  // Explicit engine selection
  static Future<InferenceSession> loadWithCandle(String modelPath) async {
    return _loadWithEngine(modelPath, 'candle');
  }
  
  // HuggingFace integration
  static Future<InferenceSession> loadFromHuggingFace({
    required String repo,
    String? filename,
    String? revision,
  }) async {
    return await loadFromHuggingFaceWithCandle(repo, filename, revision);
  }
  
  // Linfa training
  static Future<InferenceSession> trainLinfa({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  }) async {
    return await trainLinfaModel(data, algorithm, params);
  }
  
  // Prediction
  Future<InferenceResult> predict(InferenceInput input) async {
    return await predictWithSession(sessionHandle, input);
  }
}
```

#### Input Types
```dart
// Proper input type hierarchy
abstract class InferenceInput {
  const InferenceInput();
}

class ImageInput extends InferenceInput {
  final Uint8List bytes;
  final int width;
  final int height;
  final int channels;
  
  const ImageInput({
    required this.bytes,
    required this.width,
    required this.height,
    required this.channels,
  });
  
  // Real image processing
  static Future<ImageInput> fromFile(String path) async {
    final bytes = await File(path).readAsBytes();
    final image = img.decodeImage(bytes)!;
    return ImageInput(
      bytes: Uint8List.fromList(img.encodePng(image)),
      width: image.width,
      height: image.height,
      channels: 3,
    );
  }
}

class TextInput extends InferenceInput {
  final String text;
  final String? tokenizer;
  
  const TextInput({required this.text, this.tokenizer});
  
  // Real tokenization
  static Future<TextInput> fromText(String text, {String? tokenizer}) async {
    return TextInput(text: text, tokenizer: tokenizer);
  }
}
```

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create new project structure
- [ ] Set up proper dependencies (candle-transformers, linfa)
- [ ] Implement basic engine interfaces
- [ ] Create Flutter Rust Bridge setup

### Phase 2: Candle Engine (Weeks 3-4)
- [ ] Implement real model architectures (BERT, ResNet, MobileNet)
- [ ] Add HuggingFace Hub integration
- [ ] Create proper tensor conversion
- [ ] Remove all placeholder inference

### Phase 3: Linfa Engine (Weeks 5-6)
- [ ] Expand algorithm support
- [ ] Add data preprocessing utilities
- [ ] Implement model serialization
- [ ] Create training workflows

### Phase 4: Flutter Integration (Weeks 7-8)
- [ ] Implement Flutter API
- [ ] Create input/output types
- [ ] Add error handling
- [ ] Build comprehensive examples

### Phase 5: Testing & Documentation (Weeks 9-10)
- [ ] Create comprehensive test suite
- [ ] Add performance benchmarks
- [ ] Write detailed documentation
- [ ] Create tutorial examples

## Success Criteria

### Technical Goals
- ✅ Real ML inference (no placeholders)
- ✅ Support for 10+ model architectures
- ✅ HuggingFace Hub integration
- ✅ 3-line usage pattern

### Performance Goals
- ✅ < 100ms model loading time
- ✅ < 10ms inference time for small models
- ✅ Memory usage < 50MB overhead
- ✅ GPU acceleration support

### User Experience Goals
- ✅ Zero-setup installation
- ✅ Automatic engine detection
- ✅ Clear error messages
- ✅ Comprehensive documentation

## Example Usage (Target)

```dart
// Real BERT sentiment analysis
final model = await InferenceSession.loadFromHuggingFace(
  repo: 'bert-base-uncased',
);
final input = TextInput.fromText('This movie is amazing!');
final result = await model.predict(input);
final sentiment = result.topK(1).first; // Real classification

// Real image classification with ResNet
final imageModel = await InferenceSession.loadWithCandle('resnet50.safetensors');
final imageInput = await ImageInput.fromFile('photo.jpg');
final classification = await imageModel.predict(imageInput);

// Real clustering with K-means
final clusterModel = await InferenceSession.trainLinfa(
  data: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  algorithm: 'kmeans',
  params: {'k': 2},
);
final clusters = await clusterModel.predict(TensorInput.fromList([7.0, 8.0]));

// Real SafeTensors model inference
final safetensorsModel = await InferenceSession.loadWithCandle('model.safetensors');
final prediction = await safetensorsModel.predict(tensorInput);
```

## Conclusion

This implementation plan creates a true bridge library that exposes the full power of the Rust ML ecosystem to Flutter developers. By leveraging real model architectures from candle-transformers and classical ML algorithms from Linfa, we deliver on the promise of zero-setup ML inference with actual capabilities, not placeholders.

The key is to be a **bridge, not a framework** - providing thin, efficient wrappers around existing, mature Rust ML libraries rather than reinventing the wheel. 