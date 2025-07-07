# Inference Package - Technical Implementation Guide

## Package Overview

**Name:** `inference`  
**Mission:** Zero-setup ML inference for Flutter developers  
**Core Value:** Expose the full power of Rust ML ecosystem (`candle`, `ort`, `linfa`) to Flutter with a unified, dead-simple API

## User Experience

### Installation (1 command)
```bash
flutter pub add inference
```

### Basic Usage (3 lines of code)
```dart
final model = await Inference.load('assets/model.onnx');
final result = await model.predict(inputData);
print(result); // Done!
```

### Advanced Usage (Engine selection)
```dart
// Automatic engine selection (recommended)
final model = await Inference.load('assets/model.onnx');

// Explicit engine selection
final candleModel = await Inference.loadWithCandle('assets/pytorch_model.safetensors');
final ortModel = await Inference.loadWithOnnx('assets/model.onnx');
final linfaModel = await Inference.trainLinfa(dataset, algorithm: 'kmeans');
```

## Package Dependencies

### Rust Dependencies
```toml
[dependencies]
# Core ML frameworks
candle-core = { version = "0.8", features = ["cuda", "mkl"] }
candle-nn = "0.8"
candle-transformers = "0.8"
ort = { version = "2.0.0-rc.9", features = ["load-dynamic", "cuda", "tensorrt"] }
linfa = "0.7"
linfa-linear = "0.7" 
linfa-clustering = "0.7"
linfa-svm = "0.7"
linfa-trees = "0.7"

# Bridge and utilities
flutter_rust_bridge = "2.0"
tokio = { version = "1.0", features = ["rt-multi-thread"] }
ndarray = "0.16"
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"

# Image/data processing
image = "0.25"
hf-hub = "0.3" # For HuggingFace model downloads
safetensors = "0.4"
```

## Project Structure

### Rust Side (`rust/`)
```
rust/
├── Cargo.toml
├── build.rs                  # Build script for codegen
├── src/
│   ├── lib.rs                 # Main bridge exports
│   ├── api.rs                 # Flutter API definitions
│   ├── engines/
│   │   ├── mod.rs
│   │   ├── candle_engine.rs   # Candle implementation
│   │   ├── ort_engine.rs      # ONNX Runtime implementation
│   │   └── linfa_engine.rs    # Linfa implementation
│   ├── models/
│   │   ├── mod.rs
│   │   ├── session.rs         # Unified session interface
│   │   ├── tensor.rs          # Tensor operations
│   │   └── preprocessing.rs   # Data preprocessing
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── model_detector.rs  # Auto engine detection
│   │   └── converters.rs      # Data type conversions
│   └── bridge_generated.rs    # Generated bridge code
└── .cargo/
    └── config.toml            # Platform-specific configs
```

### Flutter Side (`lib/`)
```
lib/
├── inference.dart             # Main export
├── src/
│   ├── inference_session.dart # Main user interface
│   ├── inference_model.dart   # Model representation
│   ├── engines/
│   │   ├── candle.dart       # Candle-specific APIs
│   │   ├── onnx.dart         # ONNX-specific APIs
│   │   └── linfa.dart        # Linfa-specific APIs
│   ├── data/
│   │   ├── tensor.dart       # Tensor data structures
│   │   ├── image_input.dart  # Image preprocessing
│   │   └── text_input.dart   # Text preprocessing
│   ├── exceptions/
│   │   └── inference_exceptions.dart
│   └── bridge_generated.dart # Generated bridge code
├── android/                   # Android-specific code
├── ios/                       # iOS-specific code  
├── windows/                   # Windows-specific code
├── macos/                     # macOS-specific code
├── linux/                     # Linux-specific code
└── pubspec.yaml
```

### pubspec.yaml
```yaml
name: inference
description: Zero-setup ML inference for Flutter using Rust engines (Candle, ONNX, Linfa)
version: 1.0.0

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.16.0"

dependencies:
  flutter:
    sdk: flutter
  ffi: ^2.1.0
  meta: ^1.12.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_rust_bridge: ^2.0.0

flutter:
  plugin:
    platforms:
      android:
        ffiPlugin: true
      ios:
        ffiPlugin: true
      linux:
        ffiPlugin: true
      macos:
        ffiPlugin: true
      windows:
        ffiPlugin: true
```

## Core API Design

### Main Interface (`InferenceSession`)
```dart
class InferenceSession {
  // Static factory methods
  static Future<InferenceSession> load(String modelPath) async {
    // Auto-detect engine and load
  }
  
  static Future<InferenceSession> loadWithCandle(String modelPath) async {
    // Explicit Candle engine
  }
  
  static Future<InferenceSession> loadWithOnnx(String modelPath) async {
    // Explicit ONNX engine  
  }
  
  static Future<InferenceSession> trainLinfa({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  }) async {
    // Train Linfa model on-device
  }
  
  // Prediction methods
  Future<InferenceResult> predict(InferenceInput input) async;
  Future<List<InferenceResult>> predictBatch(List<InferenceInput> inputs) async;
  
  // Model information
  List<TensorInfo> get inputSpecs;
  List<TensorInfo> get outputSpecs;
  String get engine;
  
  // Cleanup
  void dispose();
}
```

### Input Types (`InferenceInput`)
```dart
abstract class InferenceInput {}

class ImageInput extends InferenceInput {
  final Uint8List bytes;
  final int width;
  final int height;
  final int channels;
  
  // Convenience constructors
  factory ImageInput.fromFile(File file) async;
  factory ImageInput.fromAsset(String assetPath) async;
  factory ImageInput.fromBytes(Uint8List bytes) async;
}

class NLPInput extends InferenceInput {
  final String text;
  final String? tokenizer; // Optional tokenizer specification
}

class TensorInput extends InferenceInput {
  final List<double> data;
  final List<int> shape;
  
  TensorInput(this.data, this.shape);
  
  // Type-safe constructors
  factory TensorInput.fromList(List<List<double>> data);
  factory TensorInput.from3D(List<List<List<double>>> data);
}

class AudioInput extends InferenceInput {
  final Float32List samples;
  final int sampleRate;
}
```

### Results (`InferenceResult`)
```dart
class InferenceResult {
  final List<double> data;
  final List<int> shape;
  final String dataType;
  
  // Convenience accessors
  double get scalar => data.first;
  List<double> get vector => data;
  List<List<double>> get matrix;
  
  // Classification helpers
  int get argmax;
  List<ClassificationResult> get topK;
}

class ClassificationResult {
  final int classIndex;
  final double confidence;
  final String? className;
}
```

## Engine-Specific APIs

### Candle Engine
```dart
class CandleSession extends InferenceSession {
  // HuggingFace integration
  static Future<CandleSession> fromHuggingFace({
    required String repo,
    String? revision,
    String? filename,
  });
  
  // PyTorch model loading
  static Future<CandleSession> fromPyTorch(String safetensorsPath);
  
  // Custom model architectures
  static Future<CandleSession> fromArchitecture({
    required String architecture, // 'resnet', 'bert', 'gpt2', etc.
    required String weightsPath,
  });
}
```

### ONNX Engine  
```dart
class OnnxSession extends InferenceSession {
  // Execution providers
  static Future<OnnxSession> withCuda(String modelPath);
  static Future<OnnxSession> withCoreML(String modelPath);
  static Future<OnnxSession> withCpu(String modelPath);
  
  // Optimization
  Future<void> optimize({
    GraphOptimizationLevel level = GraphOptimizationLevel.all,
    int? numThreads,
  });
}
```

### Linfa Engine
```dart
class LinfaSession extends InferenceSession {
  // Supported algorithms
  static Future<LinfaSession> trainKMeans({
    required List<List<double>> data,
    required int numClusters,
    int maxIterations = 100,
  });
  
  static Future<LinfaSession> trainLinearRegression({
    required List<List<double>> features,
    required List<double> targets,
    double? l1Ratio,
    double? l2Ratio,
  });
  
  static Future<LinfaSession> trainSVM({
    required List<List<double>> features, 
    required List<int> labels,
    String kernel = 'rbf',
    Map<String, dynamic>? params,
  });
  
  // Model persistence
  Future<Uint8List> serialize();
  static Future<LinfaSession> deserialize(Uint8List bytes);
}
```

## Rust Implementation Details

### Engine Trait (`engines/mod.rs`)
```rust
pub trait InferenceEngine: Send + Sync {
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError>;
    async fn load_from_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Model>, InferenceError>;
    fn supports_format(&self, format: &ModelFormat) -> bool;
    fn engine_name(&self) -> &'static str;
}

pub trait Model: Send + Sync {
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError>;
    async fn predict_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError>;
    fn input_specs(&self) -> &[TensorSpec];
    fn output_specs(&self) -> &[TensorSpec];
}
```

### Candle Engine (`engines/candle_engine.rs`)
```rust
pub struct CandleEngine;

impl InferenceEngine for CandleEngine {
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError> {
        let device = Device::Cpu; // or Device::Cuda(0) if available
        
        // Auto-detect model type
        if path.ends_with(".safetensors") {
            let model = load_safetensors_model(path, &device)?;
            Ok(Box::new(CandleModel::new(model, device)))
        } else {
            Err(InferenceError::UnsupportedFormat(path.to_string()))
        }
    }
}

pub struct CandleModel {
    model: Box<dyn candle_nn::Module>,
    device: Device,
}

impl Model for CandleModel {
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        let candle_tensor = input.to_candle_tensor(&self.device)?;
        let output = self.model.forward(&candle_tensor)?;
        Ok(Tensor::from_candle(output))
    }
}
```

### ORT Engine (`engines/ort_engine.rs`)
```rust
pub struct OrtEngine;

impl InferenceEngine for OrtEngine {
    async fn load_model(&self, path: &str) -> Result<Box<dyn Model>, InferenceError> {
        let session = ort::SessionBuilder::new()?
            .with_optimization_level(ort::GraphOptimizationLevel::All)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(path)?;
            
        Ok(Box::new(OrtModel::new(session)))
    }
}

pub struct OrtModel {
    session: ort::Session,
}

impl Model for OrtModel {
    async fn predict(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        let ort_inputs = input.to_ort_inputs()?;
        let outputs = self.session.run(ort_inputs)?;
        Ok(Tensor::from_ort_outputs(outputs))
    }
}
```

### Linfa Engine (`engines/linfa_engine.rs`)
```rust
pub struct LinfaEngine;

impl LinfaBackend {
    pub async fn train_kmeans(
        data: &Array2<f64>,
        n_clusters: usize,
    ) -> Result<Box<dyn Model>, InferenceError> {
        let dataset = Dataset::new(data.clone(), Array1::zeros(data.nrows()));
        let model = KMeans::params(n_clusters)
            .max_n_iterations(100)
            .fit(&dataset)?;
            
        Ok(Box::new(LinfaKMeansModel::new(model)))
    }
    
    pub async fn train_linear_regression(
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> Result<Box<dyn Model>, InferenceError> {
        let dataset = Dataset::new(features.clone(), targets.clone());
        let model = LinearRegression::default().fit(&dataset)?;
        
        Ok(Box::new(LinfaLinearModel::new(model)))
    }
}
```

## Auto Engine Detection (`utils/model_detector.rs`)
```rust
pub fn detect_engine(path: &str) -> InferenceEngine {
    match path {
        p if p.ends_with(".onnx") => InferenceEngine::Ort,
        p if p.ends_with(".safetensors") => InferenceEngine::Candle,
        p if p.ends_with(".pt") => InferenceEngine::Candle,
        p if p.ends_with(".pth") => InferenceEngine::Candle,
        _ => {
            // Try to read file header to detect format
            detect_from_content(path).unwrap_or(InferenceEngine::Ort)
        }
    }
}

fn detect_from_content(path: &str) -> Option<InferenceEngine> {
    let bytes = std::fs::read(path).ok()?;
    
    // ONNX files start with specific magic bytes
    if bytes.starts_with(b"\x08") {
        return Some(InferenceEngine::Ort);
    }
    
    // SafeTensors files start with JSON metadata
    if bytes.starts_with(b"{") {
        return Some(InferenceEngine::Candle);
    }
    
    None
}
```

## Error Handling
```dart
// Flutter exceptions
abstract class InferenceException implements Exception {
  final String message;
  const InferenceException(this.message);
}

class ModelLoadException extends InferenceException {
  const ModelLoadException(String message) : super(message);
}

class PredictionException extends InferenceException {
  const PredictionException(String message) : super(message);
}

class UnsupportedFormatException extends InferenceException {
  const UnsupportedFormatException(String format) 
    : super('Unsupported model format: $format');
}
```

```rust
// Rust errors
#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    
    #[error("Prediction failed: {0}")]
    Prediction(String),
    
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Engine error: {0}")]
    Engine(#[from] anyhow::Error),
}
```

## Usage Examples

### Image Classification
```dart
// Load model
final model = await Inference.load('assets/mobilenet_v2.onnx');

// Predict from camera
final image = await ImagePicker().pickImage(source: ImageSource.camera);
final input = await ImageInput.fromFile(File(image!.path));
final result = await model.predict(input);

// Get top 5 predictions
final top5 = result.topK(5);
print('Top predictions:');
for (final pred in top5) {
  print('${pred.className}: ${pred.confidence * 100:.1f}%');
}
```

### Text Sentiment Analysis
```dart
final model = await Inference.loadWithOnnx('assets/sentiment_model.onnx');

final input = NLPInput('This movie is amazing!');
final result = await model.predict(input);

final sentiment = result.scalar > 0.5 ? 'Positive' : 'Negative';
final confidence = result.scalar;
print('Sentiment: $sentiment (${confidence * 100:.1f}% confidence)');
```

### On-Device Training
```dart
// Prepare training data
final features = [
  [1.0, 2.0],
  [2.0, 3.0], 
  [3.0, 1.0],
  [8.0, 9.0],
  [9.0, 8.0],
  [10.0, 9.0],
];

// Train K-means model
final model = await Inference.trainLinfa(
  data: features,
  algorithm: 'kmeans',
  params: {'n_clusters': 2},
);

// Predict cluster for new point
final result = await model.predict(TensorInput([5.0, 6.0], [1, 2]));
final cluster = result.argmax;
print('Point belongs to cluster: $cluster');
```

### Real-time Object Detection
```dart
class ObjectDetector {
  late InferenceSession _model;
  
  Future<void> initialize() async {
    _model = await Inference.loadWithOnnx('assets/yolo_v8.onnx');
  }
  
  Future<List<Detection>> detectObjects(Uint8List imageBytes) async {
    final input = await ImageInput.fromBytes(imageBytes);
    final result = await _model.predict(input);
    
    return parseYoloOutput(result);
  }
}
```

## Configuration Options
```dart
// Global configuration
Inference.configure(
  defaultEngine: InferenceEngine.auto,
  cacheModels: true,
  maxCacheSize: 500, // MB
  numThreads: Platform.numberOfProcessors,
  enableGpuAcceleration: true,
);

// Per-session configuration
final model = await Inference.load(
  'assets/model.onnx',
  options: SessionOptions(
    engine: InferenceEngine.onnx,
    executionProvider: ExecutionProvider.cuda,
    optimizationLevel: OptimizationLevel.all,
    enableProfiling: true,
  ),
);
```

This gives developers everything they need - dead simple for basic use cases, but powerful enough for advanced scenarios. The API is intuitive, the setup is zero-effort, and it exposes the full power of the Rust ML ecosystem to Flutter developers.

## Build Configuration

### Cargo.toml (Rust)
```toml
[package]
name = "inference_core"
version = "1.0.0"
edition = "2021"

[lib]
crate-type = ["staticlib", "cdylib"]

[dependencies]
# ... dependencies listed above

[build-dependencies]
flutter_rust_bridge_codegen = "2.0"

# Platform-specific configs
[target.'cfg(target_os = "android")'.dependencies]
jni = "0.21"

[target.'cfg(target_os = "ios")'.dependencies]
objc = "0.2"

[features]
default = ["candle", "ort", "linfa"]
candle = ["candle-core", "candle-nn", "candle-transformers"]
ort = ["dep:ort"]
linfa = ["dep:linfa", "linfa-linear", "linfa-clustering", "linfa-svm"]
gpu = ["candle-core/cuda", "ort/cuda"]
```

### build.rs
```rust
fn main() {
    flutter_rust_bridge_codegen::generate();
}
```

### .cargo/config.toml
```toml
[target.aarch64-apple-ios]
rustflags = ["-C", "link-arg=-Wl,-application_extension"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-Wl,-rpath,@loader_path"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-Wl,-rpath,@loader_path"]
```