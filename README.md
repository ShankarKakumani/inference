# Inference

[![pub package](https://img.shields.io/pub/v/inference.svg)](https://pub.dev/packages/inference)
[![package publisher](https://img.shields.io/pub/publisher/inference.svg)](https://pub.dev/packages/inference/publisher)
[![Flutter compatibility](https://img.shields.io/badge/Flutter-3.16%2B-blue)](https://flutter.dev)
[![Dart compatibility](https://img.shields.io/badge/Dart-3.0%2B-blue)](https://dart.dev)

📦 **[View on pub.dev](https://pub.dev/packages/inference)** | 🚀 **[Try it now](https://pub.dev/packages/inference/install)**



> **🚧 Beta Release**: This package is in beta. The API is stable but may have minor changes based on community feedback. Perfect for testing and early adoption!



**Zero-setup machine learning inference for Flutter applications.**

Inference brings the full power of modern ML engines (Candle, Linfa) to Flutter with a unified, developer-friendly API. Load models from anywhere—assets, URLs, Hugging Face Hub—run predictions on any platform, and even train models on-device, all with just a few lines of code.

---

## Features

- **🚀 Zero Configuration**: Install and start using ML models immediately
- **🌐 Universal Loading**: Load from assets, URLs, files, or Hugging Face Hub
- **🤗 Hugging Face Ready**: Direct integration with Hugging Face Hub models
- **🔧 Unified API**: One interface for PyTorch and classical ML models  
- **📱 Cross-Platform**: Android, iOS, Windows, macOS, Linux support
- **⚡ Hardware Acceleration**: Automatic GPU/NPU detection and optimization
- **🎯 Auto-Detection**: Intelligent engine selection based on model format
- **💾 Smart Caching**: Automatic model caching with size management
- **🧠 On-Device Training**: Train classical ML models directly on device
- **🔒 Type-Safe**: Full Dart type safety with comprehensive error handling
- **📊 Rich I/O**: Built-in support for images, text, tensors, and audio

## Quick Start

### Installation

Add `inference` to your `pubspec.yaml`:

```yaml
dependencies:
  inference: ^0.1.0-beta.2
```

Or install via command line:

```bash
flutter pub add inference
```

**Note**: Beta versions require explicit version specification in `pubspec.yaml`.

### Basic Usage

```dart
import 'package:inference/inference.dart';

// Load any ML model with automatic engine detection
final model = await InferenceSession.load('assets/model.safetensors');

// Make predictions with type-safe inputs
final input = await ImageInput.fromAsset('assets/test_image.jpg');
final result = await model.predict(input);

// Access results with convenience methods
final topPrediction = result.topK(1).first;
print('Prediction: ${topPrediction.classIndex} (${topPrediction.confidence})');

// Clean up resources
model.dispose();
```

## Model Loading Options

Inference supports multiple ways to load models, giving you maximum flexibility:

### 1. Asset Loading (Bundled Models)

```dart
// Load from app assets
final model = await InferenceSession.load('assets/models/classifier.safetensors');
```

### 2. URL Loading (Remote Models)

```dart
// Load from any URL with automatic caching
final model = await InferenceSession.loadFromUrl(
  'https://example.com/models/classifier.safetensors',
  cache: true, // Enable caching (default)
);

// Custom cache key for organization
final model = await InferenceSession.loadFromUrl(
  'https://example.com/large_model.safetensors',
  cache: true,
  cacheKey: 'production_classifier_v2',
);
```

### 3. Hugging Face Hub Integration

```dart
// Load directly from Hugging Face
final detector = await InferenceSession.loadFromHuggingFace(
  'qualcomm/EasyOCR',
  filename: 'EasyOCR.safetensors',
);

// Specify model revision
final model = await InferenceSession.loadFromHuggingFace(
  'microsoft/DialoGPT-medium',
  filename: 'pytorch_model.safetensors',
  revision: 'v1.0.0',
);
```

### 4. File System Loading

```dart
// Load from local file system
final model = await InferenceSession.loadFromFile('/path/to/downloaded/model.safetensors');
```

### 5. Cache Management

```dart
// Check cache size
final sizeBytes = await InferenceSession.getCacheSize();
print('Cache size: ${(sizeBytes / 1024 / 1024).toStringAsFixed(1)} MB');

// Clear cache when needed
await InferenceSession.clearCache();
```

## Examples

### Real-World OCR with EasyOCR

```dart
import 'package:inference/inference.dart';

class EasyOCRPipeline {
  late InferenceSession _detector;
  late InferenceSession _recognizer;
  
  Future<void> initialize() async {
    // Load both models from Hugging Face
    _detector = await InferenceSession.loadFromHuggingFace(
      'qualcomm/EasyOCR',
      filename: 'EasyOCR.safetensors', // Text detection model (79.2 MB)
    );
    
    _recognizer = await InferenceSession.loadFromHuggingFace(
      'qualcomm/EasyOCR',
      filename: 'EasyOCRRecognizer.safetensors', // Text recognition model (14.7 MB)
    );
  }
  
  Future<List<String>> extractText(String imagePath) async {
    // Step 1: Detect text regions
    final imageInput = await ImageInput.fromFile(File(imagePath));
    final detectionResult = await _detector.predict(imageInput);
    
    // Step 2: Extract text from each region
    final textRegions = parseDetectionResult(detectionResult);
    final extractedTexts = <String>[];
    
    for (final region in textRegions) {
      final regionInput = await ImageInput.fromCrop(imageInput, region);
      final recognitionResult = await _recognizer.predict(regionInput);
      final text = parseRecognitionResult(recognitionResult);
      extractedTexts.add(text);
    }
    
    return extractedTexts;
  }
  
  void dispose() {
    _detector.dispose();
    _recognizer.dispose();
  }
}
```

### Image Classification with Model Download

```dart
import 'package:inference/inference.dart';
import 'package:image_picker/image_picker.dart';

class ImageClassifier {
  late InferenceSession _model;
  
  Future<void> initialize() async {
    // Download and cache MobileNet from a remote source
    _model = await InferenceSession.loadFromUrl(
      'https://example.com/models/mobilenet_v2.safetensors',
      cache: true,
      cacheKey: 'mobilenet_v2_imagenet',
    );
  }
  
  Future<String> classifyImage() async {
    // Get image from camera
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);
    if (image == null) return 'No image selected';
    
    // Create input and predict
    final input = await ImageInput.fromFile(File(image.path));
    final result = await _model.predict(input);
    
    // Get top prediction
    final prediction = result.topK(1).first;
    return 'Class: ${prediction.classIndex}, Confidence: ${(prediction.confidence * 100).toStringAsFixed(1)}%';
  }
  
  void dispose() => _model.dispose();
}
```

### Text Sentiment Analysis

```dart
import 'package:inference/inference.dart';

class SentimentAnalyzer {
  late InferenceSession _model;
  
  Future<void> initialize() async {
    _model = await InferenceSession.loadWithCandle('assets/bert_sentiment.safetensors');
  }
  
  Future<Map<String, dynamic>> analyzeSentiment(String text) async {
    final input = NLPInput(text);
    final result = await _model.predict(input);
    
    final isPositive = result.scalar > 0.5;
    final confidence = isPositive ? result.scalar : 1 - result.scalar;
    
    return {
      'sentiment': isPositive ? 'positive' : 'negative',
      'confidence': confidence,
      'score': result.scalar,
    };
  }
  
  void dispose() => _model.dispose();
}
```

### On-Device Training

```dart
import 'package:inference/inference.dart';

class OnDeviceTrainer {
  Future<InferenceSession> trainClustering(List<List<double>> data) async {
    // Train K-means clustering on device
    final model = await InferenceSession.trainLinfa(
      data: data,
      algorithm: 'kmeans',
      params: {
        'n_clusters': 3,
        'max_iterations': 100,
        'tolerance': 1e-4,
      },
    );
    
    return model;
  }
  
  Future<int> predictCluster(InferenceSession model, List<double> point) async {
    final input = TensorInput(point, [point.length]);
    final result = await model.predict(input);
    return result.argmax;
  }
}
```

### Batch Processing

```dart
import 'package:inference/inference.dart';

class BatchProcessor {
  late InferenceSession _model;
  
  Future<void> initialize() async {
    _model = await InferenceSession.load('assets/classifier.safetensors');
  }
  
  Future<List<InferenceResult>> processImages(List<String> imagePaths) async {
    // Create inputs for all images
    final inputs = await Future.wait(
      imagePaths.map((path) => ImageInput.fromFile(File(path))),
    );
    
    // Process all images in a single batch for better performance
    return await _model.predictBatch(inputs);
  }
  
  void dispose() => _model.dispose();
}
```

## API Reference

### Core Classes

#### `InferenceSession`

The main interface for ML inference sessions.

```dart
class InferenceSession {
  // Asset loading
  static Future<InferenceSession> load(String modelPath);
  
  // URL loading with caching
  static Future<InferenceSession> loadFromUrl(String url, {bool cache = true, String? cacheKey});
  
  // File system loading
  static Future<InferenceSession> loadFromFile(String filePath);
  
  // Hugging Face Hub integration
  static Future<InferenceSession> loadFromHuggingFace(String modelId, {required String filename, String? revision});
  
  // Engine-specific loading
  static Future<CandleSession> loadWithCandle(String modelPath);

  
  // On-device training
  static Future<LinfaSession> trainLinfa({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  });
  
  // Cache management
  static Future<void> clearCache();
  static Future<int> getCacheSize();
  
  // Inference methods
  Future<InferenceResult> predict(InferenceInput input);
  Future<List<InferenceResult>> predictBatch(List<InferenceInput> inputs);
  
  // Resource management
  void dispose();
  
  // Properties
  List<TensorSpec> get inputSpecs;
  List<TensorSpec> get outputSpecs;
  String get engine;
}
```

### Input Types

#### `ImageInput`

For computer vision models.

```dart
class ImageInput extends InferenceInput {
  // Constructors
  ImageInput({required Uint8List bytes, required int width, required int height, required int channels});
  
  // Convenience factories
  static Future<ImageInput> fromFile(File file);
  static Future<ImageInput> fromAsset(String assetPath);
  static Future<ImageInput> fromBytes(Uint8List bytes);
  static ImageInput.fromPixels({required Float32List pixels, required int width, required int height, required int channels});
}
```

#### `NLPInput`

For natural language processing models.

```dart
class NLPInput extends InferenceInput {
  NLPInput(String text, {String? tokenizer, List<int>? tokenIds});
  
  // Pre-tokenized input
  factory NLPInput.fromTokens(List<int> tokens);
}
```

#### `TensorInput`

For direct tensor data.

```dart
class TensorInput extends InferenceInput {
  TensorInput(List<double> data, List<int> shape);
  
  // Convenience factories
  factory TensorInput.fromList(List<List<double>> data);
  factory TensorInput.from3D(List<List<List<double>>> data);
}
```

#### `AudioInput`

For audio processing models.

```dart
class AudioInput extends InferenceInput {
  AudioInput({required Float32List samples, required int sampleRate});
  
  static Future<AudioInput> fromFile(File file);
}
```

### Results

#### `InferenceResult`

Contains prediction results with convenience accessors.

```dart
class InferenceResult {
  // Raw data access
  Float32List get data;
  List<int> get shape;
  String get dataType;
  
  // Convenience accessors
  double get scalar;                           // Single value
  List<double> get vector;                     // 1D array
  List<List<double>> get matrix;               // 2D array
  
  // Classification helpers
  int get argmax;                              // Index of maximum value
  List<ClassificationResult> topK([int k]);   // Top K predictions
  List<ClassificationResult> topKSoftmax([int k]); // Top K with softmax
}
```

#### `ClassificationResult`

Individual classification prediction.

```dart
class ClassificationResult {
  final int classIndex;
  final double confidence;
  final String? className;
}
```

### Engine-Specific Sessions

#### `CandleSession`

For PyTorch models with HuggingFace integration.

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
  
  // Custom architectures
  static Future<CandleSession> fromArchitecture({
    required String architecture,
    required String weightsPath,
  });
  
  // Device management
  bool get isCudaAvailable;
  bool get isMklAvailable;
  String get device;
  Future<void> toDevice(String device);
}
```



#### `LinfaSession`

For classical ML algorithms with on-device training.

```dart
class LinfaSession extends InferenceSession {
  // Clustering
  static Future<LinfaSession> trainKMeans({
    required List<List<double>> data,
    required int numClusters,
    int maxIterations = 100,
    double tolerance = 1e-4,
  });
  
  // Regression
  static Future<LinfaSession> trainLinearRegression({
    required List<List<double>> features,
    required List<double> targets,
    double? l1Ratio,
    double? l2Ratio,
  });
  
  // Classification
  static Future<LinfaSession> trainSVM({
    required List<List<double>> features,
    required List<int> labels,
    String kernel = 'rbf',
    Map<String, dynamic>? params,
  });
  
  // Decision trees
  static Future<LinfaSession> trainDecisionTree({
    required List<List<double>> features,
    required List<int> labels,
    int? maxDepth,
    int? minSamplesSplit,
  });
  
  // Model persistence
  Future<Uint8List> serialize();
  static Future<LinfaSession> deserialize(Uint8List bytes);
}
```

## Supported Formats

| Engine | Formats | Use Cases |
|--------|---------|-----------|
| **Candle** | `.safetensors`, `.pt`, `.pth` | PyTorch models, HuggingFace models, Computer vision, NLP |
| **Linfa** | Training data | Classical ML, On-device training, Small datasets |

## Platform Support

| Platform | Candle | Linfa |
|----------|--------|-------|
| Android  |   ✅   |   ✅  |
| iOS      |   ✅   |   ✅  |
| Windows  |   ✅   |   ✅  |
| macOS    |   ✅   |   ✅  |
| Linux    |   ✅   |   ✅  |

## Performance Tips

### Memory Management

Always dispose of sessions when done:

```dart
final model = await InferenceSession.load('model.safetensors');
try {
  // Use model...
} finally {
  model.dispose(); // Important: prevents memory leaks
}
```

### Batch Processing

Use batch predictions for better performance:

```dart
// ✅ Good: Process multiple inputs together
final results = await model.predictBatch(inputs);

// ❌ Avoid: Processing inputs one by one
final results = <InferenceResult>[];
for (final input in inputs) {
  results.add(await model.predict(input));
}
```

### GPU Acceleration

Enable GPU acceleration when available:

```dart
// Automatically detect and use best execution provider
final model = await InferenceSession.load('model.safetensors');
```

### Smart Caching

Models downloaded from URLs are cached automatically:

```dart
// First load: downloads and caches
final model1 = await InferenceSession.loadFromUrl('https://example.com/model.safetensors');

// Second load: uses cache (much faster)
final model2 = await InferenceSession.loadFromUrl('https://example.com/model.safetensors');

// Manage cache size
final sizeBytes = await InferenceSession.getCacheSize();
if (sizeBytes > 500 * 1024 * 1024) { // If cache > 500MB
  await InferenceSession.clearCache();
}
```



## Troubleshooting

### Common Issues

**Model loading fails**
- Verify the model file exists and is accessible
- Check that the model format is supported  
- Ensure sufficient memory is available
- For URL loading: check internet connectivity and URL validity
- For Hugging Face models: verify repository and filename exist

**Slow inference**
- Enable GPU acceleration if available
- Use batch processing for multiple inputs
- Optimize model with appropriate optimization level

**Memory issues**
- Always call `dispose()` on sessions
- Avoid loading multiple large models simultaneously
- Consider model quantization for memory-constrained devices

### Error Handling

```dart
try {
  final model = await InferenceSession.load('model.safetensors');
  final result = await model.predict(input);
} on ModelLoadException catch (e) {
  print('Failed to load model: ${e.message}');
} on PredictionException catch (e) {
  print('Prediction failed: ${e.message}');
} on UnsupportedEngineException catch (e) {
  print('Engine not supported: ${e.message}');
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install dependencies: `flutter pub get`
3. Run tests: `flutter test`
4. Run example: `cd example && flutter run`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Candle**: Rust-based PyTorch implementation
- **Linfa**: Rust machine learning toolkit
- **Flutter Rust Bridge**: Seamless Rust-Flutter integration

---

<p align="center">
  Made with ❤️ by the Flutter community
</p>

