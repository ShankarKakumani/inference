/// Zero-setup machine learning inference for Flutter applications.
///
/// This library provides a unified API for running ML models across different engines:
/// - **Candle**: PyTorch models (.safetensors, .pt, .pth)
/// - **Linfa**: Classical ML with on-device training
///
/// ## Quick Start
///
/// ```dart
/// import 'package:inference/inference.dart';
///
/// // Load any ML model with automatic engine detection
///
/// // Make predictions with type-safe inputs
/// final input = await ImageInput.fromAsset('assets/test_image.jpg');
/// final result = await model.predict(input);
///
/// // Access results with convenience methods
/// final topPrediction = result.topK(1).first;
/// print('Prediction: ${topPrediction.classIndex} (${topPrediction.confidence})');
///
/// // Clean up resources
/// model.dispose();
/// ```
///
/// ## Features
///
/// - 🚀 Zero Configuration: Install and start using ML models immediately
/// - 🌐 Universal Loading: Load from assets, URLs, files, or Hugging Face Hub
/// - 🔧 Unified API: One interface for PyTorch, and classical ML models
/// - 📱 Cross-Platform: Android, iOS, Windows, macOS, Linux support
/// - ⚡ Hardware Acceleration: Automatic GPU/NPU detection and optimization
/// - 🎯 Auto-Detection: Intelligent engine selection based on model format
/// - 💾 Smart Caching: Automatic model caching with size management
/// - 🧠 On-Device Training: Train classical ML models directly on device
/// - 🔒 Type-Safe: Full Dart type safety with comprehensive error handling
/// - 📊 Rich I/O: Built-in support for images, text, tensors, and audio
library;

// Main API (convenience wrapper)
export 'src/inference.dart';

// Core API
export 'src/inference_session.dart';
export 'src/inference_result.dart';
export 'src/inference_input.dart';

// Engine-specific sessions
export 'src/engines/candle_session.dart';
export 'src/engines/linfa_session.dart';

// Exceptions
export 'src/exceptions/inference_exceptions.dart';

// Rust bridge (for advanced usage)
export 'src/rust/api/inference.dart'
    show
        SessionConfig,
        getAvailableEngines,
        isEngineAvailable,
        detectEngineFromPath,
        detectEngineFromBytes;
export 'src/rust/models/tensor.dart';
export 'src/rust/frb_generated.dart' show RustLib;

// Re-export commonly used types for convenience
export 'src/engines/linfa_session.dart' show LinfaAlgorithm;
