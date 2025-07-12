## 1.0.0

* **🎉 First stable release** of the Inference Flutter package
* **Core API Implementation**: Complete implementation of the unified inference API
  - ✅ `InferenceSession.load()` with automatic engine detection
  - ✅ `InferenceSession.loadFromUrl()` with smart caching
  - ✅ `InferenceSession.loadFromHuggingFace()` for direct Hub integration
  - ✅ `InferenceSession.trainLinfa()` for on-device training
* **Model Architecture Support**: 
  - ✅ **BERT Models**: Complete wrapper for text classification and NLP
  - ✅ **ResNet Models**: Complete wrapper for image classification
  - ✅ **Generic SafeTensors**: Support for any SafeTensors model format
* **Engine Integration**:
  - ✅ **Candle Engine**: Full PyTorch model support with GPU acceleration
  - ✅ **Linfa Engine**: On-device training with K-means clustering
  - ✅ **Automatic Detection**: Smart engine selection based on model format
* **Input/Output System**:
  - ✅ `ImageInput`, `NLPInput`, `TensorInput`, `AudioInput` classes
  - ✅ `InferenceResult` with convenience accessors (`scalar`, `vector`, `matrix`)
  - ✅ Classification helpers (`argmax`, `topK()`)
* **HuggingFace Integration**: 
  - ✅ Direct model loading from HuggingFace Hub
  - ✅ Automatic URL construction and caching
  - ✅ Support for specific revisions and filenames
* **Performance Features**:
  - ✅ Automatic GPU acceleration when available
  - ✅ Batch processing with `predictBatch()`
  - ✅ Smart model caching with size management
  - ✅ Resource management with `dispose()` methods
* **Cross-Platform Support**: Verified on Android, iOS, Windows, macOS, Linux
* **Documentation**: Comprehensive README with real-world examples
* **Future Roadmap**: [Model Wrappers Roadmap](knowledge/model_wrappers_roadmap.md) with 20+ additional architectures planned

## 0.1.0-beta.4

* **Enhanced examples and updated dependencies** for better pub.dev experience
* Added comprehensive standalone example.dart showcasing core functionality
* Created detailed example README.md following Dart package layout conventions
* Updated dependencies to latest versions with proper version constraints
* Improved pub.dev score with comprehensive documentation and code quality fixes
* Added detailed library documentation with examples and feature overview
* Documented all missing constructors and classes for 100% API coverage
* Fixed static analysis issues including unused imports and style improvements
* Added package topics for better discoverability
* Enhanced code formatting and consistency

## 0.1.0-beta.2

* **Bug fixes and improvements** for the second beta release
* Enhanced stability and performance optimizations
* Improved error handling and documentation
* Updated dependencies and build configurations

## 0.1.0-beta.1

* **Initial beta release** of the Inference Flutter package
* Zero-setup ML inference with unified API for Candle and Linfa engines
* Support for automatic model format detection (.safetensors, .pt, .pth)
* Cross-platform support (Android, iOS, Windows, macOS, Linux)
* Multiple input types: ImageInput, NLPInput, TensorInput, AudioInput
* Comprehensive example app with image classification, text sentiment, and on-device training
* Built with Flutter Rust Bridge 2.0 for optimal performance
* GPU acceleration support where available

## 0.0.1

* TODO: Describe initial release.
