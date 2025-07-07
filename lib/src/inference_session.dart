import 'dart:typed_data';
import 'package:meta/meta.dart';

import 'inference_input.dart';
import 'inference_result.dart';
import 'engines/candle_session.dart';
import 'engines/linfa_session.dart';
import 'rust/api/inference.dart' as rust_api;
import 'rust/models/tensor.dart';
import 'exceptions/inference_exceptions.dart';

/// Main interface for ML inference sessions
///
/// This is the primary entry point for the Inference package, providing
/// a unified interface for loading models and making predictions across
/// different ML engines (Candle, Linfa).
abstract class InferenceSession {
  /// Handle to the underlying Rust session
  @protected
  final BigInt sessionHandle;

  /// Engine type used by this session
  final String engine;

  /// Input tensor specifications
  final List<TensorSpec> inputSpecs;

  /// Output tensor specifications
  final List<TensorSpec> outputSpecs;

  @protected
  InferenceSession({
    required this.sessionHandle,
    required this.engine,
    required this.inputSpecs,
    required this.outputSpecs,
  });

  /// Load a model with automatic engine detection
  ///
  /// This method automatically detects the appropriate engine based on
  /// the model file format and loads it accordingly.
  ///
  /// Example:
  /// ```dart
  /// final model = await InferenceSession.load('assets/model.safetensors');
  /// ```
  static Future<InferenceSession> load(String modelPath) async {
    try {
      final sessionInfo = await rust_api.loadModel(modelPath: modelPath);
      return _createSessionFromInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load model: $e');
    }
  }

  /// Load a model with explicit Candle engine
  ///
  /// Forces the use of the Candle engine for PyTorch models.
  /// Supports .safetensors, .pt, and .pth files.
  ///
  /// Example:
  /// ```dart
  /// final model = await InferenceSession.loadWithCandle('assets/model.safetensors');
  /// ```
  static Future<CandleSession> loadWithCandle(String modelPath) async {
    try {
      final sessionInfo =
          await rust_api.loadModelWithCandle(modelPath: modelPath);
      return CandleSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load Candle model: $e');
    }
  }

  /// Load a model from bytes with automatic engine detection
  ///
  /// This method automatically detects the appropriate engine based on
  /// the model file format and loads it accordingly.
  ///
  /// Example:
  /// ```dart
  /// final bytes = await rootBundle.load('assets/model.safetensors');
  /// final model = await InferenceSession.loadFromBytes(bytes.buffer.asUint8List());
  /// ```
  static Future<InferenceSession> loadFromBytes(Uint8List modelBytes) async {
    try {
      final config = await rust_api.SessionConfig.default_();
      final sessionInfo = await rust_api.loadModelFromBytes(
        modelBytes: modelBytes,
        config: config,
      );
      return _createSessionFromInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load model from bytes: $e');
    }
  }

  /// Load a model from bytes with explicit Candle engine
  ///
  /// Forces the use of the Candle engine for PyTorch models.
  /// Supports .safetensors, .pt, and .pth files.
  ///
  /// Example:
  /// ```dart
  /// final bytes = await rootBundle.load('assets/model.safetensors');
  /// final model = await InferenceSession.loadFromBytesWithCandle(bytes.buffer.asUint8List());
  /// ```
  static Future<CandleSession> loadFromBytesWithCandle(
      Uint8List modelBytes) async {
    try {
      final defaultConfig = await rust_api.SessionConfig.default_();
      final config = rust_api.SessionConfig(
        engineType: 'candle',
        gpuAcceleration: defaultConfig.gpuAcceleration,
        numThreads: defaultConfig.numThreads,
        optimizationLevel: defaultConfig.optimizationLevel,
      );
      final sessionInfo = await rust_api.loadModelFromBytes(
        modelBytes: modelBytes,
        config: config,
      );
      return CandleSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load Candle model from bytes: $e');
    }
  }

  /// Load a model from a URL with automatic caching
  ///
  /// Downloads the model from the specified URL and caches it locally
  /// for faster subsequent loads. Supports automatic engine detection.
  ///
  /// Example:
  /// ```dart
  /// final model = await InferenceSession.loadFromUrl(
  ///   'https://huggingface.co/qualcomm/EasyOCR/resolve/main/EasyOCR.safetensors',
  ///   cache: true,
  /// );
  /// ```
  static Future<InferenceSession> loadFromUrl(
    String url, {
    bool cache = true,
    String? cacheKey,
  }) async {
    try {
      final sessionInfo = await rust_api.loadModelFromUrl(
        url: url,
        cache: cache,
        cacheKey: cacheKey,
      );
      return _createSessionFromInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load model from URL: $e');
    }
  }

  /// Load a model from local file system
  ///
  /// Loads a model from an absolute file path with automatic engine detection.
  ///
  /// Example:
  /// ```dart
  /// final model = await InferenceSession.loadFromFile('/path/to/model.safetensors');
  /// ```
  static Future<InferenceSession> loadFromFile(String filePath) async {
    try {
      final sessionInfo = await rust_api.loadModelFromFile(filePath: filePath);
      return _createSessionFromInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load model from file: $e');
    }
  }

  /// Load a model from Hugging Face Hub
  ///
  /// Convenient method for loading models directly from Hugging Face.
  /// Automatically handles URL construction and caching.
  ///
  /// Example:
  /// ```dart
  /// // Load EasyOCR detector
  /// final detector = await InferenceSession.loadFromHuggingFace(
  ///   'qualcomm/EasyOCR',
  ///   filename: 'EasyOCR.safetensors',
  /// );
  ///
  /// // Load EasyOCR recognizer
  /// final recognizer = await InferenceSession.loadFromHuggingFace(
  ///   'qualcomm/EasyOCR',
  ///   filename: 'EasyOCRRecognizer.safetensors',
  /// );
  /// ```
  static Future<InferenceSession> loadFromHuggingFace(
    String modelId, {
    required String filename,
    String? revision,
  }) async {
    try {
      final sessionInfo = await rust_api.loadFromHuggingface(
        repo: modelId,
        revision: revision,
        filename: filename,
      );
      return _createSessionFromInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to load model from Hugging Face: $e');
    }
  }

  /// Clear the model cache
  ///
  /// Removes all cached models to free up disk space.
  ///
  /// Example:
  /// ```dart
  /// await InferenceSession.clearCache();
  /// ```
  static Future<void> clearCache() async {
    try {
      await rust_api.clearCache();
    } catch (e) {
      throw ModelLoadException('Failed to clear cache: $e');
    }
  }

  /// Get the current cache size in bytes
  ///
  /// Returns the total size of all cached models.
  ///
  /// Example:
  /// ```dart
  /// final sizeBytes = await InferenceSession.getCacheSize();
  /// final sizeMB = sizeBytes / (1024 * 1024);
  /// print('Cache size: ${sizeMB.toStringAsFixed(1)} MB');
  /// ```
  static Future<int> getCacheSize() async {
    try {
      final size = await rust_api.getCacheSize();
      return size.toInt();
    } catch (e) {
      throw ModelLoadException('Failed to get cache size: $e');
    }
  }

  /// Train a Linfa model on-device
  ///
  /// Creates a new model using classical ML algorithms from the Linfa library.
  /// Supports various algorithms like k-means, linear regression, SVM, etc.
  ///
  /// Example:
  /// ```dart
  /// final model = await InferenceSession.trainLinfa(
  ///   data: [[1.0, 2.0], [3.0, 4.0]],
  ///   algorithm: 'kmeans',
  ///   params: {'n_clusters': 2},
  /// );
  /// ```
  static Future<LinfaSession> trainLinfa({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  }) async {
    try {
      // Convert data to Float64List format for Linfa (which requires f64)
      final features = data.map((row) => Float64List.fromList(row)).toList();

      // Convert params to string map
      final stringParams = params?.map((k, v) => MapEntry(k, v.toString())) ??
          <String, String>{};

      final sessionInfo = await rust_api.trainLinfaModel(
        features: features,
        algorithm: algorithm,
        params: stringParams,
      );

      return LinfaSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException('Failed to train Linfa model: $e');
    }
  }

  /// Make a prediction with the loaded model
  ///
  /// Takes an input and returns a prediction result.
  /// The input type must match the model's expected input format.
  ///
  /// Example:
  /// ```dart
  /// final result = await model.predict(input);
  /// ```
  Future<InferenceResult> predict(InferenceInput input) async {
    try {
      final rustInput = input.toRustInput();
      final rustResult = await rust_api.predict(
        sessionHandle: sessionHandle,
        input: rustInput,
      );
      return InferenceResult.fromRustResult(rustResult);
    } catch (e) {
      throw PredictionException('Prediction failed: $e');
    }
  }

  /// Make batch predictions
  ///
  /// Processes multiple inputs in a single call for better performance.
  /// All inputs must have the same format and dimensions.
  ///
  /// Example:
  /// ```dart
  /// final results = await model.predictBatch([input1, input2, input3]);
  /// ```
  Future<List<InferenceResult>> predictBatch(
      List<InferenceInput> inputs) async {
    try {
      final rustInputs = inputs.map((input) => input.toRustInput()).toList();
      final rustResults = await rust_api.predictBatch(
        sessionHandle: sessionHandle,
        inputs: rustInputs,
      );
      return rustResults
          .map((result) => InferenceResult.fromRustResult(result))
          .toList();
    } catch (e) {
      throw PredictionException('Batch prediction failed: $e');
    }
  }

  /// Dispose of the session and free resources
  ///
  /// This method should be called when the session is no longer needed
  /// to prevent memory leaks. After calling dispose(), the session
  /// cannot be used for further predictions.
  ///
  /// Example:
  /// ```dart
  /// model.dispose();
  /// ```
  void dispose() {
    rust_api.disposeSession(sessionHandle: sessionHandle);
  }

  /// Create a session instance from Rust session info
  static InferenceSession _createSessionFromInfo(
      rust_api.SessionInfo sessionInfo) {
    switch (sessionInfo.engineType.toLowerCase()) {
      case 'candle':
        return CandleSession.fromSessionInfo(sessionInfo);

      case 'linfa':
        return LinfaSession.fromSessionInfo(sessionInfo);
      default:
        throw UnsupportedEngineException(
            'Unknown engine type: ${sessionInfo.engineType}');
    }
  }
}
