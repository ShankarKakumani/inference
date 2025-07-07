import '../inference_session.dart';
import '../rust/api/inference.dart' as rust_api;
import '../exceptions/inference_exceptions.dart';

/// Candle engine session for PyTorch models
/// 
/// This class provides specialized functionality for the Candle engine,
/// which supports PyTorch models and HuggingFace integration.
class CandleSession extends InferenceSession {
  /// Create a Candle session from session info
  CandleSession.fromSessionInfo(rust_api.SessionInfo sessionInfo)
      : super(
          sessionHandle: sessionInfo.handle,
          engine: sessionInfo.engineType,
          inputSpecs: sessionInfo.inputSpecs,
          outputSpecs: sessionInfo.outputSpecs,
        );

  /// Load a model from HuggingFace Hub
  /// 
  /// Downloads and loads a model from the HuggingFace model hub.
  /// This is convenient for accessing pre-trained models.
  /// 
  /// Example:
  /// ```dart
  /// final session = await CandleSession.fromHuggingFace(
  ///   repo: 'microsoft/DialoGPT-medium',
  ///   revision: 'main',
  ///   filename: 'pytorch_model.bin',
  /// );
  /// ```
  static Future<CandleSession> fromHuggingFace({
    required String repo,
    String? revision,
    String? filename,
  }) async {
    try {
      final sessionInfo = await rust_api.loadFromHuggingface(
        repo: repo,
        revision: revision,
        filename: filename,
      );
      return CandleSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load model from HuggingFace: $e',
        cause: e,
      );
    }
  }

  /// Load a PyTorch model from SafeTensors format
  /// 
  /// SafeTensors is a secure format for storing tensors that's becoming
  /// the standard for PyTorch models.
  /// 
  /// Example:
  /// ```dart
  /// final session = await CandleSession.fromPyTorch('path/to/model.safetensors');
  /// ```
  static Future<CandleSession> fromPyTorch(String safetensorsPath) async {
    try {
      final sessionInfo = await rust_api.loadModelWithCandle(modelPath: safetensorsPath);
      return CandleSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load PyTorch model: $e',
        modelPath: safetensorsPath,
        engineType: 'candle',
        cause: e,
      );
    }
  }

  /// Load a model from a specific architecture
  /// 
  /// This method allows loading models with known architectures
  /// like ResNet, BERT, GPT-2, etc.
  /// 
  /// Example:
  /// ```dart
  /// final session = await CandleSession.fromArchitecture(
  ///   architecture: 'resnet18',
  ///   weightsPath: 'path/to/weights.safetensors',
  /// );
  /// ```
  static Future<CandleSession> fromArchitecture({
    required String architecture,
    required String weightsPath,
  }) async {
    // For now, this is the same as fromPyTorch
    // In a full implementation, this would handle architecture-specific loading
    return fromPyTorch(weightsPath);
  }

  /// Get information about the Candle engine
  /// 
  /// Returns details about the current Candle engine configuration,
  /// including supported features and available devices.
  Map<String, dynamic> get engineInfo {
    return {
      'engine': 'candle',
      'version': '0.8.0', // This would be dynamically retrieved
      'features': ['cuda', 'mkl'],
      'supported_formats': ['.safetensors', '.pt', '.pth'],
      'device': 'cpu', // This would be dynamically determined
    };
  }

  /// Check if CUDA is available for this session
  /// 
  /// Returns true if the model is running on CUDA, false otherwise.
  bool get isCudaAvailable {
    // This would be implemented by querying the Rust backend
    return false; // Placeholder
  }

  /// Check if MKL (Intel Math Kernel Library) is available
  /// 
  /// Returns true if MKL optimizations are available, false otherwise.
  bool get isMklAvailable {
    // This would be implemented by querying the Rust backend
    return false; // Placeholder
  }

  /// Get the device being used for inference
  /// 
  /// Returns a string describing the device (e.g., 'cpu', 'cuda:0').
  String get device {
    // This would be implemented by querying the Rust backend
    return 'cpu'; // Placeholder
  }

  /// Move the model to a different device
  /// 
  /// This allows switching between CPU and GPU inference.
  /// 
  /// Example:
  /// ```dart
  /// await session.toDevice('cuda:0'); // Move to first GPU
  /// await session.toDevice('cpu');    // Move to CPU
  /// ```
  Future<void> toDevice(String device) async {
    // This would be implemented by calling the Rust backend
    // For now, this is a placeholder
    throw UnsupportedError('Device switching not yet implemented');
  }

  /// Get model statistics
  /// 
  /// Returns information about the model structure and parameters.
  Map<String, dynamic> get modelStats {
    return {
      'parameters': 0, // This would be calculated from the model
      'layers': 0,
      'memory_usage': 0,
      'input_shapes': inputSpecs.map((spec) => spec.shape).toList(),
      'output_shapes': outputSpecs.map((spec) => spec.shape).toList(),
    };
  }

  @override
  String toString() {
    return 'CandleSession(engine: $engine, device: $device, inputs: ${inputSpecs.length}, outputs: ${outputSpecs.length})';
  }
} 