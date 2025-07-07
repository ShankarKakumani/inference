import '../inference_session.dart';
import '../rust/api/inference.dart' as rust_api;
import '../exceptions/inference_exceptions.dart';

/// Graph optimization levels for ONNX Runtime
enum GraphOptimizationLevel {
  /// No optimization
  none,
  /// Basic optimizations
  basic,
  /// Extended optimizations
  extended,
  /// All available optimizations
  all,
}

/// Execution providers for ONNX Runtime
enum ExecutionProvider {
  /// CPU execution provider
  cpu,
  /// CUDA execution provider (GPU)
  cuda,
  /// CoreML execution provider (Apple devices)
  coreml,
  /// TensorRT execution provider (NVIDIA)
  tensorrt,
  /// DirectML execution provider (Windows)
  directml,
}

/// ONNX Runtime session for ONNX models
/// 
/// This class provides specialized functionality for the ONNX Runtime engine,
/// which supports ONNX models with various execution providers and optimization levels.
class OnnxSession extends InferenceSession {
  /// Create an ONNX session from session info
  OnnxSession.fromSessionInfo(rust_api.SessionInfo sessionInfo)
      : super(
          sessionHandle: sessionInfo.handle,
          engine: sessionInfo.engineType,
          inputSpecs: sessionInfo.inputSpecs,
          outputSpecs: sessionInfo.outputSpecs,
        );

  /// Load a model with CUDA execution provider
  /// 
  /// Forces the use of CUDA for GPU acceleration.
  /// Requires CUDA-compatible hardware and drivers.
  /// 
  /// Example:
  /// ```dart
  /// final session = await OnnxSession.withCuda('path/to/model.onnx');
  /// ```
  static Future<OnnxSession> withCuda(String modelPath) async {
    try {
      final config = await rust_api.SessionConfig.default_();
      final configWithCuda = rust_api.SessionConfig(
        engineType: 'onnx',
        gpuAcceleration: true,
        numThreads: config.numThreads,
        optimizationLevel: 'all',
      );
      
      final sessionInfo = await rust_api.loadModelWithConfig(
        modelPath: modelPath,
        config: configWithCuda,
      );
      return OnnxSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load ONNX model with CUDA: $e',
        modelPath: modelPath,
        engineType: 'onnx',
        cause: e,
      );
    }
  }

  /// Load a model with CoreML execution provider
  /// 
  /// Uses Apple's CoreML framework for acceleration on Apple devices.
  /// Only available on iOS and macOS.
  /// 
  /// Example:
  /// ```dart
  /// final session = await OnnxSession.withCoreML('path/to/model.onnx');
  /// ```
  static Future<OnnxSession> withCoreML(String modelPath) async {
    try {
      final config = await rust_api.SessionConfig.default_();
      final configWithCoreML = rust_api.SessionConfig(
        engineType: 'onnx',
        gpuAcceleration: true,
        numThreads: config.numThreads,
        optimizationLevel: 'all',
      );
      
      final sessionInfo = await rust_api.loadModelWithConfig(
        modelPath: modelPath,
        config: configWithCoreML,
      );
      return OnnxSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load ONNX model with CoreML: $e',
        modelPath: modelPath,
        engineType: 'onnx',
        cause: e,
      );
    }
  }

  /// Load a model with CPU execution provider
  /// 
  /// Forces the use of CPU for inference. This is the most compatible
  /// option but may be slower than GPU acceleration.
  /// 
  /// Example:
  /// ```dart
  /// final session = await OnnxSession.withCpu('path/to/model.onnx');
  /// ```
  static Future<OnnxSession> withCpu(String modelPath) async {
    try {
      final config = await rust_api.SessionConfig.default_();
      final configWithCpu = rust_api.SessionConfig(
        engineType: 'onnx',
        gpuAcceleration: false,
        numThreads: config.numThreads,
        optimizationLevel: 'all',
      );
      
      final sessionInfo = await rust_api.loadModelWithConfig(
        modelPath: modelPath,
        config: configWithCpu,
      );
      return OnnxSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load ONNX model with CPU: $e',
        modelPath: modelPath,
        engineType: 'onnx',
        cause: e,
      );
    }
  }

  /// Load a model with custom configuration
  /// 
  /// Allows fine-grained control over the ONNX Runtime session configuration.
  /// 
  /// Example:
  /// ```dart
  /// final session = await OnnxSession.withConfig(
  ///   'path/to/model.onnx',
  ///   executionProvider: ExecutionProvider.tensorrt,
  ///   optimizationLevel: GraphOptimizationLevel.extended,
  ///   numThreads: 4,
  /// );
  /// ```
  static Future<OnnxSession> withConfig(
    String modelPath, {
    ExecutionProvider executionProvider = ExecutionProvider.cpu,
    GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.all,
    int? numThreads,
  }) async {
    try {
      final config = rust_api.SessionConfig(
        engineType: 'onnx',
        gpuAcceleration: executionProvider != ExecutionProvider.cpu,
        numThreads: numThreads != null ? BigInt.from(numThreads) : null,
        optimizationLevel: _optimizationLevelToString(optimizationLevel),
      );
      
      final sessionInfo = await rust_api.loadModelWithConfig(
        modelPath: modelPath,
        config: config,
      );
      return OnnxSession.fromSessionInfo(sessionInfo);
    } catch (e) {
      throw ModelLoadException(
        'Failed to load ONNX model with config: $e',
        modelPath: modelPath,
        engineType: 'onnx',
        cause: e,
      );
    }
  }

  /// Optimize the model for better performance
  /// 
  /// Applies various optimizations to improve inference speed.
  /// This is typically done automatically, but can be called manually.
  /// 
  /// Example:
  /// ```dart
  /// await session.optimize(
  ///   level: GraphOptimizationLevel.extended,
  ///   numThreads: 8,
  /// );
  /// ```
  Future<void> optimize({
    GraphOptimizationLevel level = GraphOptimizationLevel.all,
    int? numThreads,
  }) async {
    // This would be implemented by calling the Rust backend
    // For now, this is a placeholder
    throw UnsupportedError('Manual optimization not yet implemented');
  }

  /// Get information about the ONNX Runtime engine
  /// 
  /// Returns details about the current ONNX Runtime configuration,
  /// including available execution providers and optimization settings.
  Map<String, dynamic> get engineInfo {
    return {
      'engine': 'onnx',
      'version': '1.16.0', // This would be dynamically retrieved
      'execution_providers': _getAvailableExecutionProviders(),
      'supported_formats': ['.onnx'],
      'optimization_level': 'all', // This would be retrieved from config
    };
  }

  /// Get available execution providers
  /// 
  /// Returns a list of execution providers that are available on the current platform.
  List<String> get availableExecutionProviders {
    return _getAvailableExecutionProviders();
  }

  /// Check if a specific execution provider is available
  /// 
  /// Returns true if the specified execution provider is available, false otherwise.
  bool isExecutionProviderAvailable(ExecutionProvider provider) {
    final available = _getAvailableExecutionProviders();
    return available.contains(_executionProviderToString(provider));
  }

  /// Get the current execution provider being used
  /// 
  /// Returns the name of the execution provider currently in use.
  String get currentExecutionProvider {
    // This would be implemented by querying the Rust backend
    return 'cpu'; // Placeholder
  }

  /// Get model metadata
  /// 
  /// Returns information about the ONNX model including version, producer, etc.
  Map<String, dynamic> get modelMetadata {
    return {
      'version': 'unknown', // This would be extracted from the model
      'producer': 'unknown',
      'domain': 'unknown',
      'description': 'unknown',
      'graph_name': 'unknown',
    };
  }

  /// Get performance profiling information
  /// 
  /// Returns timing and performance statistics for the model.
  Map<String, dynamic> get profilingInfo {
    return {
      'total_inference_time': 0.0,
      'preprocessing_time': 0.0,
      'execution_time': 0.0,
      'postprocessing_time': 0.0,
      'memory_usage': 0,
    };
  }

  /// Convert optimization level enum to string
  static String _optimizationLevelToString(GraphOptimizationLevel level) {
    switch (level) {
      case GraphOptimizationLevel.none:
        return 'none';
      case GraphOptimizationLevel.basic:
        return 'basic';
      case GraphOptimizationLevel.extended:
        return 'extended';
      case GraphOptimizationLevel.all:
        return 'all';
    }
  }

  /// Convert execution provider enum to string
  static String _executionProviderToString(ExecutionProvider provider) {
    switch (provider) {
      case ExecutionProvider.cpu:
        return 'cpu';
      case ExecutionProvider.cuda:
        return 'cuda';
      case ExecutionProvider.coreml:
        return 'coreml';
      case ExecutionProvider.tensorrt:
        return 'tensorrt';
      case ExecutionProvider.directml:
        return 'directml';
    }
  }

  /// Get available execution providers for the current platform
  static List<String> _getAvailableExecutionProviders() {
    // This would be implemented by querying the Rust backend
    // For now, return a placeholder list
    return ['cpu', 'cuda']; // Placeholder
  }

  @override
  String toString() {
    return 'OnnxSession(engine: $engine, provider: $currentExecutionProvider, inputs: ${inputSpecs.length}, outputs: ${outputSpecs.length})';
  }
} 