import 'inference_session.dart';
import 'engines/candle_session.dart';
import 'engines/linfa_session.dart';

/// Main entry point for the Inference package
/// 
/// This class provides the exact API specified in the BRD with static methods
/// for loading models and training algorithms. It serves as a convenience
/// wrapper around the more specific session classes.
class Inference {
  // Private constructor to prevent instantiation
  Inference._();

  /// Load a model with automatic engine detection
  /// 
  /// This method automatically detects the appropriate engine based on
  /// the model file format and loads it accordingly.
  /// 
  /// Example:
  /// ```dart
  /// final model = await Inference.load('assets/model.safetensors');
  /// final result = await model.predict(inputData);
  /// print(result); // Done!
  /// ```
  static Future<InferenceSession> load(String modelPath) {
    return InferenceSession.load(modelPath);
  }

  /// Load a model with explicit Candle engine
  /// 
  /// Forces the use of the Candle engine for PyTorch models.
  /// Supports .safetensors, .pt, and .pth files.
  /// 
  /// Example:
  /// ```dart
  /// final model = await Inference.loadWithCandle('assets/model.safetensors');
  /// ```
  static Future<CandleSession> loadWithCandle(String modelPath) {
    return InferenceSession.loadWithCandle(modelPath);
  }


  /// Train a Linfa model on-device
  /// 
  /// Creates a new model using classical ML algorithms from the Linfa library.
  /// Supports various algorithms like k-means, linear regression, SVM, etc.
  /// 
  /// Example:
  /// ```dart
  /// final model = await Inference.trainLinfa(
  ///   data: [[1.0, 2.0], [3.0, 4.0]],
  ///   algorithm: 'kmeans',
  ///   params: {'n_clusters': 2},
  /// );
  /// ```
  static Future<LinfaSession> trainLinfa({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  }) {
    return InferenceSession.trainLinfa(
      data: data,
      algorithm: algorithm,
      params: params,
    );
  }
} 