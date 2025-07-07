import 'dart:math';
import 'dart:typed_data';

import 'rust/api/inference.dart' as rust_api;

/// Result from inference operations
/// 
/// This class wraps the raw tensor output from ML models and provides
/// convenient methods for accessing the data in different formats.
class InferenceResult {
  /// Raw tensor data as a flat array
  final Float32List data;

  /// Shape of the tensor (dimensions)
  final List<int> shape;

  /// Data type of the tensor
  final String dataType;

  /// Create an inference result
  const InferenceResult({
    required this.data,
    required this.shape,
    required this.dataType,
  });

  /// Create from Rust API result
  factory InferenceResult.fromRustResult(rust_api.InferenceResult rustResult) {
    return InferenceResult(
      data: rustResult.data,
      shape: rustResult.shape.map((e) => e.toInt()).toList(),
      dataType: rustResult.dataType,
    );
  }

  /// Get the first element as a scalar value
  /// 
  /// Useful for regression models or single-value outputs.
  /// 
  /// Example:
  /// ```dart
  /// final confidence = result.scalar;
  /// ```
  double get scalar {
    if (data.isEmpty) {
      throw StateError('Cannot get scalar from empty result');
    }
    return data.first;
  }

  /// Get the data as a vector (1D array)
  /// 
  /// Returns the raw data array. Useful for classification probabilities
  /// or feature vectors.
  /// 
  /// Example:
  /// ```dart
  /// final probabilities = result.vector;
  /// ```
  List<double> get vector => data.toList();

  /// Get the data as a 2D matrix
  /// 
  /// Reshapes the flat data array into a 2D matrix based on the tensor shape.
  /// Only works for 2D tensors.
  /// 
  /// Example:
  /// ```dart
  /// final matrix = result.matrix;
  /// ```
  List<List<double>> get matrix {
    if (shape.length != 2) {
      throw StateError('Cannot convert ${shape.length}D tensor to matrix');
    }
    
    final rows = shape[0];
    final cols = shape[1];
    final result = <List<double>>[];
    
    for (int i = 0; i < rows; i++) {
      final row = <double>[];
      for (int j = 0; j < cols; j++) {
        row.add(data[i * cols + j]);
      }
      result.add(row);
    }
    
    return result;
  }

  /// Get the index of the maximum value (argmax)
  /// 
  /// Useful for classification tasks to get the predicted class index.
  /// 
  /// Example:
  /// ```dart
  /// final predictedClass = result.argmax;
  /// ```
  int get argmax {
    if (data.isEmpty) {
      throw StateError('Cannot get argmax from empty result');
    }
    
    int maxIndex = 0;
    double maxValue = data[0];
    
    for (int i = 1; i < data.length; i++) {
      if (data[i] > maxValue) {
        maxValue = data[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  /// Get the top K predictions with their indices and values
  /// 
  /// Returns a list of classification results sorted by confidence in descending order.
  /// Useful for getting the most likely predictions from a classification model.
  /// 
  /// Example:
  /// ```dart
  /// final top5 = result.topK(5);
  /// for (final pred in top5) {
  ///   print('Class ${pred.classIndex}: ${pred.confidence}');
  /// }
  /// ```
  List<ClassificationResult> topK([int k = 5]) {
    if (data.isEmpty) {
      return [];
    }
    
    // Create list of (index, value) pairs
    final indexed = <_IndexedValue>[];
    for (int i = 0; i < data.length; i++) {
      indexed.add(_IndexedValue(i, data[i]));
    }
    
    // Sort by value in descending order
    indexed.sort((a, b) => b.value.compareTo(a.value));
    
    // Take top K and convert to ClassificationResult
    final topK = indexed.take(min(k, indexed.length));
    return topK.map((item) => ClassificationResult(
      classIndex: item.index,
      confidence: item.value,
    )).toList();
  }

  /// Get the top K predictions with softmax normalization
  /// 
  /// Applies softmax to convert raw logits to probabilities before selecting top K.
  /// 
  /// Example:
  /// ```dart
  /// final top3 = result.topKSoftmax(3);
  /// ```
  List<ClassificationResult> topKSoftmax([int k = 5]) {
    if (data.isEmpty) {
      return [];
    }
    
    // Apply softmax
    final softmaxData = _softmax(data);
    
    // Create temporary result with softmax data
    final softmaxResult = InferenceResult(
      data: Float32List.fromList(softmaxData),
      shape: shape,
      dataType: dataType,
    );
    
    return softmaxResult.topK(k);
  }

  /// Apply softmax to convert logits to probabilities
  List<double> _softmax(Float32List logits) {
    // Find max for numerical stability
    double maxLogit = logits.reduce(max);
    
    // Compute exp(x - max) for each element
    final expValues = logits.map((x) => exp(x - maxLogit)).toList();
    
    // Compute sum of exponentials
    final sumExp = expValues.reduce((a, b) => a + b);
    
    // Normalize
    return expValues.map((x) => x / sumExp).toList();
  }

  /// Get the total number of elements in the tensor
  int get size => data.length;

  /// Check if the result is empty
  bool get isEmpty => data.isEmpty;

  /// Get a string representation of the tensor shape
  String get shapeString => '[${shape.join(', ')}]';

  @override
  String toString() {
    return 'InferenceResult(shape: $shapeString, size: $size, dataType: $dataType)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is InferenceResult &&
          runtimeType == other.runtimeType &&
          data == other.data &&
          shape == other.shape &&
          dataType == other.dataType;

  @override
  int get hashCode => data.hashCode ^ shape.hashCode ^ dataType.hashCode;
}

/// Represents a classification result with class index and confidence
class ClassificationResult {
  /// The index of the predicted class
  final int classIndex;

  /// The confidence score for this prediction
  final double confidence;

  /// Optional class name (if available)
  final String? className;

  const ClassificationResult({
    required this.classIndex,
    required this.confidence,
    this.className,
  });

  @override
  String toString() {
    final name = className ?? 'Class $classIndex';
    return '$name: ${(confidence * 100).toStringAsFixed(2)}%';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ClassificationResult &&
          runtimeType == other.runtimeType &&
          classIndex == other.classIndex &&
          confidence == other.confidence &&
          className == other.className;

  @override
  int get hashCode => classIndex.hashCode ^ confidence.hashCode ^ className.hashCode;
}

/// Helper class for sorting indexed values
class _IndexedValue {
  final int index;
  final double value;

  _IndexedValue(this.index, this.value);
} 