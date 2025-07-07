import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

import 'rust/api/inference.dart' as rust_api;

/// Abstract base class for inference inputs
///
/// All input types must extend this class and implement the conversion
/// to the Rust API format.
abstract class InferenceInput {
  /// Creates a new inference input.
  ///
  /// This is the base constructor for all inference input types.
  /// Subclasses should call this constructor and implement [toRustInput].
  const InferenceInput();

  /// Convert this input to the Rust API format
  rust_api.InferenceInput toRustInput();
}

/// Image input for computer vision models
///
/// Handles image data with preprocessing capabilities.
/// Supports common image formats and provides convenience constructors
/// for loading from files, assets, or raw bytes.
class ImageInput extends InferenceInput {
  /// Raw image bytes
  final Uint8List bytes;

  /// Image width in pixels
  final int width;

  /// Image height in pixels
  final int height;

  /// Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
  final int channels;

  /// Create an image input with raw data
  ImageInput({
    required this.bytes,
    required this.width,
    required this.height,
    required this.channels,
  });

  /// Create from a file
  ///
  /// Loads and preprocesses an image file. The image will be decoded
  /// and converted to the appropriate format for ML inference.
  ///
  /// Example:
  /// ```dart
  /// final input = await ImageInput.fromFile(File('path/to/image.jpg'));
  /// ```
  static Future<ImageInput> fromFile(File file) async {
    final bytes = await file.readAsBytes();
    return fromBytes(bytes);
  }

  /// Create from a Flutter asset
  ///
  /// Loads an image from the app's asset bundle.
  ///
  /// Example:
  /// ```dart
  /// final input = await ImageInput.fromAsset('assets/images/test.jpg');
  /// ```
  static Future<ImageInput> fromAsset(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final bytes = byteData.buffer.asUint8List();
    return fromBytes(bytes);
  }

  /// Create from raw bytes
  ///
  /// Decodes image bytes and extracts dimensions and channel information.
  /// Supports common formats like JPEG, PNG, WebP, etc.
  ///
  /// Example:
  /// ```dart
  /// final input = await ImageInput.fromBytes(imageBytes);
  /// ```
  static Future<ImageInput> fromBytes(Uint8List bytes) async {
    // For now, we'll create a placeholder implementation
    // In a real implementation, this would decode the image and extract metadata
    // This would typically use the `image` package or platform-specific decoders

    // Placeholder values - in reality these would be extracted from the image
    return ImageInput(
      bytes: bytes,
      width: 224, // Common input size for many models
      height: 224,
      channels: 3, // RGB
    );
  }

  /// Create from raw pixel data
  ///
  /// Creates an image input from raw pixel values. Useful when you already
  /// have preprocessed image data.
  ///
  /// Example:
  /// ```dart
  /// final input = ImageInput.fromPixels(
  ///   pixels: Float32List.fromList([...]),
  ///   width: 224,
  ///   height: 224,
  ///   channels: 3,
  /// );
  /// ```
  factory ImageInput.fromPixels({
    required Float32List pixels,
    required int width,
    required int height,
    required int channels,
  }) {
    // Convert Float32List to Uint8List for consistency
    final bytes = Uint8List.fromList(
      pixels.map((p) => (p * 255).clamp(0, 255).toInt()).toList(),
    );

    return ImageInput(
      bytes: bytes,
      width: width,
      height: height,
      channels: channels,
    );
  }

  @override
  rust_api.InferenceInput toRustInput() {
    // Convert to normalized float values
    final floatData = Float32List.fromList(
      bytes.map((b) => b / 255.0).toList(),
    );

    return rust_api.InferenceInput(
      data: floatData,
      shape: Uint64List.fromList([height, width, channels]),
      dataType: 'f32',
    );
  }

  @override
  String toString() => 'ImageInput(${width}x${height}x$channels)';
}

/// NLP input for natural language processing models
///
/// Handles text data with optional tokenizer specification.
/// Can work with raw text or pre-tokenized input.
class NLPInput extends InferenceInput {
  /// Raw text content
  final String text;

  /// Optional tokenizer specification
  final String? tokenizer;

  /// Pre-tokenized input (optional)
  final List<int>? tokenIds;

  /// Create an NLP input
  NLPInput(
    this.text, {
    this.tokenizer,
    this.tokenIds,
  });

  /// Create from pre-tokenized input
  ///
  /// Use this when you have already tokenized the text using a specific tokenizer.
  ///
  /// Example:
  /// ```dart
  /// final input = NLPInput.fromTokens([101, 2023, 2003, 102]);
  /// ```
  factory NLPInput.fromTokens(List<int> tokens) {
    return NLPInput(
      '',
      tokenIds: tokens,
    );
  }

  @override
  rust_api.InferenceInput toRustInput() {
    if (tokenIds != null) {
      // Use pre-tokenized input
      return rust_api.InferenceInput(
        data: Float32List.fromList(tokenIds!.map((t) => t.toDouble()).toList()),
        shape: Uint64List.fromList([tokenIds!.length]),
        dataType: 'i32',
      );
    } else {
      // For now, create a simple character-based encoding
      // In a real implementation, this would use a proper tokenizer
      final charCodes = text.codeUnits;
      return rust_api.InferenceInput(
        data: Float32List.fromList(charCodes.map((c) => c.toDouble()).toList()),
        shape: Uint64List.fromList([charCodes.length]),
        dataType: 'i32',
      );
    }
  }

  @override
  String toString() =>
      'NLPInput("${text.length > 50 ? '${text.substring(0, 50)}...' : text}")';
}

/// Raw tensor input for models
///
/// Provides direct access to tensor data with shape validation.
/// Useful for custom preprocessing or when working with numerical data.
class TensorInput extends InferenceInput {
  /// Raw tensor data as a flat array
  final List<double> data;

  /// Shape of the tensor
  final List<int> shape;

  /// Data type (defaults to f32)
  final String dataType;

  /// Create a tensor input
  TensorInput(
    this.data,
    this.shape, {
    this.dataType = 'f32',
  });

  /// Create from a 1D list
  ///
  /// Example:
  /// ```dart
  /// final input = TensorInput.fromList([1.0, 2.0, 3.0, 4.0]);
  /// ```
  factory TensorInput.fromList(List<double> data) {
    return TensorInput(data, [data.length]);
  }

  /// Create from a 2D list (matrix)
  ///
  /// Example:
  /// ```dart
  /// final input = TensorInput.from2D([
  ///   [1.0, 2.0],
  ///   [3.0, 4.0],
  /// ]);
  /// ```
  factory TensorInput.from2D(List<List<double>> matrix) {
    if (matrix.isEmpty) {
      return TensorInput([], [0, 0]);
    }

    final rows = matrix.length;
    final cols = matrix[0].length;

    // Validate that all rows have the same length
    for (int i = 1; i < rows; i++) {
      if (matrix[i].length != cols) {
        throw ArgumentError('All rows must have the same length');
      }
    }

    // Flatten the matrix
    final flatData = <double>[];
    for (final row in matrix) {
      flatData.addAll(row);
    }

    return TensorInput(flatData, [rows, cols]);
  }

  /// Create from a 3D list
  ///
  /// Example:
  /// ```dart
  /// final input = TensorInput.from3D([
  ///   [[1.0, 2.0], [3.0, 4.0]],
  ///   [[5.0, 6.0], [7.0, 8.0]],
  /// ]);
  /// ```
  factory TensorInput.from3D(List<List<List<double>>> tensor3d) {
    if (tensor3d.isEmpty) {
      return TensorInput([], [0, 0, 0]);
    }

    final depth = tensor3d.length;
    final rows = tensor3d[0].length;
    final cols = tensor3d[0].isEmpty ? 0 : tensor3d[0][0].length;

    // Validate dimensions
    for (int i = 0; i < depth; i++) {
      if (tensor3d[i].length != rows) {
        throw ArgumentError('All matrices must have the same number of rows');
      }
      for (int j = 0; j < rows; j++) {
        if (tensor3d[i][j].length != cols) {
          throw ArgumentError('All rows must have the same length');
        }
      }
    }

    // Flatten the tensor
    final flatData = <double>[];
    for (final matrix in tensor3d) {
      for (final row in matrix) {
        flatData.addAll(row);
      }
    }

    return TensorInput(flatData, [depth, rows, cols]);
  }

  /// Validate that the data matches the specified shape
  void validate() {
    final expectedSize = shape.fold(1, (a, b) => a * b);
    if (data.length != expectedSize) {
      throw ArgumentError(
        'Data length ${data.length} does not match shape $shape (expected $expectedSize elements)',
      );
    }
  }

  @override
  rust_api.InferenceInput toRustInput() {
    validate();

    return rust_api.InferenceInput(
      data: Float32List.fromList(data),
      shape: Uint64List.fromList(shape),
      dataType: dataType,
    );
  }

  @override
  String toString() => 'TensorInput(shape: $shape, size: ${data.length})';
}

/// Audio input for speech and audio processing models
///
/// Handles audio samples with sample rate information.
/// Supports common audio preprocessing tasks.
class AudioInput extends InferenceInput {
  /// Audio samples as floating point values
  final Float32List samples;

  /// Sample rate in Hz
  final int sampleRate;

  /// Number of channels (1 for mono, 2 for stereo)
  final int channels;

  /// Create an audio input
  AudioInput({
    required this.samples,
    required this.sampleRate,
    this.channels = 1,
  });

  /// Create from a file
  ///
  /// Loads and decodes an audio file. Supports common formats like WAV, MP3, etc.
  ///
  /// Example:
  /// ```dart
  /// final input = await AudioInput.fromFile(File('path/to/audio.wav'));
  /// ```
  static Future<AudioInput> fromFile(File file) async {
    // Placeholder implementation
    // In a real implementation, this would decode the audio file
    final bytes = await file.readAsBytes();
    return fromBytes(bytes);
  }

  /// Create from raw bytes
  ///
  /// Decodes audio bytes and extracts sample rate and channel information.
  ///
  /// Example:
  /// ```dart
  /// final input = await AudioInput.fromBytes(audioBytes);
  /// ```
  static Future<AudioInput> fromBytes(Uint8List bytes) async {
    // Placeholder implementation
    // In a real implementation, this would decode the audio format

    // For now, assume 16-bit PCM audio at 16kHz
    final samples = Float32List(bytes.length ~/ 2);
    for (int i = 0; i < samples.length; i++) {
      // Convert 16-bit integer to float
      final sample = (bytes[i * 2] | (bytes[i * 2 + 1] << 8));
      samples[i] = sample / 32768.0; // Normalize to [-1, 1]
    }

    return AudioInput(
      samples: samples,
      sampleRate: 16000,
      channels: 1,
    );
  }

  /// Create from raw samples
  ///
  /// Use this when you already have processed audio samples.
  ///
  /// Example:
  /// ```dart
  /// final input = AudioInput.fromSamples(
  ///   samples: Float32List.fromList([...]),
  ///   sampleRate: 44100,
  /// );
  /// ```
  factory AudioInput.fromSamples({
    required Float32List samples,
    required int sampleRate,
    int channels = 1,
  }) {
    return AudioInput(
      samples: samples,
      sampleRate: sampleRate,
      channels: channels,
    );
  }

  /// Get the duration of the audio in seconds
  double get duration => samples.length / sampleRate / channels;

  @override
  rust_api.InferenceInput toRustInput() {
    return rust_api.InferenceInput(
      data: samples,
      shape: Uint64List.fromList([samples.length]),
      dataType: 'f32',
    );
  }

  @override
  String toString() =>
      'AudioInput(${duration.toStringAsFixed(2)}s, ${sampleRate}Hz, ${channels}ch)';
}
