/// Base exception class for all inference-related errors
/// 
/// All inference exceptions extend this class to provide consistent
/// error handling across the package.
abstract class InferenceException implements Exception {
  /// The error message
  final String message;

  /// Optional underlying cause
  final Object? cause;

  /// Stack trace where the error occurred
  final StackTrace? stackTrace;

  const InferenceException(
    this.message, {
    this.cause,
    this.stackTrace,
  });

  @override
  String toString() {
    final buffer = StringBuffer('$runtimeType: $message');
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when model loading fails
/// 
/// This can occur due to various reasons:
/// - Invalid model file format
/// - Corrupted model file
/// - Unsupported model architecture
/// - Missing dependencies
/// - File system errors
class ModelLoadException extends InferenceException {
  /// The path to the model that failed to load
  final String? modelPath;

  /// The engine that was attempted to be used
  final String? engineType;

  const ModelLoadException(
    String message, {
    this.modelPath,
    this.engineType,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  @override
  String toString() {
    final buffer = StringBuffer('ModelLoadException: $message');
    if (modelPath != null) {
      buffer.write(' (path: $modelPath)');
    }
    if (engineType != null) {
      buffer.write(' (engine: $engineType)');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when prediction fails
/// 
/// This can occur due to:
/// - Invalid input data
/// - Input shape mismatch
/// - Runtime errors in the model
/// - Memory allocation failures
/// - Hardware acceleration issues
class PredictionException extends InferenceException {
  /// The input that caused the prediction to fail
  final String? inputDescription;

  /// The expected input shape (if known)
  final List<int>? expectedShape;

  /// The actual input shape that was provided
  final List<int>? actualShape;

  const PredictionException(
    String message, {
    this.inputDescription,
    this.expectedShape,
    this.actualShape,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  @override
  String toString() {
    final buffer = StringBuffer('PredictionException: $message');
    if (inputDescription != null) {
      buffer.write(' (input: $inputDescription)');
    }
    if (expectedShape != null && actualShape != null) {
      buffer.write(' (expected shape: $expectedShape, actual: $actualShape)');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when an unsupported model format is encountered
/// 
/// This occurs when trying to load a model file that is not supported
/// by any of the available engines.
class UnsupportedFormatException extends InferenceException {
  /// The file format that is not supported
  final String format;

  /// The file path (if available)
  final String? filePath;

  /// List of supported formats
  final List<String>? supportedFormats;

  const UnsupportedFormatException(
    this.format, {
    this.filePath,
    this.supportedFormats,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(
          'Unsupported model format: $format',
          cause: cause,
          stackTrace: stackTrace,
        );

  @override
  String toString() {
    final buffer = StringBuffer('UnsupportedFormatException: Unsupported format "$format"');
    if (filePath != null) {
      buffer.write(' (file: $filePath)');
    }
    if (supportedFormats != null && supportedFormats!.isNotEmpty) {
      buffer.write(' (supported formats: ${supportedFormats!.join(', ')})');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when an unsupported or unavailable engine is requested
/// 
/// This occurs when trying to use an engine that is either:
/// - Not compiled into the build
/// - Not available on the current platform
/// - Not properly initialized
class UnsupportedEngineException extends InferenceException {
  /// The engine type that is not supported
  final String engineType;

  /// List of available engines
  final List<String>? availableEngines;

  /// The platform where this error occurred
  final String? platform;

  const UnsupportedEngineException(
    this.engineType, {
    this.availableEngines,
    this.platform,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(
          'Unsupported or unavailable engine: $engineType',
          cause: cause,
          stackTrace: stackTrace,
        );

  @override
  String toString() {
    final buffer = StringBuffer('UnsupportedEngineException: Engine "$engineType" is not available');
    if (platform != null) {
      buffer.write(' (platform: $platform)');
    }
    if (availableEngines != null && availableEngines!.isNotEmpty) {
      buffer.write(' (available engines: ${availableEngines!.join(', ')})');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when input validation fails
/// 
/// This occurs when the provided input data doesn't match the expected format:
/// - Invalid tensor shapes
/// - Wrong data types
/// - Missing required fields
/// - Out of range values
class InputValidationException extends InferenceException {
  /// The field that failed validation
  final String? field;

  /// The expected value or format
  final String? expected;

  /// The actual value that was provided
  final String? actual;

  const InputValidationException(
    String message, {
    this.field,
    this.expected,
    this.actual,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  @override
  String toString() {
    final buffer = StringBuffer('InputValidationException: $message');
    if (field != null) {
      buffer.write(' (field: $field)');
    }
    if (expected != null && actual != null) {
      buffer.write(' (expected: $expected, actual: $actual)');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when resource management fails
/// 
/// This can occur during:
/// - Session disposal
/// - Memory cleanup
/// - Resource allocation
/// - Thread management
class ResourceException extends InferenceException {
  /// The resource that failed
  final String? resource;

  /// The operation that failed
  final String? operation;

  const ResourceException(
    String message, {
    this.resource,
    this.operation,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  @override
  String toString() {
    final buffer = StringBuffer('ResourceException: $message');
    if (resource != null) {
      buffer.write(' (resource: $resource)');
    }
    if (operation != null) {
      buffer.write(' (operation: $operation)');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
}

/// Exception thrown when configuration is invalid
/// 
/// This occurs when:
/// - Invalid configuration parameters
/// - Conflicting settings
/// - Missing required configuration
/// - Platform-specific configuration issues
class ConfigurationException extends InferenceException {
  /// The configuration key that is invalid
  final String? configKey;

  /// The invalid value
  final Object? invalidValue;

  /// Valid options (if applicable)
  final List<String>? validOptions;

  const ConfigurationException(
    String message, {
    this.configKey,
    this.invalidValue,
    this.validOptions,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  @override
  String toString() {
    final buffer = StringBuffer('ConfigurationException: $message');
    if (configKey != null) {
      buffer.write(' (key: $configKey)');
    }
    if (invalidValue != null) {
      buffer.write(' (value: $invalidValue)');
    }
    if (validOptions != null && validOptions!.isNotEmpty) {
      buffer.write(' (valid options: ${validOptions!.join(', ')})');
    }
    if (cause != null) {
      buffer.write(' (caused by: $cause)');
    }
    return buffer.toString();
  }
} 