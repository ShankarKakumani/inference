library;

// Main API (convenience wrapper)
export 'src/inference.dart';

// Core API
export 'src/inference_session.dart';
export 'src/inference_result.dart';
export 'src/inference_input.dart';

// Engine-specific sessions
export 'src/engines/candle_session.dart';
export 'src/engines/onnx_session.dart';
export 'src/engines/linfa_session.dart';

// Exceptions
export 'src/exceptions/inference_exceptions.dart';

// Rust bridge (for advanced usage)
export 'src/rust/api/inference.dart' show SessionConfig, getAvailableEngines, isEngineAvailable, detectEngineFromPath, detectEngineFromBytes;
export 'src/rust/models/tensor.dart';
export 'src/rust/frb_generated.dart' show RustLib;

// Re-export commonly used types for convenience
export 'src/engines/onnx_session.dart' show GraphOptimizationLevel, ExecutionProvider;
export 'src/engines/linfa_session.dart' show LinfaAlgorithm;
