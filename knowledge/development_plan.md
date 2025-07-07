# Inference Package - Development Plan

## Overview
This document outlines the step-by-step development plan for implementing the Inference Flutter package - a zero-setup ML inference library that bridges Flutter with the Rust ML ecosystem (Candle, ORT, Linfa).

## Development Strategy

### Core Principles
- **Start Simple**: Get basic structure working before adding complexity
- **Test Early**: Validate each engine independently before integration
- **Follow BRD Exactly**: Stick to the exact API design and dependencies specified in `inference_brd.md`
- **Incremental**: Each phase builds on the previous one
- **Zero-Setup Promise**: Maintain the 1-command install, 3-line usage goal

### Key Milestone
After Phase 4, we should have a working (but basic) inference package that can load a simple model and make predictions.

## Phase 1: Foundation Setup
**Goal**: Get the basic project structure and dependencies set up

### Task 1: Initialize FRB Project
- **Action**: Use `flutter_rust_bridge_codegen create-integration` or `init`
- **Output**: Basic Flutter package structure with Rust backend
- **Validation**: Project builds successfully on host platform

### Task 2: Setup Rust Dependencies
- **Action**: Configure `Cargo.toml` with exact dependencies from BRD
- **Key Dependencies**:
  - `candle-core = { version = "0.8", features = ["cuda", "mkl"] }`
  - `ort = { version = "2.0.0-rc.9", features = ["load-dynamic", "cuda", "tensorrt"] }`
  - `linfa = "0.7"` (with algorithm crates)
  - `flutter_rust_bridge = "2.0"`
- **Output**: Rust project compiles with all ML dependencies

### Task 3: Setup Flutter Dependencies
- **Action**: Configure `pubspec.yaml` with Flutter dependencies and FFI plugin setup
- **Key Config**: FFI plugin setup for all platforms (Android, iOS, Windows, macOS, Linux)
- **Output**: Flutter package structure ready for development

## Phase 2: Core Architecture
**Goal**: Build the foundational abstractions

### Task 4: Implement Core Traits
- **Action**: Create `InferenceEngine` and `Model` traits in `engines/mod.rs`
- **Requirements**: 
  - Both traits must be `Send + Sync`
  - Async methods for model loading and prediction
  - Format support detection
- **Output**: Trait definitions that all engines will implement

### Task 5: Implement Tensor System
- **Action**: Build tensor data structures and conversion utilities
- **Components**:
  - `models/tensor.rs` - Core tensor operations
  - `utils/converters.rs` - Conversions between engine formats
  - Bridge between Rust tensors and Flutter data
- **Output**: Unified tensor interface across all engines

### Task 6: Implement Model Detector
- **Action**: Create auto-detection logic in `utils/model_detector.rs`
- **Logic**:
  - File extension mapping (.onnx → ORT, .safetensors → Candle)
  - Content-based detection (magic bytes)
  - Default fallback to ORT
- **Output**: Automatic engine selection working

## Phase 3: Engine Implementations
**Goal**: Implement the three ML engines

### Task 7: Implement Candle Engine
- **Action**: Build Candle engine in `engines/candle_engine.rs`
- **Features**:
  - SafeTensors and PyTorch model loading
  - GPU acceleration support
  - HuggingFace integration hooks
- **Output**: Candle engine can load and run PyTorch models

### Task 8: Implement ORT Engine
- **Action**: Build ONNX Runtime engine in `engines/ort_engine.rs`
- **Features**:
  - SessionBuilder configuration
  - Multiple execution providers (CUDA, CoreML, CPU)
  - Graph optimization
- **Output**: ORT engine can load and run ONNX models

### Task 9: Implement Linfa Engine
- **Action**: Build Linfa engine in `engines/linfa_engine.rs`
- **Features**:
  - K-means clustering
  - Linear regression
  - SVM algorithms
  - Model serialization/deserialization
- **Output**: Linfa engine can train and predict with classical ML

## Phase 4: Bridge Integration
**Goal**: Connect Rust and Flutter

### Task 10: Implement Rust API
- **Action**: Create `api.rs` with FRB annotations
- **Components**:
  - Session management functions
  - Input/output data structures
  - Error handling across FFI boundary
- **Output**: Complete Rust API ready for code generation

### Task 11: Generate Bridge Code
- **Action**: Run FRB codegen to generate bridge files
- **Process**: `flutter_rust_bridge_codegen generate`
- **Output**: `bridge_generated.rs` and `bridge_generated.dart` created

## Phase 5: Flutter API
**Goal**: Create the developer-facing API

### Task 12: Implement Flutter API
- **Action**: Create main Flutter classes
- **Components**:
  - `InferenceSession` - Main user interface
  - `InferenceResult` - Result wrapper with convenience methods
  - Engine-specific session classes
- **Output**: Core Flutter API matching BRD specification

### Task 13: Implement Input Types
- **Action**: Build complete input type hierarchy
- **Components**:
  - `ImageInput` - Image preprocessing and convenience constructors
  - `NLPInput` - NLP input with tokenizer support
  - `TensorInput` - Raw tensor data with shape validation
  - `AudioInput` - Audio sample handling
- **Output**: All input types with `fromFile`, `fromAsset`, `fromBytes` constructors

### Task 14: Implement Error Handling
- **Action**: Create comprehensive error handling system
- **Components**:
  - Dart exception hierarchy
  - Proper error propagation from Rust
  - User-friendly error messages
- **Output**: Robust error handling across the entire stack

## Phase 6: Validation
**Goal**: Prove it works

### Task 15: Create Basic Example
- **Action**: Build simple 3-line usage example
- **Code**:
  ```dart
  final model = await Inference.load('assets/model.onnx');
  final result = await model.predict(inputData);
  print(result); // Done!
  ```
- **Output**: Basic usage example works end-to-end

### Task 16: Test Cross-Platform
- **Action**: Test package builds and runs on all platforms
- **Platforms**: Android, iOS, Windows, macOS, Linux
- **Output**: Package works on all supported Flutter platforms

## Implementation Notes

### Dependencies Management
- Use exact versions specified in BRD
- Feature flags for optional GPU support
- Platform-specific configurations in `.cargo/config.toml`

### Build System
- `build.rs` for FRB code generation
- Proper crate types: `["staticlib", "cdylib"]`
- Platform-specific native library loading

### Testing Strategy
- Unit tests for each engine independently
- Integration tests for the complete pipeline
- Example apps for each major use case

### Performance Considerations
- Async operations to prevent UI blocking
- Proper memory management with `dispose()` methods
- GPU acceleration where available
- Model caching for repeated loads

## Success Criteria

### Technical
- [ ] All engines load and run models correctly
- [ ] Auto-detection works reliably
- [ ] Cross-platform compatibility confirmed
- [ ] Memory management prevents leaks
- [ ] Error handling is comprehensive

### User Experience
- [ ] Installation with single `flutter pub add inference`
- [ ] Basic usage in 3 lines of code
- [ ] Auto-detection "just works"
- [ ] Clear error messages for troubleshooting
- [ ] Good performance on mobile devices

### Code Quality
- [ ] Follows BRD API specification exactly
- [ ] Comprehensive documentation
- [ ] Example code for all major use cases
- [ ] Proper error handling throughout
- [ ] Clean, maintainable code structure

## Risk Mitigation

### Technical Risks
- **Large Bundle Size**: Use feature flags to include only needed engines
- **Build Complexity**: Comprehensive CI/CD for all platforms
- **Version Conflicts**: Lock dependency versions, test thoroughly
- **Performance Issues**: Profile and optimize critical paths

### Development Risks
- **Scope Creep**: Stick to BRD specification strictly
- **Platform Issues**: Test early and often on all platforms
- **API Changes**: Version lock all dependencies
- **Integration Problems**: Validate each phase before proceeding

## Next Steps

1. **Start with Task 1**: Initialize the FRB project structure
2. **Validate Each Phase**: Don't proceed until current phase is solid
3. **Document Progress**: Update this plan as we learn and adapt
4. **Test Continuously**: Validate functionality at each step

This plan provides a clear roadmap from empty directory to fully functional ML inference package. Each task builds on the previous ones, ensuring we maintain momentum while building a solid foundation. 