# Inference Package - Phase Prompts

## Instructions for Use
1. Copy the prompt for the current phase
2. Paste it into a new chat (or continue current chat)
3. Let AI complete the phase
4. When done, say "move on to next phase" and use the next prompt

---

## Phase 1: Foundation Setup

### Phase 1 Start Prompt
```
Start Phase 1 implementation following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Tasks to complete:
1. Initialize FRB project structure using flutter_rust_bridge_codegen commands
2. Setup Rust dependencies with exact versions from BRD
3. Setup Flutter dependencies and FFI plugin configuration

Requirements:
- Use FRB 2.0 commands, not manual creation
- Follow exact dependency versions from BRD
- Configure FFI plugin for all platforms (Android, iOS, Windows, macOS, Linux)
- Validate project builds successfully

Update @knowledge/project_context.md with progress when complete.
```

---

## Phase 2: Core Architecture

### Phase 2 Start Prompt
```
Start Phase 2 implementation following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Tasks to complete:
1. Implement InferenceEngine and Model traits in engines/mod.rs
2. Build tensor system with conversion utilities
3. Create model detector for auto-engine selection

Requirements:
- Both traits must be Send + Sync
- Async methods for model loading and prediction
- Unified tensor interface across all engines
- Auto-detection based on file extensions and content magic bytes
- Support .onnx → ORT, .safetensors/.pt/.pth → Candle

Update @knowledge/project_context.md with progress when complete.
```

---

## Phase 3: Engine Implementations

### Phase 3A: Candle Engine Prompt
```
Implement Candle Engine following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Complete Candle engine implementation in engines/candle_engine.rs

Requirements:
- CandleEngine struct implementing InferenceEngine trait
- Support for .safetensors, .pt, .pth files
- GPU acceleration with CUDA detection
- Device management (CPU/CUDA)
- HuggingFace integration hooks
- Proper error handling with InferenceError
- Use exact candle-core 0.8 dependencies from BRD

Update @knowledge/project_context.md with progress when complete.
```

### Phase 3B: ORT Engine Prompt
```
Implement ORT Engine following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Complete ONNX Runtime engine implementation in engines/ort_engine.rs

Requirements:
- OrtEngine struct implementing InferenceEngine trait
- SessionBuilder configuration with optimization
- Multiple execution providers (CUDA, CoreML, CPU)
- Graph optimization levels
- Thread configuration
- Proper error handling with InferenceError
- Use exact ort 2.0.0-rc.9 dependencies from BRD

Update @knowledge/project_context.md with progress when complete.
```

### Phase 3C: Linfa Engine Prompt
```
Implement Linfa Engine following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Complete Linfa engine implementation in engines/linfa_engine.rs

Requirements:
- LinfaEngine struct implementing InferenceEngine trait
- K-means clustering algorithm
- Linear regression algorithm
- SVM algorithm support
- Model serialization/deserialization
- Dataset creation and management
- Proper error handling with InferenceError
- Use exact linfa 0.7 dependencies from BRD

Update @knowledge/project_context.md with progress when complete.
```

---

## Phase 4: Bridge Integration

### Phase 4 Start Prompt
```
Start Phase 4 implementation following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Tasks to complete:
1. Create api.rs with FRB annotations for Flutter bridge
2. Run FRB codegen to generate bridge files

Requirements:
- Complete Rust API with proper FRB annotations
- Session management functions
- Input/output data structures for FFI
- Error handling across FFI boundary
- Generate bridge_generated.rs and bridge_generated.dart
- Validate FFI communication works

Update @knowledge/project_context.md with progress when complete.
```

---

## Phase 5: Flutter API

### Phase 5A: Core Flutter API Prompt
```
Implement Core Flutter API following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Tasks to complete:
1. Create InferenceSession main interface
2. Implement InferenceResult with convenience methods
3. Create engine-specific session classes

Requirements:
- InferenceSession with static factory methods (load, loadWithCandle, loadWithOnnx, trainLinfa)
- Prediction methods (predict, predictBatch)
- Properties (inputSpecs, outputSpecs, engine)
- Resource management (dispose)
- InferenceResult with scalar, vector, matrix accessors
- Classification helpers (argmax, topK)
- Engine-specific classes (CandleSession, OnnxSession, LinfaSession)

Update @knowledge/project_context.md with progress when complete.
```

### Phase 5B: Input Types Prompt
```
Implement Input Types following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Build complete input type hierarchy

Requirements:
- Abstract InferenceInput base class
- ImageInput with convenience constructors (fromFile, fromAsset, fromBytes)
- NLPInput with tokenizer support
- TensorInput with shape validation and type-safe constructors
- AudioInput with sample rate handling
- All input types must extend InferenceInput
- Proper error handling for invalid inputs

Update @knowledge/project_context.md with progress when complete.
```

### Phase 5C: Error Handling Prompt
```
Implement Error Handling following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Create comprehensive error handling system

Requirements:
- Dart exception hierarchy extending InferenceException
- ModelLoadException, PredictionException, UnsupportedFormatException
- Proper error propagation from Rust to Flutter
- User-friendly error messages
- Structured error types matching Rust InferenceError
- Error handling for all edge cases

Update @knowledge/project_context.md with progress when complete.
```

---

## Phase 6: Validation

### Phase 6A: Basic Example Prompt
```
Create Basic Example following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Build and validate 3-line usage example

Requirements:
- Create simple example demonstrating:
  final model = await Inference.load('assets/model.onnx');
  final result = await model.predict(inputData);
  print(result);
- Test auto-detection works
- Validate all input types work
- Test error handling
- Ensure zero-setup promise is met

Update @knowledge/project_context.md with progress when complete.
```

### Phase 6B: Cross-Platform Testing Prompt
```
Implement Cross-Platform Testing following @knowledge/development_plan.md and @knowledge/inference_brd.md.

Task: Test package builds and runs on all platforms

Requirements:
- Test compilation on Android, iOS, Windows, macOS, Linux
- Validate FFI plugin works on all platforms
- Test native library loading
- Verify GPU acceleration where available
- Test model loading and prediction on each platform
- Fix any platform-specific issues

Update @knowledge/project_context.md with progress when complete.
```

---

## Quick Reference Commands

### Continue Current Phase
```
Continue current phase implementation following @knowledge/project_context.md current status.
```

### Move to Next Phase
```
Move on to next phase.
```

### Check Status
```
Check current implementation status from @knowledge/project_context.md and summarize progress.
```

### Emergency Recovery
```
Recover project state from @knowledge/project_context.md and continue from last completed task.
```

### Final Validation
```
Perform final validation of complete Inference package:
- Test all engines work correctly
- Validate 3-line usage example
- Test cross-platform compatibility
- Verify zero-setup promise is met
- Create final package ready for publication
```

---

## Usage Notes

1. **Copy exact prompts** - they contain all necessary context
2. **One prompt per phase** - don't mix phases
3. **Wait for completion** - let AI finish before moving to next phase
4. **Check progress** - AI will update project_context.md automatically
5. **Use "move on to next phase"** - AI will know what comes next

Each prompt is self-contained with all requirements and context needed for successful implementation. 