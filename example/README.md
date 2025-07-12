# HuggingFace Integration Demo

This is a Flutter app that demonstrates the HuggingFace integration capabilities of the Inference package. It showcases how to download and use models directly from the HuggingFace Hub.

## Features

- **Real HuggingFace Integration**: Download models directly from HuggingFace Hub using the `hf-hub` Rust crate
- **Automatic Caching**: Models are cached locally after first download for faster subsequent loads
- **Fallback Mechanism**: If `hf-hub` fails, the app falls back to direct URL downloads
- **Multiple Model Types**: Supports both BERT (NLP) and ResNet (Computer Vision) models
- **Real-time Inference**: Run inference on downloaded models with performance metrics

## Supported Models

### BERT (Natural Language Processing)
- **Repository**: `google-bert/bert-base-uncased`
- **Use Case**: Text embeddings and NLP tasks
- **Input**: Text strings
- **Output**: 768-dimensional embeddings

### ResNet (Computer Vision)
- **Repository**: `microsoft/resnet-50`
- **Use Case**: Image classification
- **Input**: Tensor data (demo uses dummy input)
- **Output**: 1000-class logits

## Getting Started

1. **Install Dependencies**:
   ```bash
   flutter pub get
   ```

2. **Run the App**:
   ```bash
   flutter run
   ```

3. **Load Models**:
   - Tap "Load" next to either BERT or ResNet
   - Models will be downloaded from HuggingFace Hub
   - First download may take time depending on your internet connection

4. **Run Inference**:
   - Select a model (BERT or ResNet)
   - For BERT: Enter text or use sample texts
   - For ResNet: Uses dummy tensor input
   - Tap "Run Inference" to see results

## Technical Details

### HuggingFace Integration
The app uses the `hf-hub` Rust crate to download models from HuggingFace Hub. If the download fails (due to network issues), it falls back to direct HTTP downloads.

### Caching
Models are automatically cached after first download using the inference package's built-in caching mechanism.

### Error Handling
The app includes comprehensive error handling with informative error messages for:
- Network connectivity issues
- Missing models
- Authentication problems
- Invalid model formats

## Architecture

```
┌─────────────────────────────────────────┐
│             Flutter UI                  │
│  (HuggingFace Demo Screen)             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         InferenceService                │
│  (State Management & API Calls)        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Inference Package              │
│  (Rust FFI Bridge)                     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            Rust Backend                │
│  (Candle + hf-hub Integration)         │
└─────────────────────────────────────────┘
```

## Performance

The app includes performance metrics showing:
- Model loading times
- Inference execution times
- Memory usage (via model caching)

## Requirements

- Flutter 3.16.0 or later
- Rust toolchain (for building the native library)
- Internet connection (for downloading models)

## Troubleshooting

### Network Issues
If you encounter network connectivity issues:
1. Check your internet connection
2. The app will automatically try fallback download methods
3. Models are cached after successful download

### Model Loading Failures
If models fail to load:
1. Verify the repository names are correct
2. Check if the model files exist on HuggingFace Hub
3. Ensure you have sufficient storage space for caching

### Performance Issues
If inference is slow:
1. Models are loaded once and cached
2. Subsequent inferences should be faster
3. Consider using GPU acceleration if available
