# Inference Example App - Business Requirements Document (BRD)

## 📋 **Project Overview**

### **Purpose**
Create a simple, clean showcase application that demonstrates all capabilities of the Inference Flutter library across iOS and Android platforms.

### **Mission Statement**
"A straightforward demo app that proves the Inference library works flawlessly with real models and real data - nothing fancy, just solid functionality."

---

## 🎯 **Core Principles**

### **Simplicity First**
- Single-platform UI (works on both iOS/Android)
- No desktop-specific features (multi-window, keyboard shortcuts)
- Clean Material Design 3 interface
- Minimal dependencies beyond the inference library

### **Library-Focused**
- The library handles all GPU acceleration, Vulkan, optimization
- Example app just demonstrates usage patterns
- Real models, real data, real results
- Performance metrics show library capabilities, not app features

---

## 📱 **App Structure**

### **Navigation Pattern**
```
🏠 Main Dashboard
├── 📸 Image Classification (Candle Engine)
├── 💬 Text Sentiment (ONNX Engine) 
├── 🎵 Audio Classification (ONNX Engine)
├── 🧠 On-Device Training (Linfa Engine)
├── ⚡ Performance Comparison
└── ℹ️ About Library
```

### **Screen Flow**
- **Single Activity/ViewController** with bottom navigation
- **Tab-based navigation** for different engine demos
- **Modal sheets** for results and settings
- **Snackbars** for status messages

---

## 🔧 **Technical Requirements**

### **Platform Support**
- **Primary**: iOS 14+ and Android API 21+
- **UI Framework**: Flutter Material Design 3
- **State Management**: Provider (simple, proven)
- **Navigation**: Bottom Navigation Bar

### **Dependencies (Minimal)**
```yaml
dependencies:
  flutter: sdk
  inference: path: ../
  provider: ^6.0.5
  image_picker: ^1.0.4
  file_picker: ^6.1.1
  path_provider: ^2.1.1
```

### **Asset Structure**
```
assets/
├── models/
│   ├── candle/          # PyTorch models (.safetensors, .pt)
│   ├── onnx/            # ONNX models (.onnx)
│   └── linfa/           # Training datasets (.csv, .json)
├── images/
│   └── test_images/     # Sample images for classification
├── audio/               # Sample audio files
└── sample_data/         # Test datasets
```

---

## 🎨 **User Interface Specifications**

### **Design System**
- **Theme**: Material Design 3 with dynamic colors
- **Typography**: Default Material typography scale
- **Colors**: System dynamic colors (iOS) / Material You (Android)
- **Icons**: Material Icons (consistent across platforms)

### **Layout Principles**
- **Responsive**: Single layout that works on phones and tablets
- **Accessible**: Proper semantic labels and contrast ratios
- **Consistent**: Same UI behavior on iOS and Android
- **Loading States**: Clear progress indicators for model operations

---

## 📸 **Feature Specifications**

### **1. Image Classification (Candle Engine)**
**Purpose**: Demonstrate PyTorch model loading and image inference

**Models Required**:
- MobileNet v2 (image classification)
- ResNet-50 (alternative classifier)

**User Flow**:
1. User taps "Image Classification" tab
2. Options: "Take Photo" or "Choose from Gallery"
3. Image displays with loading spinner
4. Results show: Top 5 predictions with confidence scores
5. Processing time displayed

**UI Elements**:
- Image preview (max 300x300)
- Prediction cards with confidence bars
- "Try Another Image" button
- Processing time badge

### **2. Text Sentiment Analysis (ONNX Engine)**
**Purpose**: Demonstrate ONNX model loading and text processing

**Models Required**:
- DistilBERT sentiment classifier
- Simple LSTM sentiment model (backup)

**User Flow**:
1. User taps "Text Sentiment" tab
2. Text input field with sample texts dropdown
3. "Analyze" button triggers inference
4. Results show: Sentiment label + confidence score
5. Processing time displayed

**UI Elements**:
- Multi-line text input
- Sample text chips (tap to fill)
- Sentiment result card (Positive/Negative/Neutral)
- Confidence percentage
- Processing time badge

### **3. Audio Classification (ONNX Engine)**
**Purpose**: Demonstrate audio model loading and audio inference

**Models Required**:
- YAMNet (audio event classification)
- Simple audio classifier (backup)

**User Flow**:
1. User taps "Audio Classification" tab
2. Options: "Record Audio" or "Choose File"
3. Audio waveform visualization (simple)
4. Results show: Top 3 audio event predictions
5. Processing time displayed

**UI Elements**:
- Record button with waveform
- Audio file picker
- Prediction list with confidence
- Processing time badge

### **4. On-Device Training (Linfa Engine)**
**Purpose**: Demonstrate on-device model training capabilities

**Training Algorithm**:
- K-Means clustering (simple, visual)
- Linear regression (backup)

**User Flow**:
1. User taps "Training" tab
2. Pre-loaded dataset visualization (scatter plot)
3. "Train Model" button starts training
4. Real-time training progress
5. Results show: Cluster visualization or regression line
6. Training time displayed

**UI Elements**:
- Dataset scatter plot
- Training progress bar
- Results visualization
- Training metrics display

### **5. Performance Comparison**
**Purpose**: Show library performance across engines

**Metrics Displayed**:
- Model loading time per engine
- Inference time per engine
- Memory usage (basic)
- Engine-specific optimizations enabled

**UI Elements**:
- Comparison table
- Bar charts for timing
- Engine capability badges

### **6. About Library**
**Purpose**: Library information and documentation links

**Content**:
- Inference library version
- Supported engines and versions
- Links to documentation
- GitHub repository link
- License information

---

## 📊 **Performance Requirements**

### **Loading Times**
- App launch: < 3 seconds
- Model loading: < 5 seconds per model
- Image inference: < 2 seconds
- Text inference: < 1 second
- Audio inference: < 3 seconds

### **Memory Usage**
- Base app: < 50MB
- With models loaded: < 200MB
- No memory leaks during model switching

### **Responsiveness**
- UI remains responsive during inference
- Loading indicators for all operations > 500ms
- Graceful error handling with user-friendly messages

---

## 🛡️ **Error Handling**

### **Model Loading Errors**
- File not found: "Model file missing - please check installation"
- Corrupted model: "Invalid model format - please reinstall app"
- Unsupported format: "Model format not supported by selected engine"

### **Inference Errors**
- Invalid input: "Please provide valid input data"
- Processing failure: "Inference failed - please try again"
- Memory issues: "Insufficient memory - please close other apps"

### **Permission Errors**
- Camera: "Camera permission required for photo capture"
- Microphone: "Microphone permission required for audio recording"
- Storage: "Storage permission required for file access"

---

## 🧪 **Testing Strategy**

### **Functional Testing**
- All engine types load successfully
- All inference types produce results
- Error handling works correctly
- UI responds appropriately to all states

### **Performance Testing**
- Memory usage stays within limits
- Inference times meet requirements
- App remains responsive during processing
- No crashes during extended use

### **Platform Testing**
- iOS and Android UI consistency
- Platform-specific permissions work
- File system access works correctly
- Performance parity between platforms

---

## 🚀 **Success Criteria**

### **Primary Goals**
1. ✅ All three engines (Candle, ONNX, Linfa) work with real models
2. ✅ Real inference results (not placeholder data)
3. ✅ Clean, professional UI that works on iOS and Android
4. ✅ Performance metrics prove library capabilities
5. ✅ Zero crashes during normal usage

### **Secondary Goals**
1. ✅ Comprehensive error handling
2. ✅ Helpful loading states and progress indicators
3. ✅ Educational value for developers learning the library
4. ✅ Easy to extend with new models/engines

---

## 📝 **Implementation Notes**

### **Development Approach**
- Start with ONNX engine (most straightforward)
- Add Candle engine (PyTorch models)
- Implement Linfa engine (on-device training)
- Polish UI and add performance metrics
- Comprehensive testing across platforms

### **Asset Preparation**
- Models should be optimized for mobile (< 50MB each)
- Test images should cover various categories
- Audio samples should be short (< 10 seconds)
- Training datasets should be small but meaningful

### **Code Organization**
```
lib/
├── main.dart
├── app.dart
├── screens/
│   ├── dashboard_screen.dart
│   ├── image_classification_screen.dart
│   ├── text_sentiment_screen.dart
│   ├── audio_classification_screen.dart
│   ├── training_screen.dart
│   ├── performance_screen.dart
│   └── about_screen.dart
├── services/
│   ├── inference_service.dart
│   └── asset_service.dart
├── models/
│   ├── prediction_result.dart
│   └── performance_metrics.dart
└── widgets/
    ├── prediction_card.dart
    ├── loading_indicator.dart
    └── performance_chart.dart
```

---

## 🎯 **Final Validation**

The example app is successful when:
1. A developer can download and run it immediately
2. All inference examples work with real models and data
3. The UI clearly demonstrates library capabilities
4. Performance metrics prove the library's efficiency
5. Code serves as a reference for implementing the library

**This BRD focuses on simplicity, functionality, and showcasing the Inference library's true capabilities without unnecessary complexity.** 