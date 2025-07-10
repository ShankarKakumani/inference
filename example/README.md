# Inference Package Examples

This directory contains comprehensive examples demonstrating how to use the **Inference** package for machine learning in Flutter applications.

## 🚀 Quick Start

The simplest way to get started with the Inference package:

```dart
import 'package:inference/inference.dart';

// Initialize the library
await RustLib.init();

// Load any ML model with automatic engine detection
final model = await InferenceSession.load('path/to/model.safetensors');

// Create input and run inference
final input = await ImageInput.fromAsset('path/to/image.jpg');
final result = await model.predict(input);

// Get top prediction
final topPrediction = result.topK(1).first;
print('Prediction: Class ${topPrediction.classIndex} (${topPrediction.confidence})');

// Clean up
model.dispose();
```

## 📁 Example Structure

### 🎯 **Standalone Examples**
- **`example.dart`** - Simple standalone example showing core functionality
- **`main.dart`** - Full Flutter app with comprehensive UI

### 🖼️ **Image Classification**
```dart
// Load MobileNet model
final session = await InferenceSession.load('assets/models/mobilenet_v2.safetensors');

// Process image
final input = await ImageInput.fromAsset('assets/images/cat.jpg');
final result = await session.predict(input);

// Get top 5 predictions
final predictions = result.topK(5);
for (final pred in predictions) {
  print('${pred.classIndex}: ${(pred.confidence * 100).toStringAsFixed(2)}%');
}
```

### 🧠 **On-Device Training**
```dart
// Train K-means clustering on device
final session = await LinfaSession.trainKMeans(
  data: [[1.0, 2.0], [8.0, 9.0]], // Your training data
  numClusters: 2,
);

// Make predictions
final input = TensorInput.fromList([5.0, 6.0]);
final result = await session.predict(input);
print('Cluster: ${result.argmax}');
```

### 📝 **Text Sentiment Analysis**
```dart
// Load safetensors sentiment model
final session = await InferenceSession.load('assets/models/sentiment_model.safetensors');

// Analyze text
final input = NLPInput('This movie is amazing!');
final result = await session.predict(input);

// Get sentiment (0=negative, 1=positive)
final sentiment = result.argmax == 1 ? 'Positive' : 'Negative';
print('Sentiment: $sentiment');
```

## 🔧 **Supported Engines**

The package automatically detects and uses the appropriate engine:

| Engine | Models | Use Cases |
|--------|--------|-----------|
| **Candle** | `.safetensors`, `.pt`, `.pth` | PyTorch models, Neural networks |
| **Linfa** | On-device training | Classical ML, K-means, SVM, Decision trees |

## 🏃‍♂️ **Running the Examples**

### Prerequisites
1. Add the inference package to your `pubspec.yaml`:
   ```yaml
   dependencies:
     inference: ^0.1.0-beta.4
   ```

2. Place model files in your `assets/` directory and declare them in `pubspec.yaml`:
   ```yaml
   flutter:
     assets:
       - assets/models/
       - assets/images/
   ```

### Run the Full Example App
```bash
cd example
flutter run
```

### Run Standalone Examples
```bash
dart run example/example.dart
```

## 📱 **Platform Support**

✅ **Android** - Full support with GPU acceleration  
✅ **iOS** - Full support with Metal acceleration  
✅ **Windows** - Full support  
✅ **macOS** - Full support  
✅ **Linux** - Full support  

## 🎯 **Key Features Demonstrated**

- **Zero Setup**: No complex configuration required
- **Auto-Detection**: Automatically selects the right engine for your model
- **Type Safety**: Full Dart type safety with comprehensive error handling
- **Performance**: Hardware acceleration when available
- **Flexibility**: Support for images, text, tensors, and audio inputs
- **Memory Management**: Proper resource cleanup and disposal

## 🔗 **Learn More**

- 📖 [Package Documentation](https://pub.dev/packages/inference)
- 🐙 [GitHub Repository](https://github.com/ShankarKakumani/inference)
- 🎯 [API Reference](https://pub.dev/documentation/inference/latest/)

## 💡 **Need Help?**

- Check out the [comprehensive Flutter app](lib/main.dart) for advanced usage
- Browse the [screen examples](lib/screens/) for specific use cases
- Review the [service layer](lib/services/) for architecture patterns

---

**Happy inferencing! 🚀**
