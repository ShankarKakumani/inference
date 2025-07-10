import 'package:inference/inference.dart';

/// Simple example demonstrating the Inference package
///
/// This example shows how to:
/// 1. Load a machine learning model
/// 2. Create input data
/// 3. Run inference
/// 4. Process results
Future<void> main() async {
  print('🚀 Inference Package Example');
  print('============================');

  try {
    // Initialize the Rust library
    await RustLib.init();
    print('✅ Inference library initialized');

    // Example 1: Load a model (auto-detects engine based on file extension)
    print('\n📦 Loading ML model...');
    final session = await InferenceSession.load(
      'assets/models/candle/mobilenet_v2.safetensors',
    );
    print('✅ Model loaded successfully using ${session.engine} engine');

    // Example 2: Create input data for image classification
    print('\n🖼️  Preparing image input...');
    final imageInput = await ImageInput.fromAsset(
      'assets/images/test_images/cat.jpg',
    );
    print(
      '✅ Image input created: ${imageInput.width}x${imageInput.height}x${imageInput.channels}',
    );

    // Example 3: Run inference
    print('\n🧠 Running inference...');
    final result = await session.predict(imageInput);
    print('✅ Inference completed');

    // Example 4: Process results
    print('\n📊 Processing results...');
    final topPredictions = result.topK(3);

    print('Top 3 predictions:');
    for (int i = 0; i < topPredictions.length; i++) {
      final pred = topPredictions[i];
      print(
        '  ${i + 1}. Class ${pred.classIndex}: ${(pred.confidence * 100).toStringAsFixed(2)}%',
      );
    }

    // Example 5: Clean up resources
    session.dispose();
    print('\n✅ Resources cleaned up');

    print('\n🎉 Example completed successfully!');
  } catch (e) {
    print('❌ Error: $e');
    print('\nNote: This example requires model files in assets/models/');
    print('In a real app, you would handle errors appropriately.');
  }
}

/// Alternative example for on-device training with Linfa
Future<void> trainingExample() async {
  print('\n🎓 On-Device Training Example');
  print('==============================');

  try {
    // Load training data
    final session = await LinfaSession.trainKMeans(
      data: [
        [1.0, 2.0],
        [2.0, 1.0],
        [8.0, 9.0],
        [9.0, 8.0],
      ],
      numClusters: 2,
    );

    print('✅ K-means model trained on-device');

    // Make predictions
    final testInput = TensorInput.fromList([5.0, 6.0]);
    final result = await session.predict(testInput);

    print('📊 Prediction for [5.0, 6.0]: Cluster ${result.argmax}');

    session.dispose();
    print('✅ Training example completed');
  } catch (e) {
    print('❌ Training error: $e');
  }
}

/// Quick start example - minimal code for getting started
Future<void> quickStartExample() async {
  print('\n⚡ Quick Start (3 lines of code)');
  print('==================================');

  // Just 3 lines to get ML inference working!
  final model = await InferenceSession.load('path/to/model.safetensors');
  final input = await ImageInput.fromAsset('path/to/image.jpg');
  final result = await model.predict(input);

  print('🎯 Top prediction: Class ${result.argmax}');
  model.dispose();
}
