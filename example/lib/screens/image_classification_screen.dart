import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:inference/inference.dart';
import '../services/inference_service.dart';

class ImageClassificationScreen extends StatefulWidget {
  const ImageClassificationScreen({super.key});

  @override
  State<ImageClassificationScreen> createState() =>
      _ImageClassificationScreenState();
}

class _ImageClassificationScreenState extends State<ImageClassificationScreen> {
  final ImagePicker _picker = ImagePicker();
  InferenceSession? _model;
  File? _selectedImage;
  InferenceResult? _result;
  double? _processingTime;
  String? _loadError;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    // Load model after the widget is built to avoid Provider state issues
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadModel();
    });
  }

  @override
  void dispose() {
    _model?.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    final service = context.read<InferenceService>();

    setState(() {
      _isLoading = true;
      _loadError = null;
    });

    try {
      const modelPath = 'assets/models/candle/mobilenet_v2.safetensors';
      print('🔄 Loading Candle model: $modelPath');

      // Check if running in debug mode to provide more info
      if (kDebugMode) {
        print('📱 Debug mode: Checking asset bundle...');
        try {
          final data = await DefaultAssetBundle.of(context).load(modelPath);
          print('📁 Asset found! Size: ${data.lengthInBytes} bytes');
        } catch (assetError) {
          print('❌ Asset not found: $assetError');
          print('💡 Make sure the model file is placed at: example/$modelPath');
        }
      }

      // Load Candle model (MobileNet v2 for image classification)
      final model = await service.loadModelFromAsset(modelPath, 'candle');

      print('✅ Model loaded successfully: $model');

      if (model != null && mounted) {
        setState(() {
          _model = model;
          _isLoading = false;
          _loadError = null;
        });
      } else {
        throw Exception(
          'Model loading returned null - check service error state',
        );
      }
    } catch (e, stackTrace) {
      print('❌ Failed to load Candle model: $e');
      print('📍 Stack trace: $stackTrace');

      if (mounted) {
        setState(() {
          _isLoading = false;
          _loadError = e.toString();
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to load model: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );

      if (image != null && mounted) {
        setState(() {
          _selectedImage = File(image.path);
          _result = null;
          _processingTime = null;
        });

        // Automatically classify when image is selected
        _classifyImage();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick image: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _classifyImage() async {
    if (_model == null || _selectedImage == null) return;

    final service = context.read<InferenceService>();

    try {
      // Create image input
      final input = await ImageInput.fromFile(_selectedImage!);

      // Run inference with timing
      final stopwatch = Stopwatch()..start();
      final result = await service.runInference(_model!, input, 'candle');
      stopwatch.stop();

      if (result != null && mounted) {
        setState(() {
          _result = result;
          _processingTime = stopwatch.elapsedMilliseconds.toDouble();
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Classification failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      builder: (BuildContext context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Take Photo'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Choose from Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  List<ClassificationResult> _getTopPredictions() {
    if (_result == null) return [];

    // Get top 5 predictions using the topK method
    return _result!.topK(5);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Classification'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: _showImageSourceDialog,
            icon: const Icon(Icons.add_a_photo),
            tooltip: 'Add Image',
          ),
        ],
      ),
      body: Consumer<InferenceService>(
        builder: (context, service, child) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Model Status Card
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              _model != null
                                  ? Icons.check_circle
                                  : _loadError != null
                                  ? Icons.error
                                  : Icons.hourglass_empty,
                              color:
                                  _model != null
                                      ? Colors.green
                                      : _loadError != null
                                      ? Colors.red
                                      : Colors.orange,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Candle Image Classification Model',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _model != null
                              ? 'MobileNet v2 loaded successfully'
                              : _loadError != null
                              ? 'Failed to load model'
                              : _isLoading
                              ? 'Loading model...'
                              : 'Model not loaded',
                          style: Theme.of(
                            context,
                          ).textTheme.bodySmall?.copyWith(
                            color: _loadError != null ? Colors.red : null,
                          ),
                        ),
                        if (_loadError != null) ...[
                          const SizedBox(height: 8),
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.red[50],
                              border: Border.all(color: Colors.red[200]!),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Error Details:',
                                  style: Theme.of(
                                    context,
                                  ).textTheme.bodySmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: Colors.red[800],
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _loadError!,
                                  style: Theme.of(
                                    context,
                                  ).textTheme.bodySmall?.copyWith(
                                    color: Colors.red[700],
                                    fontFamily: 'monospace',
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 8),
                          OutlinedButton.icon(
                            onPressed: _loadModel,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Retry Loading'),
                          ),
                        ],
                        if (_isLoading || service.isLoading)
                          const LinearProgressIndicator(),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Image Selection
                if (_selectedImage == null) ...[
                  Container(
                    height: 200,
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: Colors.grey[300]!,
                        style: BorderStyle.solid,
                        width: 2,
                      ),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: InkWell(
                      onTap: _showImageSourceDialog,
                      child: const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.add_a_photo,
                              size: 64,
                              color: Colors.grey,
                            ),
                            SizedBox(height: 16),
                            Text(
                              'Tap to add an image',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.grey,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              'Take photo or choose from gallery',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ] else ...[
                  // Selected Image Display
                  Container(
                    height: 300,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.grey[300]!),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        _selectedImage!,
                        fit: BoxFit.cover,
                        width: double.infinity,
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _showImageSourceDialog,
                          icon: const Icon(Icons.image),
                          label: const Text('Try Another'),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: FilledButton.icon(
                          onPressed:
                              _model != null && !service.isLoading
                                  ? _classifyImage
                                  : null,
                          icon:
                              service.isLoading
                                  ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                  : const Icon(Icons.psychology),
                          label: Text(
                            service.isLoading ? 'Classifying...' : 'Classify',
                          ),
                        ),
                      ),
                    ],
                  ),
                ],

                const SizedBox(height: 24),

                // Results
                if (_result != null) ...[
                  Text(
                    'Top 5 Predictions:',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: ListView.builder(
                      itemCount: _getTopPredictions().length,
                      itemBuilder: (context, index) {
                        final prediction = _getTopPredictions()[index];
                        final isTop = index == 0;

                        return Card(
                          margin: const EdgeInsets.only(bottom: 8),
                          color:
                              isTop
                                  ? Theme.of(
                                    context,
                                  ).colorScheme.primaryContainer
                                  : null,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                        vertical: 4,
                                      ),
                                      decoration: BoxDecoration(
                                        color:
                                            isTop
                                                ? Theme.of(
                                                  context,
                                                ).colorScheme.primary
                                                : Theme.of(
                                                  context,
                                                ).colorScheme.secondary,
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Text(
                                        '#${index + 1}',
                                        style: TextStyle(
                                          color:
                                              isTop
                                                  ? Theme.of(
                                                    context,
                                                  ).colorScheme.onPrimary
                                                  : Theme.of(
                                                    context,
                                                  ).colorScheme.onSecondary,
                                          fontSize: 12,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        prediction.className ??
                                            'Class ${prediction.classIndex}',
                                        style: Theme.of(
                                          context,
                                        ).textTheme.titleSmall?.copyWith(
                                          fontWeight:
                                              isTop
                                                  ? FontWeight.bold
                                                  : FontWeight.normal,
                                        ),
                                      ),
                                    ),
                                    Text(
                                      '${(prediction.confidence * 100).toStringAsFixed(1)}%',
                                      style: Theme.of(
                                        context,
                                      ).textTheme.titleSmall?.copyWith(
                                        fontWeight: FontWeight.bold,
                                        color:
                                            isTop
                                                ? Theme.of(
                                                  context,
                                                ).colorScheme.primary
                                                : null,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                LinearProgressIndicator(
                                  value: prediction.confidence,
                                  backgroundColor: Colors.grey[300],
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    isTop
                                        ? Theme.of(context).colorScheme.primary
                                        : Theme.of(
                                          context,
                                        ).colorScheme.secondary,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),

                  // Processing Time
                  if (_processingTime != null)
                    Container(
                      margin: const EdgeInsets.only(top: 16),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.primaryContainer,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        'Processing time: ${_processingTime!.toStringAsFixed(0)}ms',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color:
                              Theme.of(context).colorScheme.onPrimaryContainer,
                        ),
                      ),
                    ),
                ],

                // Error Display
                if (service.error != null)
                  Card(
                    color: Colors.red[50],
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Icon(Icons.error, color: Colors.red),
                              const SizedBox(width: 8),
                              Text(
                                'Error',
                                style: Theme.of(context).textTheme.titleSmall
                                    ?.copyWith(color: Colors.red),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            service.error!,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                          const SizedBox(height: 8),
                          TextButton(
                            onPressed: service.clearError,
                            child: const Text('Dismiss'),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}
