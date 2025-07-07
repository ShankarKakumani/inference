import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:inference/inference.dart';
import '../services/inference_service.dart';

class AudioClassificationScreen extends StatefulWidget {
  const AudioClassificationScreen({super.key});

  @override
  State<AudioClassificationScreen> createState() =>
      _AudioClassificationScreenState();
}

class _AudioClassificationScreenState extends State<AudioClassificationScreen> {
  InferenceSession? _model;
  File? _selectedAudio;
  InferenceResult? _result;
  double? _processingTime;

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

    try {
      // Load ONNX audio classification model (YAMNet) from asset bundle
      final model = await service.loadModelFromAsset(
        'assets/models/onnx/yamnet_audio.onnx',
        'onnx',
      );

      if (model != null && mounted) {
        setState(() {
          _model = model;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to load model: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _pickAudioFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.audio,
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        setState(() {
          _selectedAudio = File(result.files.single.path!);
          _result = null;
          _processingTime = null;
        });

        // Automatically classify when audio is selected
        _classifyAudio();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick audio file: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _recordAudio() async {
    // For now, show a placeholder message about recording
    // In a real implementation, this would use a recording plugin
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(
          'Audio recording not implemented in this demo. Please select an audio file instead.',
        ),
        duration: Duration(seconds: 3),
      ),
    );
  }

  Future<void> _classifyAudio() async {
    if (_model == null || _selectedAudio == null) return;

    final service = context.read<InferenceService>();

    try {
      // Create audio input
      final input = await AudioInput.fromFile(_selectedAudio!);

      // Run inference with timing
      final stopwatch = Stopwatch()..start();
      final result = await service.runInference(_model!, input, 'onnx');
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
            content: Text('Audio classification failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _showAudioSourceDialog() {
    showModalBottomSheet(
      context: context,
      builder: (BuildContext context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.mic),
                title: const Text('Record Audio'),
                subtitle: const Text('Record a new audio clip'),
                onTap: () {
                  Navigator.pop(context);
                  _recordAudio();
                },
              ),
              ListTile(
                leading: const Icon(Icons.audio_file),
                title: const Text('Choose Audio File'),
                subtitle: const Text('Select from device storage'),
                onTap: () {
                  Navigator.pop(context);
                  _pickAudioFile();
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

    // Get top 3 predictions for audio events
    return _result!.topK(3);
  }

  String _getAudioFileName() {
    if (_selectedAudio == null) return '';
    return _selectedAudio!.path.split('/').last;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Classification'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: _showAudioSourceDialog,
            icon: const Icon(Icons.audiotrack),
            tooltip: 'Add Audio',
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
                                  : Icons.hourglass_empty,
                              color:
                                  _model != null ? Colors.green : Colors.orange,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'ONNX Audio Classification Model',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'ONNX Runtime enabled and ready for audio classification',
                          style: Theme.of(context).textTheme.bodySmall
                              ?.copyWith(color: Colors.green[700]),
                        ),
                        if (service.isLoading) const LinearProgressIndicator(),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Audio Selection
                if (_selectedAudio == null) ...[
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
                      onTap: _showAudioSourceDialog,
                      child: const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.audiotrack,
                              size: 64,
                              color: Colors.grey,
                            ),
                            SizedBox(height: 16),
                            Text(
                              'Tap to add audio',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.grey,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              'Record audio or choose from files',
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
                  // Selected Audio Display
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Icon(
                                Icons.audio_file,
                                size: 48,
                                color: Colors.blue,
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Selected Audio',
                                      style:
                                          Theme.of(
                                            context,
                                          ).textTheme.titleSmall,
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      _getAudioFileName(),
                                      style:
                                          Theme.of(context).textTheme.bodySmall,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),

                          // Simple waveform visualization placeholder
                          Container(
                            height: 60,
                            decoration: BoxDecoration(
                              color: Colors.grey[100],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Center(
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceEvenly,
                                children: List.generate(20, (index) {
                                  final height = (index % 4 + 1) * 10.0;
                                  return Container(
                                    width: 3,
                                    height: height,
                                    decoration: BoxDecoration(
                                      color: Colors.blue[300],
                                      borderRadius: BorderRadius.circular(2),
                                    ),
                                  );
                                }),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _showAudioSourceDialog,
                          icon: const Icon(Icons.audiotrack),
                          label: const Text('Try Another'),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: FilledButton.icon(
                          onPressed:
                              _model != null && !service.isLoading
                                  ? _classifyAudio
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
                                  : const Icon(Icons.hearing),
                          label: Text(
                            service.isLoading
                                ? 'Classifying...'
                                : 'Classify Audio',
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
                    'Top 3 Audio Events:',
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
                                            'Audio Event ${prediction.classIndex}',
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
