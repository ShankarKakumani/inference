import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:inference/inference.dart';
import 'package:inference/src/rust/api/inference.dart' as rust_api;
import '../services/inference_service.dart';
import 'package:http/http.dart' as http;
import 'dart:async';

class HuggingFaceDemoScreen extends StatefulWidget {
  const HuggingFaceDemoScreen({super.key});

  @override
  State<HuggingFaceDemoScreen> createState() => _HuggingFaceDemoScreenState();
}

class _HuggingFaceDemoScreenState extends State<HuggingFaceDemoScreen> {
  final TextEditingController _textController = TextEditingController();
  InferenceSession? _bertModel;
  InferenceSession? _resnetModel;
  InferenceResult? _result;
  double? _processingTime;
  String? _loadError;
  bool _isLoadingBert = false;
  bool _isLoadingResNet = false;
  String _selectedModel = 'bert';
  String? _networkTestResult;

  // Progress tracking
  Timer? _progressTimer;
  rust_api.DownloadProgress? _downloadProgress;
  String? _currentDownloadId;

  // Sample texts for BERT testing
  final List<String> _sampleTexts = [
    "The weather is beautiful today!",
    "I love using machine learning in Flutter.",
    "This HuggingFace integration is amazing.",
    "Natural language processing with Rust is powerful.",
    "Zero-setup ML inference is the future.",
  ];

  @override
  void initState() {
    super.initState();
    _textController.text = 'The weather is beautiful today!';
  }

  @override
  void dispose() {
    _progressTimer?.cancel();
    _textController.dispose();
    super.dispose();
  }

  /// Start progress monitoring for a download
  void _startProgressMonitoring(String downloadId) {
    _currentDownloadId = downloadId;
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(milliseconds: 500), (
      timer,
    ) async {
      try {
        final progress = await rust_api.getDownloadProgress(repo: downloadId);
        if (mounted) {
          setState(() {
            _downloadProgress = progress;
          });

          // Stop monitoring when download is completed or failed
          if (progress != null &&
              (progress.phase == rust_api.DownloadPhase.completed ||
                  progress.phase == rust_api.DownloadPhase.failed)) {
            timer.cancel();
            _progressTimer = null;
          }
        }
      } catch (e) {
        print('Error getting download progress: $e');
      }
    });
  }

  /// Stop progress monitoring
  void _stopProgressMonitoring() {
    _progressTimer?.cancel();
    _progressTimer = null;
    _currentDownloadId = null;
    if (mounted) {
      setState(() {
        _downloadProgress = null;
      });
    }
  }

  Future<void> _testNetworkConnectivity() async {
    setState(() {
      _networkTestResult = null;
    });

    try {
      final response = await http
          .get(
            Uri.parse('https://huggingface.co/google-bert/bert-base-uncased'),
            headers: {'User-Agent': 'Flutter-Inference-Test/1.0'},
          )
          .timeout(const Duration(seconds: 10));

      setState(() {
        _networkTestResult = 'Network connectivity OK (${response.statusCode})';
      });
    } catch (e) {
      setState(() {
        _networkTestResult = 'Network test failed: $e';
      });
    }
  }

  Future<void> _loadBertFromHuggingFace() async {
    setState(() {
      _isLoadingBert = true;
      _loadError = null;
      _downloadProgress = null;
    });

    try {
      print('🤗 Loading BERT model from HuggingFace Hub...');

      final service = Provider.of<InferenceService>(context, listen: false);

      // Start download with progress tracking
      final downloadId = await rust_api.startDownloadWithProgress(
        repo: 'google-bert/bert-base-uncased',
        revision: 'main',
        filename: 'model.safetensors',
      );

      print('📊 Started download with ID: $downloadId');
      _startProgressMonitoring(downloadId);

      // Load the model (this will wait for the download to complete)
      final model = await service.loadFromHuggingFace(
        repo: 'google-bert/bert-base-uncased',
        filename: 'model.safetensors',
        revision: 'main',
      );

      if (model != null) {
        setState(() {
          _bertModel = model;
          _isLoadingBert = false;
        });
        print('✅ Successfully loaded BERT model');
      } else {
        throw Exception('Model loading returned null');
      }
    } catch (e) {
      setState(() {
        _loadError = e.toString();
        _isLoadingBert = false;
      });
      print('❌ Failed to load BERT model: $e');
      print('📍 Stack trace: ${StackTrace.current}');
    } finally {
      _stopProgressMonitoring();
    }
  }

  Future<void> _loadResNetFromHuggingFace() async {
    setState(() {
      _isLoadingResNet = true;
      _loadError = null;
      _downloadProgress = null;
    });

    try {
      print('🤗 Loading ResNet model from HuggingFace Hub...');

      final service = Provider.of<InferenceService>(context, listen: false);

      // Start download with progress tracking
      final downloadId = await rust_api.startDownloadWithProgress(
        repo: 'microsoft/resnet-50',
        revision: 'main',
        filename: 'model.safetensors',
      );

      print('📊 Started download with ID: $downloadId');
      _startProgressMonitoring(downloadId);

      // Load the model (this will wait for the download to complete)
      final model = await service.loadFromHuggingFace(
        repo: 'microsoft/resnet-50',
        filename: 'model.safetensors',
        revision: 'main',
      );

      if (model != null) {
        setState(() {
          _resnetModel = model;
          _isLoadingResNet = false;
        });
        print('✅ Successfully loaded ResNet model');
      } else {
        throw Exception('Model loading returned null');
      }
    } catch (e) {
      setState(() {
        _loadError = e.toString();
        _isLoadingResNet = false;
      });
      print('❌ Failed to load ResNet model: $e');
      print('📍 Stack trace: ${StackTrace.current}');
    } finally {
      _stopProgressMonitoring();
    }
  }

  Future<void> _runInference() async {
    final service = context.read<InferenceService>();
    final text = _textController.text.trim();

    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter some text to analyze'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    InferenceSession? model;
    InferenceInput input;

    if (_selectedModel == 'bert') {
      if (_bertModel == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please load BERT model first'),
            backgroundColor: Colors.orange,
          ),
        );
        return;
      }
      model = _bertModel;
      input = NLPInput(text);
    } else {
      if (_resnetModel == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please load ResNet model first'),
            backgroundColor: Colors.orange,
          ),
        );
        return;
      }
      model = _resnetModel;
      // For ResNet demo, we'll use a dummy tensor input
      input = TensorInput.fromList([1.0, 2.0, 3.0]);
    }

    try {
      print('🚀 HuggingFace Demo: Starting inference...');
      print('📊 Model: $_selectedModel');
      print('📊 Input: $input');

      // Run inference with timing
      final stopwatch = Stopwatch()..start();
      final result = await service.runInference(model!, input, 'candle');
      stopwatch.stop();

      if (result != null && mounted) {
        print('✅ HuggingFace Demo: Inference completed successfully');
        print('📊 Result: $result');
        setState(() {
          _result = result;
          _processingTime = stopwatch.elapsedMilliseconds.toDouble();
        });
      } else {
        print('❌ HuggingFace Demo: Result is null');
      }
    } catch (e, stackTrace) {
      print('❌ HuggingFace Demo: Inference failed: $e');
      print('📍 Stack trace: $stackTrace');

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Inference failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _useSampleText(String text) {
    _textController.text = text;
    setState(() {
      _result = null;
      _processingTime = null;
    });
  }

  /// Build download progress widget
  Widget _buildDownloadProgress() {
    if (_downloadProgress == null) return const SizedBox.shrink();

    final progress = _downloadProgress!;
    final isDownloading = progress.phase == rust_api.DownloadPhase.downloading;

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _getPhaseIcon(progress.phase),
                  color: _getPhaseColor(progress.phase),
                  size: 20,
                ),
                const SizedBox(width: 8),
                Text(
                  'Download Progress',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Progress bar
            if (isDownloading && progress.totalBytes != null)
              Column(
                children: [
                  LinearProgressIndicator(
                    value: progress.percentage / 100.0,
                    backgroundColor: Colors.grey[300],
                    valueColor: AlwaysStoppedAnimation<Color>(
                      _getPhaseColor(progress.phase),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '${progress.percentage.toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      Text(
                        '${_formatBytes(progress.downloadedBytes.toInt())} / ${_formatBytes(progress.totalBytes!.toInt())}',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ],
              )
            else if (isDownloading)
              const LinearProgressIndicator(),

            // Status message
            if (progress.message != null) ...[
              const SizedBox(height: 8),
              Text(
                progress.message!,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],

            // Phase indicator
            const SizedBox(height: 8),
            Chip(
              label: Text(_getPhaseText(progress.phase)),
              backgroundColor: _getPhaseColor(progress.phase).withOpacity(0.1),
              labelStyle: TextStyle(
                color: _getPhaseColor(progress.phase),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Get icon for download phase
  IconData _getPhaseIcon(rust_api.DownloadPhase phase) {
    switch (phase) {
      case rust_api.DownloadPhase.connecting:
        return Icons.wifi_find;
      case rust_api.DownloadPhase.downloading:
        return Icons.download;
      case rust_api.DownloadPhase.processing:
        return Icons.settings;
      case rust_api.DownloadPhase.caching:
        return Icons.save;
      case rust_api.DownloadPhase.completed:
        return Icons.check_circle;
      case rust_api.DownloadPhase.failed:
        return Icons.error;
    }
  }

  /// Get color for download phase
  Color _getPhaseColor(rust_api.DownloadPhase phase) {
    switch (phase) {
      case rust_api.DownloadPhase.connecting:
        return Colors.blue;
      case rust_api.DownloadPhase.downloading:
        return Colors.green;
      case rust_api.DownloadPhase.processing:
        return Colors.orange;
      case rust_api.DownloadPhase.caching:
        return Colors.purple;
      case rust_api.DownloadPhase.completed:
        return Colors.green;
      case rust_api.DownloadPhase.failed:
        return Colors.red;
    }
  }

  /// Get text for download phase
  String _getPhaseText(rust_api.DownloadPhase phase) {
    switch (phase) {
      case rust_api.DownloadPhase.connecting:
        return 'Connecting';
      case rust_api.DownloadPhase.downloading:
        return 'Downloading';
      case rust_api.DownloadPhase.processing:
        return 'Processing';
      case rust_api.DownloadPhase.caching:
        return 'Caching';
      case rust_api.DownloadPhase.completed:
        return 'Completed';
      case rust_api.DownloadPhase.failed:
        return 'Failed';
    }
  }

  /// Format bytes to human readable format
  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    if (bytes < 1024 * 1024 * 1024)
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🤗 HuggingFace Integration Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: _runInference,
            icon: const Icon(Icons.play_arrow),
            tooltip: 'Run Inference',
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
                // HuggingFace Integration Info Card
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(
                              Icons.cloud_download,
                              color: Colors.blue,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'HuggingFace Hub Integration',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'This demo shows real model downloading from HuggingFace Hub using the hf-hub crate. Models are automatically cached after first download.',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        const SizedBox(height: 12),
                        // Network Test Button
                        Row(
                          children: [
                            ElevatedButton.icon(
                              onPressed: _testNetworkConnectivity,
                              icon: const Icon(Icons.wifi, size: 16),
                              label: const Text('Test Network'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blue[50],
                                foregroundColor: Colors.blue[700],
                              ),
                            ),
                            const SizedBox(width: 8),
                            if (_networkTestResult != null)
                              Expanded(
                                child: Text(
                                  _networkTestResult!,
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Model Loading Section
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Load Models from HuggingFace Hub',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 16),

                        // BERT Model Loading
                        Row(
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      Icon(
                                        _bertModel != null
                                            ? Icons.check_circle
                                            : Icons.download,
                                        color:
                                            _bertModel != null
                                                ? Colors.green
                                                : Colors.blue,
                                      ),
                                      const SizedBox(width: 8),
                                      const Text('BERT Base Uncased'),
                                    ],
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'bert-base-uncased',
                                    style: Theme.of(
                                      context,
                                    ).textTheme.bodySmall?.copyWith(
                                      fontFamily: 'monospace',
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            ElevatedButton(
                              onPressed:
                                  _isLoadingBert
                                      ? null
                                      : _loadBertFromHuggingFace,
                              child:
                                  _isLoadingBert
                                      ? const SizedBox(
                                        width: 20,
                                        height: 20,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      )
                                      : Text(
                                        _bertModel != null ? 'Loaded' : 'Load',
                                      ),
                            ),
                          ],
                        ),

                        const SizedBox(height: 16),

                        // ResNet Model Loading
                        Row(
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      Icon(
                                        _resnetModel != null
                                            ? Icons.check_circle
                                            : Icons.download,
                                        color:
                                            _resnetModel != null
                                                ? Colors.green
                                                : Colors.blue,
                                      ),
                                      const SizedBox(width: 8),
                                      const Text('ResNet-50'),
                                    ],
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'microsoft/resnet-50',
                                    style: Theme.of(
                                      context,
                                    ).textTheme.bodySmall?.copyWith(
                                      fontFamily: 'monospace',
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            ElevatedButton(
                              onPressed:
                                  _isLoadingResNet
                                      ? null
                                      : _loadResNetFromHuggingFace,
                              child:
                                  _isLoadingResNet
                                      ? const SizedBox(
                                        width: 20,
                                        height: 20,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      )
                                      : Text(
                                        _resnetModel != null
                                            ? 'Loaded'
                                            : 'Load',
                                      ),
                            ),
                          ],
                        ),

                        if (service.isLoading) ...[
                          const SizedBox(height: 16),
                          const LinearProgressIndicator(),
                          const SizedBox(height: 8),
                          Text(
                            service.loadingMessage,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ],
                    ),
                  ),
                ),

                // Download Progress Section
                _buildDownloadProgress(),

                // Error Display Section
                if (_loadError != null) ...[
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
                                'Loading Error',
                                style: Theme.of(context).textTheme.titleMedium
                                    ?.copyWith(color: Colors.red[700]),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _loadError!,
                            style: Theme.of(context).textTheme.bodySmall
                                ?.copyWith(color: Colors.red[700]),
                          ),
                          const SizedBox(height: 8),
                          ElevatedButton(
                            onPressed: () {
                              setState(() {
                                _loadError = null;
                              });
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.red[100],
                              foregroundColor: Colors.red[700],
                            ),
                            child: const Text('Dismiss'),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],

                const SizedBox(height: 16),

                // Model Selection
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Select Model for Inference',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        SegmentedButton<String>(
                          segments: const [
                            ButtonSegment(
                              value: 'bert',
                              label: Text('BERT'),
                              icon: Icon(Icons.text_fields),
                            ),
                            ButtonSegment(
                              value: 'resnet',
                              label: Text('ResNet'),
                              icon: Icon(Icons.image),
                            ),
                          ],
                          selected: {_selectedModel},
                          onSelectionChanged: (Set<String> newSelection) {
                            setState(() {
                              _selectedModel = newSelection.first;
                              _result = null;
                              _processingTime = null;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Input Section
                if (_selectedModel == 'bert') ...[
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Text Input for BERT',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          TextField(
                            controller: _textController,
                            maxLines: 3,
                            decoration: const InputDecoration(
                              hintText: 'Enter text to analyze with BERT...',
                              border: OutlineInputBorder(),
                            ),
                          ),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            children:
                                _sampleTexts.map((text) {
                                  return ActionChip(
                                    label: Text(
                                      text.length > 30
                                          ? '${text.substring(0, 30)}...'
                                          : text,
                                    ),
                                    onPressed: () => _useSampleText(text),
                                  );
                                }).toList(),
                          ),
                        ],
                      ),
                    ),
                  ),
                ] else ...[
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'ResNet Demo Input',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'ResNet demo uses a dummy tensor input [1.0, 2.0, 3.0] to demonstrate the model loading and inference pipeline.',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],

                const SizedBox(height: 16),

                // Run Inference Button
                ElevatedButton.icon(
                  onPressed: _runInference,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run Inference'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.all(16),
                  ),
                ),

                const SizedBox(height: 16),

                // Results Section
                if (_result != null) ...[
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Icon(Icons.analytics, color: Colors.green),
                              const SizedBox(width: 8),
                              Text(
                                'Inference Results',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),

                          // Processing Time
                          if (_processingTime != null) ...[
                            Row(
                              children: [
                                const Icon(Icons.timer, size: 16),
                                const SizedBox(width: 4),
                                Text(
                                  'Processing Time: ${_processingTime!.toStringAsFixed(2)} ms',
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],

                          // Model Output
                          Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.grey[100],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Model Output:',
                                  style: Theme.of(context).textTheme.bodySmall
                                      ?.copyWith(fontWeight: FontWeight.bold),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _selectedModel == 'bert'
                                      ? 'BERT Embeddings: ${_result!.shape} (768-dimensional embeddings)'
                                      : 'ResNet Logits: ${_result!.shape} (1000-class logits)',
                                  style: Theme.of(context).textTheme.bodySmall
                                      ?.copyWith(fontFamily: 'monospace'),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Data Type: ${_result!.dataType}',
                                  style: Theme.of(context).textTheme.bodySmall
                                      ?.copyWith(fontFamily: 'monospace'),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'First 5 values: ${_result!.data.take(5).toList()}',
                                  style: Theme.of(context).textTheme.bodySmall
                                      ?.copyWith(fontFamily: 'monospace'),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],

                // Error Display
                if (_loadError != null) ...[
                  Card(
                    color: Colors.red[50],
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.error, color: Colors.red),
                              SizedBox(width: 8),
                              Text('Error'),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _loadError!,
                            style: const TextStyle(color: Colors.red),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}
