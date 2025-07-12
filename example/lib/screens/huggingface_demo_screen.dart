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
  final TextEditingController _customModelController = TextEditingController();
  InferenceSession? _bertModel;
  InferenceSession? _resnetModel;
  InferenceResult? _result;
  double? _processingTime;
  String? _loadError;
  bool _isLoadingBert = false;
  bool _isLoadingResNet = false;
  bool _isLoadingCustom = false;
  String _selectedModel = 'bert';
  String? _networkTestResult;

  // Individual progress tracking for each model
  Timer? _bertProgressTimer;
  Timer? _resnetProgressTimer;
  Timer? _customProgressTimer;
  rust_api.DownloadProgress? _bertDownloadProgress;
  rust_api.DownloadProgress? _resnetDownloadProgress;
  rust_api.DownloadProgress? _customDownloadProgress;
  String? _bertDownloadId;
  String? _resnetDownloadId;
  String? _customDownloadId;

  // Cache management
  int? _cacheSize;
  bool _isClearingCache = false;
  bool _isLoadingCacheSize = false;

  // Downloaded models tracking
  List<String> _downloadedModels = [];
  bool _isLoadingDownloadedModels = false;

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
    _loadCacheSize();
    _loadDownloadedModels();
  }

  @override
  void dispose() {
    _bertProgressTimer?.cancel();
    _resnetProgressTimer?.cancel();
    _customProgressTimer?.cancel();
    _textController.dispose();
    _customModelController.dispose();
    super.dispose();
  }

  /// Load cache size information
  Future<void> _loadCacheSize() async {
    if (_isLoadingCacheSize) return;

    setState(() {
      _isLoadingCacheSize = true;
    });

    try {
      final size = await InferenceSession.getCacheSize();
      if (mounted) {
        setState(() {
          _cacheSize = size;
        });
      }
    } catch (e) {
      print('Error loading cache size: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingCacheSize = false;
        });
      }
    }
  }

  /// Load downloaded models list (mock implementation - you can enhance this)
  Future<void> _loadDownloadedModels() async {
    setState(() {
      _isLoadingDownloadedModels = true;
    });

    try {
      // This is a mock implementation. In a real app, you'd query the cache
      // or maintain a list of downloaded models
      await Future.delayed(const Duration(milliseconds: 500));

      List<String> models = [];
      if (_bertModel != null) models.add('google-bert/bert-base-uncased');
      if (_resnetModel != null) models.add('microsoft/resnet-50');
      // Keep existing custom models that aren't already in the list
      for (String model in _downloadedModels) {
        if (!models.contains(model)) {
          models.add(model);
        }
      }

      if (mounted) {
        setState(() {
          _downloadedModels = models;
        });
      }
    } catch (e) {
      print('Error loading downloaded models: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingDownloadedModels = false;
        });
      }
    }
  }

  /// Clear model cache
  Future<void> _clearCache() async {
    if (_isClearingCache) return;

    setState(() {
      _isClearingCache = true;
    });

    try {
      await InferenceSession.clearCache();

      // Reset model states since cache is cleared
      setState(() {
        _bertModel = null;
        _resnetModel = null;
        _result = null;
        _processingTime = null;
        _downloadedModels.clear();
      });

      await _loadCacheSize(); // Refresh cache size
      await _loadDownloadedModels(); // Refresh downloaded models

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Cache cleared successfully! Models will need to be reloaded.',
            ),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      print('Error clearing cache: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to clear cache: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isClearingCache = false;
        });
      }
    }
  }

  /// Start progress monitoring for BERT download
  void _startBertProgressMonitoring(String downloadId) {
    _bertDownloadId = downloadId;
    _bertProgressTimer?.cancel();
    _bertProgressTimer = Timer.periodic(const Duration(milliseconds: 500), (
      timer,
    ) async {
      try {
        final progress = await rust_api.getDownloadProgress(repo: downloadId);
        if (mounted) {
          setState(() {
            _bertDownloadProgress = progress;
          });

          // Stop monitoring when download is completed or failed
          if (progress != null &&
              (progress.phase == rust_api.DownloadPhase.completed ||
                  progress.phase == rust_api.DownloadPhase.failed)) {
            timer.cancel();
            _bertProgressTimer = null;
            // Refresh cache size and downloaded models after download completes
            Future.delayed(const Duration(milliseconds: 500), () {
              _loadCacheSize();
              _loadDownloadedModels();
            });
          }
        }
      } catch (e) {
        print('Error getting BERT download progress: $e');
      }
    });
  }

  /// Start progress monitoring for ResNet download
  void _startResnetProgressMonitoring(String downloadId) {
    _resnetDownloadId = downloadId;
    _resnetProgressTimer?.cancel();
    _resnetProgressTimer = Timer.periodic(const Duration(milliseconds: 500), (
      timer,
    ) async {
      try {
        final progress = await rust_api.getDownloadProgress(repo: downloadId);
        if (mounted) {
          setState(() {
            _resnetDownloadProgress = progress;
          });

          // Stop monitoring when download is completed or failed
          if (progress != null &&
              (progress.phase == rust_api.DownloadPhase.completed ||
                  progress.phase == rust_api.DownloadPhase.failed)) {
            timer.cancel();
            _resnetProgressTimer = null;
            // Refresh cache size and downloaded models after download completes
            Future.delayed(const Duration(milliseconds: 500), () {
              _loadCacheSize();
              _loadDownloadedModels();
            });
          }
        }
      } catch (e) {
        print('Error getting ResNet download progress: $e');
      }
    });
  }

  /// Start progress monitoring for custom model download
  void _startCustomProgressMonitoring(String downloadId) {
    _customDownloadId = downloadId;
    _customProgressTimer?.cancel();
    _customProgressTimer = Timer.periodic(const Duration(milliseconds: 500), (
      timer,
    ) async {
      try {
        final progress = await rust_api.getDownloadProgress(repo: downloadId);
        if (mounted) {
          setState(() {
            _customDownloadProgress = progress;
          });

          // Stop monitoring when download is completed or failed
          if (progress != null &&
              (progress.phase == rust_api.DownloadPhase.completed ||
                  progress.phase == rust_api.DownloadPhase.failed)) {
            timer.cancel();
            _customProgressTimer = null;
            // Refresh cache size and downloaded models after download completes
            Future.delayed(const Duration(milliseconds: 500), () {
              _loadCacheSize();
              _loadDownloadedModels();
            });
          }
        }
      } catch (e) {
        print('Error getting custom model download progress: $e');
      }
    });
  }

  /// Stop BERT progress monitoring
  void _stopBertProgressMonitoring() {
    _bertProgressTimer?.cancel();
    _bertProgressTimer = null;
    _bertDownloadId = null;
    if (mounted) {
      setState(() {
        _bertDownloadProgress = null;
      });
    }
  }

  /// Stop ResNet progress monitoring
  void _stopResnetProgressMonitoring() {
    _resnetProgressTimer?.cancel();
    _resnetProgressTimer = null;
    _resnetDownloadId = null;
    if (mounted) {
      setState(() {
        _resnetDownloadProgress = null;
      });
    }
  }

  /// Stop custom model progress monitoring
  void _stopCustomProgressMonitoring() {
    _customProgressTimer?.cancel();
    _customProgressTimer = null;
    _customDownloadId = null;
    if (mounted) {
      setState(() {
        _customDownloadProgress = null;
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
      _bertDownloadProgress = null;
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

      print('📊 Started BERT download with ID: $downloadId');
      _startBertProgressMonitoring(downloadId);

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
        // Refresh downloaded models list after successful load
        _loadDownloadedModels();
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
      _stopBertProgressMonitoring();
    }
  }

  Future<void> _loadResNetFromHuggingFace() async {
    setState(() {
      _isLoadingResNet = true;
      _loadError = null;
      _resnetDownloadProgress = null;
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

      print('📊 Started ResNet download with ID: $downloadId');
      _startResnetProgressMonitoring(downloadId);

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
        // Refresh downloaded models list after successful load
        _loadDownloadedModels();
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
      _stopResnetProgressMonitoring();
    }
  }

  Future<void> _loadCustomModel() async {
    final repo = _customModelController.text.trim();
    if (repo.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter a model repository'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() {
      _isLoadingCustom = true;
      _loadError = null;
      _customDownloadProgress = null;
    });

    try {
      print('🤗 Loading custom model from HuggingFace Hub: $repo');

      final service = Provider.of<InferenceService>(context, listen: false);

      // Option 1: Simple load (no progress) - RECOMMENDED
      // This handles download + caching + loading internally
      final model = await service.loadFromHuggingFace(
        repo: repo,
        filename: 'model.safetensors',
        revision: 'main',
      );

      if (model != null) {
        setState(() {
          _isLoadingCustom = false;
        });
        print('✅ Successfully loaded custom model: $repo');
        // Add custom model directly to downloaded models list
        if (!_downloadedModels.contains(repo)) {
          _downloadedModels.add(repo);
        }
        // Refresh downloaded models list after successful load
        _loadDownloadedModels();

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Successfully loaded model: $repo'),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        throw Exception('Model loading returned null');
      }
    } catch (e) {
      setState(() {
        _loadError = e.toString();
        _isLoadingCustom = false;
      });
      print('❌ Failed to load custom model: $e');
      print('📍 Stack trace: ${StackTrace.current}');
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

  /// Build a clickable model chip for popular models
  Widget _buildModelChip(String repo, String displayName) {
    return GestureDetector(
      onTap: () {
        _customModelController.text = repo;
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.blue[50],
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue[200]!),
        ),
        child: Text(
          displayName,
          style: TextStyle(
            color: Colors.blue[700],
            fontSize: 11,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }

  /// Build enhanced progress widget for downloads
  Widget _buildCompactProgress(rust_api.DownloadProgress? progress) {
    if (progress == null) return const SizedBox.shrink();

    final isDownloading = progress.phase == rust_api.DownloadPhase.downloading;
    final isCompleted = progress.phase == rust_api.DownloadPhase.completed;
    final isFailed = progress.phase == rust_api.DownloadPhase.failed;

    return Container(
      margin: const EdgeInsets.only(top: 6),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: _getPhaseColor(progress.phase).withOpacity(0.08),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: _getPhaseColor(progress.phase).withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                _getPhaseIcon(progress.phase),
                color: _getPhaseColor(progress.phase),
                size: 14,
              ),
              const SizedBox(width: 6),
              if (isDownloading && progress.totalBytes != null) ...[
                Text(
                  '${progress.percentage.toStringAsFixed(0)}%',
                  style: TextStyle(
                    color: _getPhaseColor(progress.phase),
                    fontWeight: FontWeight.w600,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  '${_formatBytes(progress.downloadedBytes)} / ${_formatBytes(progress.totalBytes!)}',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 10,
                    fontFamily: 'monospace',
                  ),
                ),
              ] else ...[
                Text(
                  _getPhaseText(progress.phase),
                  style: TextStyle(
                    color: _getPhaseColor(progress.phase),
                    fontWeight: FontWeight.w600,
                    fontSize: 11,
                  ),
                ),
              ],
            ],
          ),
          if (isDownloading && progress.totalBytes != null) ...[
            const SizedBox(height: 4),
            SizedBox(
              width: 120,
              height: 3,
              child: LinearProgressIndicator(
                value: progress.percentage / 100.0,
                backgroundColor: Colors.grey[300],
                valueColor: AlwaysStoppedAnimation<Color>(
                  _getPhaseColor(progress.phase),
                ),
                borderRadius: BorderRadius.circular(1.5),
              ),
            ),
          ],
        ],
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
  String _formatBytes(BigInt bytes) {
    final bytesAsDouble = bytes.toDouble();
    if (bytesAsDouble < 1024) return '$bytes B';
    if (bytesAsDouble < 1024 * 1024)
      return '${(bytesAsDouble / 1024).toStringAsFixed(1)} KB';
    if (bytesAsDouble < 1024 * 1024 * 1024)
      return '${(bytesAsDouble / (1024 * 1024)).toStringAsFixed(1)} MB';
    return '${(bytesAsDouble / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
  }

  /// Format bytes to human readable format (overload for int)
  String _formatBytesInt(int bytes) {
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

                        const SizedBox(height: 12),

                        // Cache Management Section
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.grey[50],
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.grey[300]!),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  const Icon(
                                    Icons.storage,
                                    size: 16,
                                    color: Colors.grey,
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    'Cache Management',
                                    style:
                                        Theme.of(context).textTheme.titleSmall,
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'Total Cache Size: ${_isLoadingCacheSize
                                              ? 'Loading...'
                                              : _cacheSize != null
                                              ? _formatBytesInt(_cacheSize!)
                                              : 'Unknown'}',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.bodySmall?.copyWith(
                                            fontWeight: FontWeight.w500,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          'Includes HuggingFace Hub cache and inference cache.',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.bodySmall?.copyWith(
                                            color: Colors.grey[600],
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          '⚠️ This will clear all cached models and require re-download.',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.bodySmall?.copyWith(
                                            color: Colors.orange[700],
                                            fontSize: 11,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  ElevatedButton.icon(
                                    onPressed:
                                        _isClearingCache ? null : _clearCache,
                                    icon:
                                        _isClearingCache
                                            ? const SizedBox(
                                              width: 16,
                                              height: 16,
                                              child: CircularProgressIndicator(
                                                strokeWidth: 2,
                                              ),
                                            )
                                            : const Icon(Icons.clear, size: 16),
                                    label: Text(
                                      _isClearingCache
                                          ? 'Clearing...'
                                          : 'Clear Cache',
                                    ),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: Colors.orange[50],
                                      foregroundColor: Colors.orange[700],
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Downloaded Models Section
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(
                              Icons.download_done,
                              color: Colors.green,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Downloaded Models',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        if (_isLoadingDownloadedModels)
                          const Center(child: CircularProgressIndicator())
                        else if (_downloadedModels.isEmpty)
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.grey[100],
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: Colors.grey[300]!),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.info_outline,
                                  color: Colors.grey[600],
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  'No models downloaded yet',
                                  style: Theme.of(context).textTheme.bodyMedium
                                      ?.copyWith(color: Colors.grey[600]),
                                ),
                              ],
                            ),
                          )
                        else
                          Column(
                            children:
                                _downloadedModels.map((model) {
                                  return Container(
                                    margin: const EdgeInsets.only(bottom: 8),
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: Colors.green[50],
                                      borderRadius: BorderRadius.circular(8),
                                      border: Border.all(
                                        color: Colors.green[200]!,
                                      ),
                                    ),
                                    child: Row(
                                      children: [
                                        const Icon(
                                          Icons.check_circle,
                                          color: Colors.green,
                                          size: 20,
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                model,
                                                style: Theme.of(context)
                                                    .textTheme
                                                    .bodyMedium
                                                    ?.copyWith(
                                                      fontWeight:
                                                          FontWeight.w500,
                                                    ),
                                              ),
                                              Text(
                                                'Ready for inference',
                                                style: Theme.of(
                                                  context,
                                                ).textTheme.bodySmall?.copyWith(
                                                  color: Colors.green[700],
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ],
                                    ),
                                  );
                                }).toList(),
                          ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Custom Model Download Section
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
                              'Download Any HuggingFace Model',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Enter any HuggingFace model repository to download and use with the inference engine.',
                          style: Theme.of(context).textTheme.bodySmall
                              ?.copyWith(color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _customModelController,
                          decoration: const InputDecoration(
                            hintText:
                                'e.g., google-bert/bert-base-uncased, microsoft/resnet-50',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.link),
                            helperText: 'Format: owner/model-name',
                          ),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            ElevatedButton.icon(
                              onPressed:
                                  _isLoadingCustom ? null : _loadCustomModel,
                              icon:
                                  _isLoadingCustom
                                      ? const SizedBox(
                                        width: 16,
                                        height: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      )
                                      : const Icon(Icons.download),
                              label: Text(
                                _isLoadingCustom
                                    ? 'Downloading...'
                                    : 'Download Model',
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blue[50],
                                foregroundColor: Colors.blue[700],
                              ),
                            ),
                            const SizedBox(width: 12),
                            if (_customDownloadProgress != null)
                              _buildCompactProgress(_customDownloadProgress),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.blue[25],
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.blue[100]!),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                    Icons.lightbulb_outline,
                                    color: Colors.blue[700],
                                    size: 16,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    'Popular Models:',
                                    style: TextStyle(
                                      fontWeight: FontWeight.w600,
                                      color: Colors.blue[700],
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 6),
                              Wrap(
                                spacing: 6,
                                runSpacing: 4,
                                children: [
                                  _buildModelChip(
                                    'google-bert/bert-base-uncased',
                                    'BERT',
                                  ),
                                  _buildModelChip(
                                    'microsoft/resnet-50',
                                    'ResNet',
                                  ),
                                  _buildModelChip(
                                    'distilbert-base-uncased',
                                    'DistilBERT',
                                  ),
                                  _buildModelChip(
                                    'google/mobilebert-uncased',
                                    'MobileBERT',
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

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
              ],
            ),
          );
        },
      ),
    );
  }
}
