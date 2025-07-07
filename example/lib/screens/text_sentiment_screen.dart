import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:inference/inference.dart';
import '../services/inference_service.dart';

class TextSentimentScreen extends StatefulWidget {
  const TextSentimentScreen({super.key});

  @override
  State<TextSentimentScreen> createState() => _TextSentimentScreenState();
}

class _TextSentimentScreenState extends State<TextSentimentScreen> {
  final TextEditingController _textController = TextEditingController();
  InferenceSession? _model;
  InferenceResult? _result;
  double? _processingTime;

  // Sample texts from BRD
  final List<String> _sampleTexts = [
    "I love this product! It works perfectly.",
    "This is terrible. I hate it.",
    "The weather is nice today.",
    "I'm feeling great about this decision.",
    "This movie was absolutely awful.",
    "The service was okay, nothing special.",
  ];

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
    _textController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    final service = context.read<InferenceService>();

    try {
      // Load ONNX sentiment model from asset bundle
      final model = await service.loadModelFromAsset(
        'assets/models/onnx/sentiment_model.onnx',
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

  Future<void> _analyzeSentiment() async {
    if (_model == null || _textController.text.trim().isEmpty) return;

    final service = context.read<InferenceService>();
    final text = _textController.text.trim();

    try {
      // Create NLP input for text
      final input = NLPInput(text);

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
            content: Text('Analysis failed: $e'),
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

  String _getSentimentLabel(double score) {
    if (score > 0.6) return 'Positive';
    if (score < 0.4) return 'Negative';
    return 'Neutral';
  }

  Color _getSentimentColor(double score) {
    if (score > 0.6) return Colors.green;
    if (score < 0.4) return Colors.red;
    return Colors.orange;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Text Sentiment Analysis'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: _model != null ? _analyzeSentiment : null,
            icon: const Icon(Icons.psychology),
            tooltip: 'Analyze Sentiment',
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
                              'ONNX Sentiment Model',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'ONNX Runtime enabled and ready for text sentiment analysis',
                          style: Theme.of(context).textTheme.bodySmall
                              ?.copyWith(color: Colors.green[700]),
                        ),
                        if (service.isLoading) const LinearProgressIndicator(),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Sample Text Chips
                Text(
                  'Sample Texts:',
                  style: Theme.of(context).textTheme.titleSmall,
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
                            style: const TextStyle(fontSize: 12),
                          ),
                          onPressed: () => _useSampleText(text),
                        );
                      }).toList(),
                ),

                const SizedBox(height: 16),

                // Text Input
                TextField(
                  controller: _textController,
                  maxLines: 4,
                  decoration: const InputDecoration(
                    labelText: 'Enter text to analyze',
                    hintText: 'Type or select a sample text above...',
                    border: OutlineInputBorder(),
                  ),
                  onChanged: (text) {
                    setState(() {
                      _result = null;
                      _processingTime = null;
                    });
                  },
                ),

                const SizedBox(height: 16),

                // Analyze Button
                FilledButton.icon(
                  onPressed:
                      _model != null &&
                              _textController.text.trim().isNotEmpty &&
                              !service.isLoading
                          ? _analyzeSentiment
                          : null,
                  icon:
                      service.isLoading
                          ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                          : const Icon(Icons.analytics),
                  label: Text(
                    service.isLoading ? 'Analyzing...' : 'Analyze Sentiment',
                  ),
                ),

                const SizedBox(height: 24),

                // Results
                if (_result != null) ...[
                  Text(
                    'Results:',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Sentiment Label
                          Row(
                            children: [
                              Icon(
                                Icons.sentiment_satisfied,
                                color: _getSentimentColor(_result!.scalar),
                                size: 32,
                              ),
                              const SizedBox(width: 12),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    _getSentimentLabel(_result!.scalar),
                                    style: Theme.of(
                                      context,
                                    ).textTheme.headlineSmall?.copyWith(
                                      color: _getSentimentColor(
                                        _result!.scalar,
                                      ),
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Text(
                                    'Confidence: ${(_result!.scalar * 100).toStringAsFixed(1)}%',
                                    style:
                                        Theme.of(context).textTheme.bodyMedium,
                                  ),
                                ],
                              ),
                            ],
                          ),

                          const SizedBox(height: 16),

                          // Confidence Bar
                          Text(
                            'Confidence Score:',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                          const SizedBox(height: 4),
                          LinearProgressIndicator(
                            value: _result!.scalar,
                            backgroundColor: Colors.grey[300],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              _getSentimentColor(_result!.scalar),
                            ),
                          ),

                          const SizedBox(height: 16),

                          // Processing Time
                          if (_processingTime != null)
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color:
                                    Theme.of(
                                      context,
                                    ).colorScheme.primaryContainer,
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: Text(
                                'Processing time: ${_processingTime!.toStringAsFixed(0)}ms',
                                style: Theme.of(
                                  context,
                                ).textTheme.bodySmall?.copyWith(
                                  color:
                                      Theme.of(
                                        context,
                                      ).colorScheme.onPrimaryContainer,
                                ),
                              ),
                            ),
                        ],
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
