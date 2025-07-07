import 'dart:math';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:inference/inference.dart';
import '../services/inference_service.dart';

class TrainingScreen extends StatefulWidget {
  const TrainingScreen({super.key});

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends State<TrainingScreen> {
  InferenceSession? _trainedModel;
  List<List<double>>? _dataset;
  List<int>? _clusterAssignments;
  double? _trainingTime;
  bool _isTraining = false;

  // Training parameters
  int _numClusters = 3;
  int _maxIterations = 100;
  final double _tolerance = 1e-4;

  @override
  void initState() {
    super.initState();
    _generateSampleDataset();
  }

  @override
  void dispose() {
    _trainedModel?.dispose();
    super.dispose();
  }

  void _generateSampleDataset() {
    final random = Random(42); // Fixed seed for reproducible results
    final dataset = <List<double>>[];

    // Generate 3 clusters of data points
    final clusterCenters = [
      [2.0, 2.0], // Cluster 1
      [6.0, 6.0], // Cluster 2
      [2.0, 6.0], // Cluster 3
    ];

    for (final center in clusterCenters) {
      // Generate 20 points around each cluster center
      for (int i = 0; i < 20; i++) {
        final x = center[0] + (random.nextDouble() - 0.5) * 2.0;
        final y = center[1] + (random.nextDouble() - 0.5) * 2.0;
        dataset.add([x, y]);
      }
    }

    // Shuffle the dataset
    dataset.shuffle(random);

    setState(() {
      _dataset = dataset;
    });
  }

  Future<void> _trainModel() async {
    if (_dataset == null) return;

    final service = context.read<InferenceService>();

    setState(() {
      _isTraining = true;
    });

    try {
      // Train K-means clustering model
      final stopwatch = Stopwatch()..start();
      final model = await service.trainLinfaModel(
        data: _dataset!,
        algorithm: 'kmeans',
        params: {
          'n_clusters': _numClusters.toString(),
          'max_iterations': _maxIterations.toString(),
          'tolerance': _tolerance.toString(),
        },
      );
      stopwatch.stop();

      if (model != null && mounted) {
        // Predict cluster assignments for visualization
        final assignments = <int>[];
        for (final point in _dataset!) {
          final input = TensorInput(point, [point.length]);
          final result = await model.predict(input);
          assignments.add(result.argmax);
        }

        setState(() {
          _trainedModel = model;
          _clusterAssignments = assignments;
          _trainingTime = stopwatch.elapsedMilliseconds.toDouble();
          _isTraining = false;
        });

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Model trained successfully!'),
              backgroundColor: Colors.green,
            ),
          );
        }
      }
    } catch (e) {
      setState(() {
        _isTraining = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Training failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _predictPoint(double x, double y) async {
    if (_trainedModel == null) return;

    try {
      final input = TensorInput([x, y], [2]);
      final result = await _trainedModel!.predict(input);
      final cluster = result.argmax;

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Point ($x, $y) belongs to cluster $cluster'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Prediction failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Widget _buildDatasetVisualization() {
    if (_dataset == null) return const SizedBox();

    return Container(
      height: 300,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey[300]!),
        borderRadius: BorderRadius.circular(8),
      ),
      child: CustomPaint(
        painter: DatasetPainter(
          dataset: _dataset!,
          clusterAssignments: _clusterAssignments,
          onTap: _trainedModel != null ? _predictPoint : null,
        ),
        child: Container(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('On-Device Training'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: _dataset != null && !_isTraining ? _trainModel : null,
            icon:
                _isTraining
                    ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                    : const Icon(Icons.model_training),
            tooltip: 'Train Model',
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
                // Training Status Card
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              _trainedModel != null
                                  ? Icons.check_circle
                                  : _isTraining
                                  ? Icons.hourglass_empty
                                  : Icons.radio_button_unchecked,
                              color:
                                  _trainedModel != null
                                      ? Colors.green
                                      : _isTraining
                                      ? Colors.orange
                                      : Colors.grey,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Linfa K-Means Clustering',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _trainedModel != null
                              ? 'Model trained successfully on ${_dataset?.length ?? 0} data points'
                              : _isTraining
                              ? 'Training in progress...'
                              : 'Ready to train on ${_dataset?.length ?? 0} data points',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        if (_isTraining || service.isLoading)
                          const LinearProgressIndicator(),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Training Parameters
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Training Parameters',
                          style: Theme.of(context).textTheme.titleSmall,
                        ),
                        const SizedBox(height: 12),

                        // Number of Clusters
                        Row(
                          children: [
                            Expanded(
                              child: Text('Number of Clusters: $_numClusters'),
                            ),
                            SizedBox(
                              width: 150,
                              child: Slider(
                                value: _numClusters.toDouble(),
                                min: 2,
                                max: 5,
                                divisions: 3,
                                label: _numClusters.toString(),
                                onChanged:
                                    _isTraining
                                        ? null
                                        : (value) {
                                          setState(() {
                                            _numClusters = value.round();
                                            _trainedModel?.dispose();
                                            _trainedModel = null;
                                            _clusterAssignments = null;
                                          });
                                        },
                              ),
                            ),
                          ],
                        ),

                        // Max Iterations
                        Row(
                          children: [
                            Expanded(
                              child: Text('Max Iterations: $_maxIterations'),
                            ),
                            SizedBox(
                              width: 150,
                              child: Slider(
                                value: _maxIterations.toDouble(),
                                min: 50,
                                max: 200,
                                divisions: 3,
                                label: _maxIterations.toString(),
                                onChanged:
                                    _isTraining
                                        ? null
                                        : (value) {
                                          setState(() {
                                            _maxIterations = value.round();
                                          });
                                        },
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Dataset Visualization
                Text(
                  'Dataset Visualization',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                _buildDatasetVisualization(),

                const SizedBox(height: 16),

                // Train Button
                FilledButton.icon(
                  onPressed:
                      _dataset != null && !_isTraining && !service.isLoading
                          ? _trainModel
                          : null,
                  icon:
                      _isTraining || service.isLoading
                          ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                          : const Icon(Icons.model_training),
                  label: Text(
                    _isTraining || service.isLoading
                        ? 'Training...'
                        : _trainedModel != null
                        ? 'Retrain Model'
                        : 'Train Model',
                  ),
                ),

                const SizedBox(height: 16),

                // Training Results
                if (_trainedModel != null) ...[
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Training Results',
                            style: Theme.of(context).textTheme.titleSmall,
                          ),
                          const SizedBox(height: 12),

                          Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Clusters Found',
                                      style:
                                          Theme.of(context).textTheme.bodySmall,
                                    ),
                                    Text(
                                      _numClusters.toString(),
                                      style: Theme.of(
                                        context,
                                      ).textTheme.headlineSmall?.copyWith(
                                        color:
                                            Theme.of(
                                              context,
                                            ).colorScheme.primary,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Data Points',
                                      style:
                                          Theme.of(context).textTheme.bodySmall,
                                    ),
                                    Text(
                                      _dataset?.length.toString() ?? '0',
                                      style: Theme.of(
                                        context,
                                      ).textTheme.headlineSmall?.copyWith(
                                        color:
                                            Theme.of(
                                              context,
                                            ).colorScheme.primary,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),

                          if (_trainingTime != null) ...[
                            const SizedBox(height: 12),
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
                                'Training time: ${_trainingTime!.toStringAsFixed(0)}ms',
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
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 8),

                  Text(
                    'Tap on the visualization to predict cluster for new points',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontStyle: FontStyle.italic,
                    ),
                    textAlign: TextAlign.center,
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

class DatasetPainter extends CustomPainter {
  final List<List<double>> dataset;
  final List<int>? clusterAssignments;
  final Function(double, double)? onTap;

  DatasetPainter({required this.dataset, this.clusterAssignments, this.onTap});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;

    // Find data bounds
    double minX = dataset.map((p) => p[0]).reduce(min);
    double maxX = dataset.map((p) => p[0]).reduce(max);
    double minY = dataset.map((p) => p[1]).reduce(min);
    double maxY = dataset.map((p) => p[1]).reduce(max);

    // Add padding
    final padding = 0.5;
    minX -= padding;
    maxX += padding;
    minY -= padding;
    maxY += padding;

    // Draw grid
    final gridPaint =
        Paint()
          ..color = Colors.grey[300]!
          ..strokeWidth = 0.5;

    for (int i = 0; i <= 10; i++) {
      final x = size.width * i / 10;
      final y = size.height * i / 10;

      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Draw data points
    for (int i = 0; i < dataset.length; i++) {
      final point = dataset[i];
      final x = (point[0] - minX) / (maxX - minX) * size.width;
      final y = size.height - (point[1] - minY) / (maxY - minY) * size.height;

      if (clusterAssignments != null) {
        final cluster = clusterAssignments![i];
        final colors = [
          Colors.red,
          Colors.blue,
          Colors.green,
          Colors.orange,
          Colors.purple,
        ];
        paint.color = colors[cluster % colors.length];
      } else {
        paint.color = Colors.grey;
      }

      canvas.drawCircle(Offset(x, y), 4, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;

  @override
  bool hitTest(Offset position) => onTap != null;
}
