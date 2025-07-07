import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/inference_service.dart';

class PerformanceScreen extends StatefulWidget {
  const PerformanceScreen({super.key});

  @override
  State<PerformanceScreen> createState() => _PerformanceScreenState();
}

class _PerformanceScreenState extends State<PerformanceScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Performance Comparison'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: () {
              // Refresh performance data
              setState(() {});
            },
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
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
                // Performance Overview Card
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.speed, color: Colors.blue),
                            const SizedBox(width: 8),
                            Text(
                              'Performance Metrics',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Comparison of inference engines across different tasks',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Model Loading Times
                Text(
                  'Model Loading Times',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        _buildPerformanceRow(
                          'Candle Engine',
                          'Image Classification',
                          service.modelLoadTimes['candle'],
                          Colors.red,
                          Icons.image,
                        ),
                        const Divider(),
                        _buildPerformanceRow(
                          'ONNX Runtime',
                          'Text/Audio Processing',
                          service.modelLoadTimes['onnx'],
                          Colors.blue,
                          Icons.chat_bubble,
                        ),
                        const Divider(),
                        _buildPerformanceRow(
                          'Linfa Engine',
                          'On-Device Training',
                          service.modelLoadTimes['linfa'],
                          Colors.green,
                          Icons.psychology,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Inference Times
                Text(
                  'Inference Times',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        _buildPerformanceRow(
                          'Candle Engine',
                          'Image Processing',
                          service.inferenceTimes['candle'],
                          Colors.red,
                          Icons.image,
                        ),
                        const Divider(),
                        _buildPerformanceRow(
                          'ONNX Runtime',
                          'Text/Audio Processing',
                          service.inferenceTimes['onnx'],
                          Colors.blue,
                          Icons.chat_bubble,
                        ),
                        const Divider(),
                        _buildPerformanceRow(
                          'Linfa Engine',
                          'Clustering Prediction',
                          service.inferenceTimes['linfa'],
                          Colors.green,
                          Icons.psychology,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Engine Capabilities
                Text(
                  'Engine Capabilities',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        _buildCapabilityRow(
                          'Candle Engine',
                          'PyTorch models, HuggingFace integration, GPU acceleration',
                          ['SafeTensors', 'PyTorch', 'CUDA', 'MKL'],
                          Colors.red,
                        ),
                        const Divider(),
                                                                          _buildCapabilityRow(
                          'ONNX Runtime',
                          'Cross-platform ONNX models with hardware acceleration',
                          ['ONNX', 'CUDA', 'CoreML', 'TensorRT'],
                          Colors.blue,
                        ),
                        const Divider(),
                        _buildCapabilityRow(
                          'Linfa Engine',
                          'On-device training, classical ML algorithms',
                          [
                            'K-Means',
                            'SVM',
                            'Linear Regression',
                            'Decision Trees',
                          ],
                          Colors.green,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Performance Summary
                Card(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Performance Summary',
                          style: Theme.of(
                            context,
                          ).textTheme.titleSmall?.copyWith(
                            color:
                                Theme.of(
                                  context,
                                ).colorScheme.onPrimaryContainer,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '• All engines demonstrate real ML inference capabilities\n'
                          '• Performance varies based on model complexity and task type\n'
                          '• GPU acceleration available when supported by hardware\n'
                          '• Memory usage optimized through proper resource management',
                          style: Theme.of(
                            context,
                          ).textTheme.bodySmall?.copyWith(
                            color:
                                Theme.of(
                                  context,
                                ).colorScheme.onPrimaryContainer,
                          ),
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

  Widget _buildPerformanceRow(
    String engineName,
    String taskDescription,
    double? timeMs,
    Color color,
    IconData icon,
  ) {
    return Row(
      children: [
        Icon(icon, color: color, size: 24),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                engineName,
                style: Theme.of(
                  context,
                ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
              ),
              Text(
                taskDescription,
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              timeMs != null ? '${timeMs.toStringAsFixed(0)}ms' : 'Not tested',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                color: timeMs != null ? color : Colors.grey,
                fontWeight: FontWeight.bold,
              ),
            ),
            if (timeMs != null)
              Container(
                width: 60,
                height: 4,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
                child: FractionallySizedBox(
                  alignment: Alignment.centerLeft,
                  widthFactor: (timeMs / 3000).clamp(
                    0.0,
                    1.0,
                  ), // Scale to 3 seconds max
                  child: Container(
                    decoration: BoxDecoration(
                      color: color,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ],
    );
  }

  Widget _buildCapabilityRow(
    String engineName,
    String description,
    List<String> features,
    Color color,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            Text(
              engineName,
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(description, style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 8),
        Wrap(
          spacing: 4,
          children:
              features.map((feature) {
                return Chip(
                  label: Text(feature, style: const TextStyle(fontSize: 10)),
                  backgroundColor: color.withOpacity(0.1),
                  side: BorderSide(color: color.withOpacity(0.3)),
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  visualDensity: VisualDensity.compact,
                );
              }).toList(),
        ),
      ],
    );
  }
}
