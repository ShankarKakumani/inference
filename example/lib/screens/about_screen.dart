import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  void _copyToClipboard(BuildContext context, String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Copied to clipboard'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('About Library'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Library Header
            Card(
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.primary,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Icon(
                        Icons.psychology,
                        size: 48,
                        color: Theme.of(context).colorScheme.onPrimary,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Inference Flutter Library',
                      style: Theme.of(context).textTheme.headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Zero-setup ML inference for Flutter applications',
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.primaryContainer,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        'Version 1.0.0',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color:
                              Theme.of(context).colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Supported Engines
            Text(
              'Supported Engines',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    _buildEngineRow(
                      context,
                      'Candle',
                      'PyTorch models with Rust performance',
                      'v0.8.0',
                      Colors.red,
                      Icons.whatshot,
                    ),
                    const Divider(),
                    _buildEngineRow(
                      context,
                      'Linfa',
                      'Classical ML algorithms in Rust',
                      'v0.7.0',
                      Colors.green,
                      Icons.science,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Features
            Text(
              'Key Features',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildFeatureItem(
                      context,
                      Icons.flash_on,
                      'Zero Configuration',
                      'Install and start using ML models immediately',
                    ),
                    _buildFeatureItem(
                      context,
                      Icons.public,
                      'Universal Loading',
                      'Load from assets, URLs, files, or Hugging Face Hub',
                    ),
                    _buildFeatureItem(
                      context,
                      Icons.auto_awesome,
                      'Auto-Detection',
                      'Intelligent engine selection based on model format',
                    ),
                    _buildFeatureItem(
                      context,
                      Icons.speed,
                      'Hardware Acceleration',
                      'Automatic GPU/NPU detection and optimization',
                    ),
                    _buildFeatureItem(
                      context,
                      Icons.school,
                      'On-Device Training',
                      'Train classical ML models directly on device',
                    ),
                    _buildFeatureItem(
                      context,
                      Icons.security,
                      'Type-Safe',
                      'Full Dart type safety with comprehensive error handling',
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Links and Actions
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Resources',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    _buildLinkRow(
                      context,
                      Icons.book,
                      'Documentation',
                      'pub.dev/packages/inference',
                      () => _copyToClipboard(
                        context,
                        'https://pub.dev/packages/inference',
                      ),
                    ),
                    _buildLinkRow(
                      context,
                      Icons.code,
                      'GitHub Repository',
                      'github.com/inference-flutter/inference',
                      () => _copyToClipboard(
                        context,
                        'https://github.com/inference-flutter/inference',
                      ),
                    ),
                    _buildLinkRow(
                      context,
                      Icons.bug_report,
                      'Report Issues',
                      'github.com/inference-flutter/inference/issues',
                      () => _copyToClipboard(
                        context,
                        'https://github.com/inference-flutter/inference/issues',
                      ),
                    ),
                    _buildLinkRow(
                      context,
                      Icons.gavel,
                      'License',
                      'MIT License',
                      () => _copyToClipboard(context, 'MIT License'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Attribution
            Text(
              'Made with ❤️ by the Flutter community',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(fontStyle: FontStyle.italic),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEngineRow(
    BuildContext context,
    String name,
    String description,
    String version,
    Color color,
    IconData icon,
  ) {
    return Row(
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 24),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                name,
                style: Theme.of(
                  context,
                ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
              ),
              Text(
                description,
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withOpacity(0.3)),
          ),
          child: Text(
            version,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildFeatureItem(
    BuildContext context,
    IconData icon,
    String title,
    String description,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(
                    context,
                  ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold),
                ),
                Text(
                  description,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLinkRow(
    BuildContext context,
    IconData icon,
    String title,
    String url,
    VoidCallback onTap,
  ) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8.0),
        child: Row(
          children: [
            Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    url,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.primary,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.copy, size: 16, color: Colors.grey[600]),
          ],
        ),
      ),
    );
  }
}
