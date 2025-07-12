import 'package:flutter/foundation.dart';
import 'package:inference/inference.dart';

class InferenceService extends ChangeNotifier {
  // Loading states
  bool _isLoading = false;
  String _loadingMessage = '';
  String? _error;

  // Performance metrics
  final Map<String, double> _modelLoadTimes = {};
  final Map<String, double> _inferenceTimes = {};

  // Getters
  bool get isLoading => _isLoading;
  String get loadingMessage => _loadingMessage;
  String? get error => _error;
  Map<String, double> get modelLoadTimes => Map.unmodifiable(_modelLoadTimes);
  Map<String, double> get inferenceTimes => Map.unmodifiable(_inferenceTimes);

  // Set loading state
  void _setLoading(bool loading, [String message = '']) {
    _isLoading = loading;
    _loadingMessage = message;
    _error = null;
    notifyListeners();
  }

  // Set error state
  void _setError(String error) {
    _isLoading = false;
    _loadingMessage = '';
    _error = error;
    notifyListeners();
  }

  // Clear error
  void clearError() {
    _error = null;
    notifyListeners();
  }

  // Record performance metrics
  void _recordLoadTime(String engine, double timeMs) {
    _modelLoadTimes[engine] = timeMs;
  }

  void _recordInferenceTime(String engine, double timeMs) {
    _inferenceTimes[engine] = timeMs;
  }

  // Run inference with timing
  Future<InferenceResult?> runInference(
    InferenceSession session,
    InferenceInput input,
    String engine,
  ) async {
    try {
      debugPrint('🚀 InferenceService: Starting $engine inference...');
      debugPrint('📊 InferenceService: Input type: ${input.runtimeType}');
      debugPrint('📊 InferenceService: Session: $session');

      _setLoading(true, 'Running $engine inference...');

      final stopwatch = Stopwatch()..start();
      debugPrint('⚡ InferenceService: Calling session.predict()...');
      final result = await session.predict(input);
      stopwatch.stop();

      final inferenceTime = stopwatch.elapsedMilliseconds.toDouble();
      _recordInferenceTime(engine, inferenceTime);

      debugPrint(
        '✅ InferenceService: Inference completed successfully in ${inferenceTime}ms',
      );
      debugPrint('📊 InferenceService: Result: $result');

      _setLoading(false);
      return result;
    } catch (e, stackTrace) {
      final errorMsg = 'Inference failed: $e';
      debugPrint('❌ InferenceService: $errorMsg');
      debugPrint('📍 InferenceService: Full error details:');
      debugPrint('   - Error type: ${e.runtimeType}');
      debugPrint('   - Error message: $e');
      debugPrint('   - Input type: ${input.runtimeType}');
      debugPrint('   - Engine: $engine');
      debugPrint('   - Session: $session');
      debugPrint('📍 InferenceService: Stack trace:');
      debugPrint('$stackTrace');

      _setError(errorMsg);
      return null;
    }
  }

  // Load model from HuggingFace Hub
  Future<InferenceSession?> loadFromHuggingFace({
    required String repo,
    required String filename,
    String? revision,
  }) async {
    try {
      debugPrint('🚀 InferenceService: Loading model from HuggingFace Hub...');
      debugPrint('📊 InferenceService: Repository: $repo');
      debugPrint('📊 InferenceService: Filename: $filename');
      debugPrint('📊 InferenceService: Revision: ${revision ?? 'main'}');

      _setLoading(true, 'Downloading model from HuggingFace Hub...');

      final stopwatch = Stopwatch()..start();

      final session = await InferenceSession.loadFromHuggingFace(
        repo,
        filename: filename,
        revision: revision,
      );

      stopwatch.stop();
      final loadTime = stopwatch.elapsedMilliseconds.toDouble();
      _recordLoadTime('huggingface', loadTime);

      debugPrint(
        '✅ InferenceService: HuggingFace model loaded successfully in ${loadTime}ms',
      );
      debugPrint('📊 InferenceService: Session details: $session');

      _setLoading(false);
      return session;
    } catch (e, stackTrace) {
      final errorMsg = 'Failed to load HuggingFace model: $e';
      debugPrint('❌ InferenceService: $errorMsg');
      debugPrint('📍 InferenceService: Stack trace: $stackTrace');

      _setError(errorMsg);
      return null;
    }
  }
}
