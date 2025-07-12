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
      print('🚀 InferenceService: Starting $engine inference...');
      print('📊 InferenceService: Input type: ${input.runtimeType}');
      print('📊 InferenceService: Session: $session');

      _setLoading(true, 'Running $engine inference...');

      final stopwatch = Stopwatch()..start();
      print('⚡ InferenceService: Calling session.predict()...');
      final result = await session.predict(input);
      stopwatch.stop();

      final inferenceTime = stopwatch.elapsedMilliseconds.toDouble();
      _recordInferenceTime(engine, inferenceTime);

      print(
        '✅ InferenceService: Inference completed successfully in ${inferenceTime}ms',
      );
      print('📊 InferenceService: Result: $result');

      _setLoading(false);
      return result;
    } catch (e, stackTrace) {
      final errorMsg = 'Inference failed: $e';
      print('❌ InferenceService: $errorMsg');
      print('📍 InferenceService: Full error details:');
      print('   - Error type: ${e.runtimeType}');
      print('   - Error message: $e');
      print('   - Input type: ${input.runtimeType}');
      print('   - Engine: $engine');
      print('   - Session: $session');
      print('📍 InferenceService: Stack trace:');
      print('$stackTrace');

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
      print('🚀 InferenceService: Loading model from HuggingFace Hub...');
      print('📊 InferenceService: Repository: $repo');
      print('📊 InferenceService: Filename: $filename');
      print('📊 InferenceService: Revision: ${revision ?? 'main'}');

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

      print(
        '✅ InferenceService: HuggingFace model loaded successfully in ${loadTime}ms',
      );
      print('📊 InferenceService: Session details: $session');

      _setLoading(false);
      return session;
    } catch (e, stackTrace) {
      final errorMsg = 'Failed to load HuggingFace model: $e';
      print('❌ InferenceService: $errorMsg');
      print('📍 InferenceService: Stack trace: $stackTrace');

      _setError(errorMsg);
      return null;
    }
  }
}
