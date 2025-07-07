import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
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

  // Load model with timing
  Future<InferenceSession?> loadModel(String modelPath, String engine) async {
    try {
      print(
        '🚀 InferenceService: Starting to load $engine model from $modelPath',
      );
      _setLoading(true, 'Loading $engine model...');

      final stopwatch = Stopwatch()..start();
      InferenceSession? session;

      switch (engine.toLowerCase()) {
        case 'candle':
          print('🔥 InferenceService: Loading with Candle engine...');
          session = await InferenceSession.loadWithCandle(modelPath);
          break;
        default:
          print('⚡ InferenceService: Loading with auto-detection...');
          session = await InferenceSession.load(modelPath);
      }

      stopwatch.stop();
      final loadTime = stopwatch.elapsedMilliseconds.toDouble();
      _recordLoadTime(engine, loadTime);

      print('✅ InferenceService: Model loaded successfully in ${loadTime}ms');
      print('📊 InferenceService: Session details: $session');

      _setLoading(false);
      return session;
    } catch (e, stackTrace) {
      final errorMsg = 'Failed to load $engine model: $e';
      print('❌ InferenceService: $errorMsg');
      print('📍 InferenceService: Stack trace: $stackTrace');

      _setError(errorMsg);
      return null;
    }
  }

  // Load model from asset bytes with timing
  Future<InferenceSession?> loadModelFromAsset(
    String assetPath,
    String engine,
  ) async {
    try {
      print(
        '🚀 InferenceService: Starting to load $engine model from asset $assetPath',
      );
      _setLoading(true, 'Loading $engine model from asset...');

      final stopwatch = Stopwatch()..start();

      // Load asset as bytes
      print('📁 InferenceService: Loading asset bytes...');
      final assetData = await rootBundle.load(assetPath);
      final modelBytes = assetData.buffer.asUint8List();
      print('📊 InferenceService: Asset loaded: ${modelBytes.length} bytes');

      InferenceSession? session;

      switch (engine.toLowerCase()) {
        case 'candle':
          print(
            '🔥 InferenceService: Loading with Candle engine from bytes...',
          );
          session = await InferenceSession.loadFromBytesWithCandle(modelBytes);
          break;
        default:
          print(
            '⚡ InferenceService: Loading with auto-detection from bytes...',
          );
          session = await InferenceSession.loadFromBytes(modelBytes);
      }

      stopwatch.stop();
      final loadTime = stopwatch.elapsedMilliseconds.toDouble();
      _recordLoadTime(engine, loadTime);

      print(
        '✅ InferenceService: Model loaded successfully from asset in ${loadTime}ms',
      );
      print('📊 InferenceService: Session details: $session');

      _setLoading(false);
      return session;
    } catch (e, stackTrace) {
      final errorMsg = 'Failed to load $engine model from asset: $e';
      print('❌ InferenceService: $errorMsg');
      print('📍 InferenceService: Stack trace: $stackTrace');

      _setError(errorMsg);
      return null;
    }
  }

  // Run inference with timing
  Future<InferenceResult?> runInference(
    InferenceSession session,
    InferenceInput input,
    String engine,
  ) async {
    try {
      _setLoading(true, 'Running $engine inference...');

      final stopwatch = Stopwatch()..start();
      final result = await session.predict(input);
      stopwatch.stop();

      _recordInferenceTime(engine, stopwatch.elapsedMilliseconds.toDouble());

      _setLoading(false);
      return result;
    } catch (e) {
      _setError('Inference failed: $e');
      return null;
    }
  }

  // Train model with Linfa
  Future<InferenceSession?> trainLinfaModel({
    required List<List<double>> data,
    required String algorithm,
    Map<String, dynamic>? params,
  }) async {
    try {
      _setLoading(true, 'Training $algorithm model...');

      final stopwatch = Stopwatch()..start();
      final session = await InferenceSession.trainLinfa(
        data: data,
        algorithm: algorithm,
        params: params,
      );
      stopwatch.stop();

      _recordLoadTime('linfa', stopwatch.elapsedMilliseconds.toDouble());

      _setLoading(false);
      return session;
    } catch (e) {
      _setError('Training failed: $e');
      return null;
    }
  }
}
