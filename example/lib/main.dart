import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:inference/inference.dart';
import 'app.dart';
import 'services/inference_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const InferenceExampleApp());
}

class InferenceExampleApp extends StatelessWidget {
  const InferenceExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [ChangeNotifierProvider(create: (_) => InferenceService())],
      child: MaterialApp(
        title: 'Inference HuggingFace Demo',
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        ),
        home: const MainApp(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
