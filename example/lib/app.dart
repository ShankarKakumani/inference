import 'package:flutter/material.dart';
import 'screens/image_classification_screen.dart';
import 'screens/text_sentiment_screen.dart';
import 'screens/audio_classification_screen.dart';
import 'screens/training_screen.dart';
import 'screens/performance_screen.dart';
import 'screens/about_screen.dart';

class MainApp extends StatefulWidget {
  const MainApp({super.key});

  @override
  State<MainApp> createState() => _MainAppState();
}

class _MainAppState extends State<MainApp> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    const ImageClassificationScreen(),
    const TextSentimentScreen(),
    const AudioClassificationScreen(),
    const TrainingScreen(),
    const PerformanceScreen(),
    const AboutScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (int index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(icon: Icon(Icons.image), label: 'Images'),
          NavigationDestination(icon: Icon(Icons.chat_bubble), label: 'Text'),
          NavigationDestination(icon: Icon(Icons.audiotrack), label: 'Audio'),
          NavigationDestination(
            icon: Icon(Icons.psychology),
            label: 'Training',
          ),
          NavigationDestination(icon: Icon(Icons.speed), label: 'Performance'),
          NavigationDestination(icon: Icon(Icons.info), label: 'About'),
        ],
      ),
    );
  }
}
