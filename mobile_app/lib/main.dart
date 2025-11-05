import 'package:flutter/material.dart';
import 'ui/capture_screen.dart';

void main() {
  runApp(const NeuroPBRApp());
}

class NeuroPBRApp extends StatelessWidget {
  const NeuroPBRApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuroPBR',
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
      ),
      home: const CaptureScreen(),
    );
  }
}
