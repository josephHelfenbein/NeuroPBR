// lib/main.dart

import 'package:flutter/material.dart';
import 'screens/carousel_screen.dart';

void main() {
  runApp(const NeuroPBRApp());
}

class NeuroPBRApp extends StatelessWidget {
  const NeuroPBRApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuroPBR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.white,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
        useMaterial3: true,
      ),
      home: const CarouselScreen(),
    );
  }
}
