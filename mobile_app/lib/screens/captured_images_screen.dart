import 'package:flutter/material.dart';
import 'dart:io';

class CapturedImagesScreen extends StatelessWidget {
  final List<String> imagePaths;

  const CapturedImagesScreen({super.key, required this.imagePaths});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text('Captured Images'),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Center(
        child: ListView.builder(
          itemCount: imagePaths.length,
          itemBuilder: (context, index) {
            return Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Text(
                    'Image ${index + 1}',
                    style: const TextStyle(color: Colors.white, fontSize: 18),
                  ),
                  const SizedBox(height: 8),
                  Image.file(
                    File(imagePaths[index]),
                    fit: BoxFit.contain,
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
