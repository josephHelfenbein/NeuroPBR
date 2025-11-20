import 'package:flutter/material.dart';
import 'start_screen.dart';
import 'scan_screen_new.dart';

class CarouselScreen extends StatefulWidget {
  const CarouselScreen({super.key});

  @override
  State<CarouselScreen> createState() => _CarouselScreenState();
}

class _CarouselScreenState extends State<CarouselScreen> {
  final PageController _pageController = PageController(initialPage: 1000);

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void navigateToCamera() {
    _pageController.animateToPage(
      _pageController.page!.round() + 1,
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeInOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    return PageView.builder(
      controller: _pageController,
      physics: const BouncingScrollPhysics(),
      itemBuilder: (context, index) {
        final screenIndex = index % 2;

        if (screenIndex == 0) {
          return StartScreen(onScanPressed: navigateToCamera);
        } else {
          return const ScanScreenNew();
        }
      },
    );
  }
}
