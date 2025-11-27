import 'package:flutter/material.dart';
import 'start_screen.dart';
import 'scan_screen.dart';
import 'tags_screen.dart';
import 'stats_screen.dart';
import 'settings_screen.dart';

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
        final screenIndex = index % 5;

        switch (screenIndex) {
          case 0:
            return StartScreen(onScanPressed: navigateToCamera);
          case 1:
            return const ScanScreenNew();
          case 2:
            return const TagsScreen();
          case 3:
            return const StatsScreen();
          case 4:
            return const SettingsScreen();
          default:
            return StartScreen(onScanPressed: navigateToCamera);
        }
      },
    );
  }
}
