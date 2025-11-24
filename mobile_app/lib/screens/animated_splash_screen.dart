import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class AnimatedSplashScreen extends StatefulWidget {
  final Widget nextScreen;

  const AnimatedSplashScreen({super.key, required this.nextScreen});

  @override
  State<AnimatedSplashScreen> createState() => _AnimatedSplashScreenState();
}

class _AnimatedSplashScreenState extends State<AnimatedSplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _logoSlideAnimation;
  late Animation<Offset> _textSlideAnimation;

  @override
  void initState() {
    super.initState();

    // Set status bar to light mode for white background
    SystemChrome.setSystemUIOverlayStyle(
      const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
        statusBarBrightness: Brightness.light,
      ),
    );

    // Initialize animation controller
    _controller = AnimationController(
      duration: const Duration(milliseconds: 500), // Animation: 500ms
      vsync: this,
    );

    // initial.png (logo) slides LEFT from center
    _logoSlideAnimation = Tween<Offset>(
      begin: Offset.zero,
      end: const Offset(-0.3, 0.0), // Logo moves left (smaller gap)
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));

    // text.png slides RIGHT from center
    _textSlideAnimation = Tween<Offset>(
      begin: Offset.zero,
      end: const Offset(0.3, 0.0), // Text moves right (smaller gap)
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));

    // Start animation after a brief delay
    Future.delayed(const Duration(milliseconds: 200), () { // Initial delay: 200ms
      _controller.forward();
    });

    // Navigate to next screen after animation completes
    _controller.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        Future.delayed(const Duration(milliseconds: 100), () { // Exit delay: 100ms
          if (mounted) {
            Navigator.of(context).pushReplacement(
              PageRouteBuilder(
                pageBuilder: (context, animation, secondaryAnimation) =>
                    widget.nextScreen,
                transitionsBuilder:
                    (context, animation, secondaryAnimation, child) {
                  return FadeTransition(opacity: animation, child: child);
                },
                transitionDuration: const Duration(milliseconds: 300),
              ),
            );
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Get screen width to calculate proper logo size
    final screenWidth = MediaQuery.of(context).size.width;
    final imageSize = screenWidth * 0.4; // Both images take 40% of screen width

    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Stack(
          alignment: Alignment.center,
          clipBehavior: Clip.none,
          children: [
            // Text behind the logo - slides left (appears from behind)
            SlideTransition(
              position: _textSlideAnimation,
              child: Image.asset(
                'assets/text.png',
                width: imageSize,
                height: imageSize,
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  // Fallback if text.png doesn't exist
                  return Text(
                    'NeuroPBR',
                    style: TextStyle(
                      fontSize: imageSize * 0.2,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                    ),
                  );
                },
              ),
            ),
            // Logo on top - slides left (reveals text behind)
            SlideTransition(
              position: _logoSlideAnimation,
              child: Image.asset(
                'assets/initial.png',
                width: imageSize,
                height: imageSize,
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  // Fallback if initial.png doesn't exist, use logo.png
                  return Image.asset(
                    'assets/logo.png',
                    width: imageSize,
                    height: imageSize,
                    fit: BoxFit.contain,
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
