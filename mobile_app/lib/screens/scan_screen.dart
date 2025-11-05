import 'package:flutter/material.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    // Controller for grid fade-in
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );

    _fadeAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    );

    // Start animation when screen loads
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Dark gradient background
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF000000), Color(0xFF0A0A1F)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Stack(
          children: [
            // Animated grid overlay
            FadeTransition(
              opacity: _fadeAnimation,
              child: CustomPaint(
                size: MediaQuery.of(context).size,
                painter: AnimatedGridPainter(_fadeAnimation.value),
              ),
            ),

            // Top bar + content
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Back button
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_ios_new),
                      color: Colors.white.withOpacity(0.9),
                    ),

                    const Spacer(),

                    // Placeholder box (replaces logo)
                    Center(
                      child: Container(
                        width: 150,
                        height: 150,
                        decoration: BoxDecoration(
                          color: Colors.transparent,
                          border: Border.all(
                            color: Colors.white.withOpacity(0.5),
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Center(
                          child: Text(
                            'Scan Area',
                            style: TextStyle(
                              color: Colors.white54,
                              fontSize: 14,
                              fontWeight: FontWeight.w400,
                            ),
                          ),
                        ),
                      ),
                    ),

                    const Spacer(),

                    // Instruction text
                    FadeTransition(
                      opacity: _fadeAnimation,
                      child: const Center(
                        child: Padding(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Text(
                            "Align the surface within the grid",
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 16,
                              fontWeight: FontWeight.w400,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Animated grid painter
class AnimatedGridPainter extends CustomPainter {
  final double opacity;
  AnimatedGridPainter(this.opacity);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.15 * opacity)
      ..strokeWidth = 1;

    const columns = 3;
    const rows = 3;

    // Vertical lines
    for (int i = 1; i < columns; i++) {
      final dx = size.width * i / columns;
      canvas.drawLine(Offset(dx, 0), Offset(dx, size.height), paint);
    }

    // Horizontal lines
    for (int i = 1; i < rows; i++) {
      final dy = size.height * i / rows;
      canvas.drawLine(Offset(0, dy), Offset(size.width, dy), paint);
    }
  }

  @override
  bool shouldRepaint(covariant AnimatedGridPainter oldDelegate) {
    return oldDelegate.opacity != opacity;
  }
}
