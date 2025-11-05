import 'package:flutter/material.dart';
import 'scan_screen.dart';

class StartScreen extends StatefulWidget {
  const StartScreen({super.key});

  @override
  State<StartScreen> createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    // Pulse controller for breathing logo glow
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
      lowerBound: 0.0,
      upperBound: 1.0,
    )..repeat(reverse: true); // loops forever

    _pulseAnimation =
        CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white, // full white background
      body: SafeArea(
        child: Container(
          width: double.infinity,
          height: double.infinity,
          color: Colors.white,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // --- Logo with breathing blue glow ---
              AnimatedBuilder(
                animation: _pulseAnimation,
                builder: (context, child) {
                  final glowStrength = 0.3 + (_pulseAnimation.value * 0.4);

                  return Stack(
                    alignment: Alignment.center,
                    children: [
                      // Outer glow
                      Container(
                        width: 220,
                        height: 220,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.blueAccent.withOpacity(glowStrength),
                              blurRadius: 40 * glowStrength,
                              spreadRadius: 8 * glowStrength,
                            ),
                          ],
                        ),
                      ),
                      // Actual logo
                      Image.asset(
                        'assets/logo.png',
                        width: 180,
                        fit: BoxFit.contain,
                      ),
                    ],
                  );
                },
              ),

              const SizedBox(height: 70),

              // --- Modern Start Scan button ---
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const ScanScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.black,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                    horizontal: 60,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                  elevation: 3,
                ),
                child: const Text(
                  'Start Scan',
                  style: TextStyle(
                    fontSize: 18,
                    letterSpacing: 1.1,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),

              const SizedBox(height: 40),

              // --- App tagline ---
              const Text(
                'NeuroPBR',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 30,
                  letterSpacing: 0.6,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
