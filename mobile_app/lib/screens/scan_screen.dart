import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with WidgetsBindingObserver, TickerProviderStateMixin {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _flashOn = false;
  int _currentCameraIndex = 0;
  String _selectedMaterial = 'MISC';
  bool _isCapturing = false;
  bool _materialSelectorExpanded = false;
  double _exposureValue = 0.0;
  int _isoValue = 400;
  String _resolutionInfo = 'Loading...';
  String _fpsInfo = '...';
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  List<double> _availableZoomLevels = [1.0];
  String _aspectRatio = '1:1'; // '1:1', '9:16', 'Full'

  late AnimationController _focusController;
  late AnimationController _captureController;
  late AnimationController _selectorController;
  late AnimationController _scanAnimationController;
  late Animation<double> _focusAnimation;
  late Animation<double> _scanLineAnimation;

  final List<String> _materialTypes = [
    'MISC',
    'WOOD',
    'METAL',
    'FABRIC',
    'STONE',
    'PLASTIC',
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _focusController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _focusAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _focusController, curve: Curves.easeInOut),
    );

    _captureController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );

    _selectorController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 200),
    );

    _scanAnimationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );

    _scanLineAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _scanAnimationController, curve: Curves.easeInOut),
    );

    _initializeCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _focusController.dispose();
    _captureController.dispose();
    _selectorController.dispose();
    _scanAnimationController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras!.isEmpty) return;

      _controller = CameraController(
        _cameras![_currentCameraIndex],
        ResolutionPreset.max, // Highest quality available
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();

      if (mounted) {
        // Get actual camera resolution
        final size = _controller!.value.previewSize;
        if (size != null) {
          final width = size.height.toInt(); // swapped because of orientation
          final height = size.width.toInt();

          // Determine resolution label
          if (width >= 3840 || height >= 2160) {
            _resolutionInfo = '4K';
          } else if (width >= 1920 || height >= 1080) {
            _resolutionInfo = '1080p';
          } else if (width >= 1280 || height >= 720) {
            _resolutionInfo = '720p';
          } else {
            _resolutionInfo = '${width}x$height';
          }

          _fpsInfo = '30fps'; // Most mobile cameras default to 30fps
        }

        // Get zoom capabilities
        _minZoom = await _controller!.getMinZoomLevel();
        _maxZoom = await _controller!.getMaxZoomLevel();

        // Generate available zoom levels (1x, 2x, 3x) within device limits
        _availableZoomLevels = [];
        for (double zoom = 1.0; zoom <= 3.0; zoom += 1.0) {
          if (zoom <= _maxZoom) {
            _availableZoomLevels.add(zoom);
          }
        }
        // If device supports more, cap at 3x for UI simplicity
        if (_availableZoomLevels.isEmpty) {
          _availableZoomLevels = [1.0];
        }
        _currentZoom = 1.0;

        setState(() => _isInitialized = true);
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      // Set quirky fallback text if camera fails
      setState(() {
        _resolutionInfo = 'Neural Vision™';
        _fpsInfo = 'Engaged';
      });
    }
  }

  void _toggleMaterialSelector() {
    HapticFeedback.lightImpact();
    setState(() => _materialSelectorExpanded = !_materialSelectorExpanded);
  }

  Future<void> _toggleFlash() async {
    if (_controller == null) return;

    HapticFeedback.lightImpact();
    final newFlashMode = _flashOn ? FlashMode.off : FlashMode.torch;
    await _controller!.setFlashMode(newFlashMode);
    setState(() => _flashOn = !_flashOn);
  }

  Future<void> _applyExposure() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      // Set exposure offset (range is typically -4.0 to 4.0, we use -3.0 to 3.0)
      await _controller!.setExposureOffset(_exposureValue);
    } catch (e) {
      debugPrint('Error setting exposure: $e');
    }
  }

  Future<void> _setZoom(double zoom) async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      HapticFeedback.lightImpact();
      await _controller!.setZoomLevel(zoom);
      setState(() => _currentZoom = zoom);
    } catch (e) {
      debugPrint('Error setting zoom: $e');
    }
  }

  Future<void> _pickFromGallery() async {
    HapticFeedback.lightImpact();
    final ImagePicker picker = ImagePicker();
    try {
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);
      if (image != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Selected: ${image.name}',
              style: GoogleFonts.robotoMono(
                fontWeight: FontWeight.w500,
              ),
            ),
            duration: const Duration(seconds: 2),
            backgroundColor: const Color(0xFF1a1a1a),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } catch (e) {
      debugPrint('Error picking image: $e');
    }
  }

  Future<void> _capturePhoto() async {
    if (_controller == null || !_controller!.value.isInitialized || _isCapturing) {
      return;
    }

    try {
      setState(() => _isCapturing = true);
      HapticFeedback.heavyImpact();

      // Start scan animation
      _scanAnimationController.forward(from: 0.0);

      await _captureController.forward();
      await _controller!.takePicture();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Material scanned: $_selectedMaterial',
              style: GoogleFonts.robotoMono(
                fontWeight: FontWeight.w500,
              ),
            ),
            duration: const Duration(seconds: 2),
            backgroundColor: const Color(0xFF1a1a1a),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }

      await Future.delayed(const Duration(milliseconds: 400));
      await _captureController.reverse();
      _scanAnimationController.reset();
      setState(() => _isCapturing = false);
    } catch (e) {
      debugPrint('Error capturing photo: $e');
      setState(() => _isCapturing = false);
      _captureController.reset();
      _scanAnimationController.reset();
    }
  }

  @override
  Widget build(BuildContext context) {
    // Make system bars appropriate for dark theme
    SystemChrome.setSystemUIOverlayStyle(
      const SystemUiOverlayStyle(
        systemNavigationBarColor: Color(0xFF000000), // Solid black for Android nav bar
        systemNavigationBarIconBrightness: Brightness.light,
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
      ),
    );

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A), // Dark background like Not Boring
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF2C2C2C), // Frame color
              borderRadius: BorderRadius.circular(40),
              border: Border.all(
                color: const Color(0xFF444444),
                width: 1,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.4),
                  blurRadius: 12,
                  offset: const Offset(0, 6),
                ),
                BoxShadow(
                  color: Colors.black.withOpacity(0.9),
                  blurRadius: 8,
                  offset: const Offset(0, 4),
                  spreadRadius: -2,
                  blurStyle: BlurStyle.inner,
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Stack(
                children: [
                  // Main content
                  Column(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Top HUD - Technical Info
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Left: Material type
                      GestureDetector(
                        onTap: _toggleMaterialSelector,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0xFFea580c),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: const Color(0xFFc2410c),
                              width: 1,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.3),
                                blurRadius: 4,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.label_outline,
                                color: Colors.white,
                                size: 14,
                              ),
                              const SizedBox(width: 6),
                              Text(
                                _selectedMaterial,
                                style: GoogleFonts.robotoMono(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 1.2,
                                ),
                              ),
                              const SizedBox(width: 4),
                              Icon(
                                _materialSelectorExpanded
                                    ? Icons.keyboard_arrow_up
                                    : Icons.keyboard_arrow_down,
                                color: Colors.white,
                                size: 16,
                              ),
                            ],
                          ),
                        ),
                      ),

                      // Right: Format & Settings
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.6),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                color: Colors.white.withOpacity(0.1),
                                width: 1,
                              ),
                            ),
                            child: Text(
                              '$_resolutionInfo · $_fpsInfo',
                              style: GoogleFonts.robotoMono(
                                color: Colors.white.withOpacity(0.9),
                                fontSize: 10,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '⚡ Neural Capture',
                            style: GoogleFonts.robotoMono(
                              color: const Color(0xFFfbbf24).withOpacity(0.8),
                              fontSize: 9,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                // Material selector dropdown
                if (_materialSelectorExpanded)
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    margin: const EdgeInsets.symmetric(horizontal: 16),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1a1a1a).withOpacity(0.95),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.1),
                        width: 1,
                      ),
                    ),
                    child: Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _materialTypes.map((type) {
                        final isSelected = _selectedMaterial == type;
                        return GestureDetector(
                          onTap: () {
                            HapticFeedback.lightImpact();
                            setState(() => _selectedMaterial = type);
                            _toggleMaterialSelector();
                          },
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 8,
                            ),
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? const Color(0xFFea580c)
                                  : const Color(0xFF282828),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: isSelected
                                    ? const Color(0xFFc2410c)
                                    : const Color(0xFF444444),
                                width: 1,
                              ),
                            ),
                            child: Text(
                              type,
                              style: GoogleFonts.robotoMono(
                                color: isSelected
                                    ? Colors.white
                                    : Colors.white.withOpacity(0.7),
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 1.2,
                              ),
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                  ),

                const Spacer(),

                // Center: Focus Box
                Center(
                  child: AnimatedBuilder(
                    animation: _focusAnimation,
                    builder: (context, child) {
                      return Opacity(
                        opacity: _focusAnimation.value,
                        child: Container(
                          width: 100,
                          height: 100,
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: const Color(0xFFfbbf24),
                              width: 2,
                            ),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Stack(
                            children: [
                              // Corner brackets
                              Positioned(
                                top: -1,
                                left: -1,
                                child: Container(
                                  width: 16,
                                  height: 16,
                                  decoration: const BoxDecoration(
                                    border: Border(
                                      top: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                      left: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              Positioned(
                                top: -1,
                                right: -1,
                                child: Container(
                                  width: 16,
                                  height: 16,
                                  decoration: const BoxDecoration(
                                    border: Border(
                                      top: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                      right: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              Positioned(
                                bottom: -1,
                                left: -1,
                                child: Container(
                                  width: 16,
                                  height: 16,
                                  decoration: const BoxDecoration(
                                    border: Border(
                                      bottom: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                      left: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              Positioned(
                                bottom: -1,
                                right: -1,
                                child: Container(
                                  width: 16,
                                  height: 16,
                                  decoration: const BoxDecoration(
                                    border: Border(
                                      bottom: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                      right: BorderSide(
                                        color: Color(0xFFfbbf24),
                                        width: 2,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),

                const Spacer(),

                // Scan animation - strong camera flash/flicker
                if (_isCapturing)
                  Positioned.fill(
                    child: AnimatedBuilder(
                      animation: _scanLineAnimation,
                      builder: (context, child) {
                        // Strong flash at the start, fades out quickly
                        double opacity = 0.0;
                        if (_scanLineAnimation.value < 0.1) {
                          // Flash up very quickly to full white
                          opacity = _scanLineAnimation.value / 0.1;
                        } else if (_scanLineAnimation.value < 0.3) {
                          // Fade out
                          opacity = (0.3 - _scanLineAnimation.value) / 0.2;
                        }

                        return IgnorePointer(
                          child: Container(
                            color: Colors.white.withOpacity(opacity),
                          ),
                        );
                      },
                    ),
                  ),

                // Bottom: Controls
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.black.withOpacity(0),
                        Colors.black.withOpacity(0.8),
                      ],
                    ),
                  ),
                  child: Column(
                    children: [
                      // Exposure controls
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _TactileButton(
                            icon: Icons.remove,
                            onTap: () async {
                              setState(() {
                                _exposureValue =
                                    (_exposureValue - 0.3).clamp(-3.0, 3.0);
                              });
                              HapticFeedback.lightImpact();
                              await _applyExposure();
                            },
                          ),
                          const SizedBox(width: 12),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 8,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.6),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: Colors.white.withOpacity(0.1),
                                width: 1,
                              ),
                            ),
                            child: Text(
                              'EV ${_exposureValue > 0 ? '+' : ''}${_exposureValue == 0 ? '0.0' : _exposureValue.toFixed(1)}',
                              style: GoogleFonts.robotoMono(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          _TactileButton(
                            icon: Icons.add,
                            onTap: () async {
                              setState(() {
                                _exposureValue =
                                    (_exposureValue + 0.3).clamp(-3.0, 3.0);
                              });
                              HapticFeedback.lightImpact();
                              await _applyExposure();
                            },
                          ),
                        ],
                      ),

                      const SizedBox(height: 16),

                      // Zoom controls
                      if (_availableZoomLevels.length > 1)
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: _availableZoomLevels.map((zoom) {
                            final isSelected = _currentZoom == zoom;
                            return Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 4),
                              child: GestureDetector(
                                onTap: () => _setZoom(zoom),
                                child: Container(
                                  width: 48,
                                  height: 48,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: isSelected
                                        ? const Color(0xFFea580c)
                                        : Colors.black.withOpacity(0.6),
                                    border: Border.all(
                                      color: isSelected
                                          ? const Color(0xFFc2410c)
                                          : Colors.white.withOpacity(0.2),
                                      width: isSelected ? 2 : 1,
                                    ),
                                    boxShadow: isSelected
                                        ? [
                                            BoxShadow(
                                              color: const Color(0xFFea580c)
                                                  .withOpacity(0.3),
                                              blurRadius: 8,
                                              spreadRadius: 1,
                                            ),
                                          ]
                                        : null,
                                  ),
                                  child: Center(
                                    child: Text(
                                      '${zoom.toInt()}×',
                                      style: GoogleFonts.robotoMono(
                                        color: Colors.white,
                                        fontSize: 14,
                                        fontWeight: isSelected
                                            ? FontWeight.w700
                                            : FontWeight.w500,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            );
                          }).toList(),
                        ),

                      if (_availableZoomLevels.length > 1)
                        const SizedBox(height: 16),

                      // Main control row
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Gallery
                          _ControlButton(
                            icon: Icons.photo_library,
                            onTap: _pickFromGallery,
                          ),

                          // Shutter button
                          GestureDetector(
                            onTap: _capturePhoto,
                            child: Container(
                              width: 80,
                              height: 80,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: const Color(0xFFea580c),
                                border: Border.all(
                                  color: const Color(0xFF7c2d12),
                                  width: 4,
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: const Color(0xFFea580c)
                                        .withOpacity(0.4),
                                    blurRadius: 20,
                                    spreadRadius: 2,
                                  ),
                                ],
                              ),
                              child: Container(
                                margin: const EdgeInsets.all(6),
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.black.withOpacity(0.1),
                                ),
                              ),
                            ),
                          ),

                          // Flash toggle
                          _ControlButton(
                            icon:
                                _flashOn ? Icons.flash_on : Icons.flash_off,
                            onTap: _toggleFlash,
                            isActive: _flashOn,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Helper extension for double formatting
extension DoubleFormat on double {
  String toFixed(int decimals) {
    return toStringAsFixed(decimals);
  }
}

// Tactile button widget (small adjustment controls)
class _TactileButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const _TactileButton({
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.black.withOpacity(0.6),
          border: Border.all(
            color: Colors.white.withOpacity(0.1),
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Icon(
          icon,
          color: Colors.white,
          size: 18,
        ),
      ),
    );
  }
}

// Control button widget (main controls)
class _ControlButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final bool isActive;

  const _ControlButton({
    required this.icon,
    required this.onTap,
    this.isActive = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          color: isActive
              ? const Color(0xFFea580c)
              : const Color(0xFF282828),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isActive
                ? const Color(0xFFc2410c)
                : const Color(0xFF444444),
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.4),
              blurRadius: 6,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Icon(
          icon,
          color: isActive ? Colors.white : Colors.white.withOpacity(0.8),
          size: 24,
        ),
      ),
    );
  }
}

