import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'captured_images_screen.dart';

class ScanScreenNew extends StatefulWidget {
  const ScanScreenNew({super.key});

  @override
  State<ScanScreenNew> createState() => _ScanScreenNewState();
}

class _ScanScreenNewState extends State<ScanScreenNew>
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
  double _minExposure = 0.0;
  double _maxExposure = 0.0;
  int _isoValue = 400;
  String _resolutionInfo = 'Loading...';
  String _fpsInfo = '...';
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  List<double> _availableZoomLevels = [1.0];
  String _aspectRatio = 'Full'; // '3:4', '9:16', 'Full'
  List<String> _capturedImages = []; // Store paths of captured images (max 3)

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
        final size = _controller!.value.previewSize;
        if (size != null) {
          final width = size.height.toInt();
          final height = size.width.toInt();

          if (width >= 3840 || height >= 2160) {
            _resolutionInfo = '4K';
          } else if (width >= 1920 || height >= 1080) {
            _resolutionInfo = '1080p';
          } else if (width >= 1280 || height >= 720) {
            _resolutionInfo = '720p';
          } else {
            _resolutionInfo = '${width}x$height';
          }

          _fpsInfo = '30fps';
        }

        _minZoom = await _controller!.getMinZoomLevel();
        _maxZoom = await _controller!.getMaxZoomLevel();
        _minExposure = await _controller!.getMinExposureOffset();
        _maxExposure = await _controller!.getMaxExposureOffset();

        _availableZoomLevels = [];
        for (double zoom = 1.0; zoom <= 3.0; zoom += 1.0) {
          if (zoom <= _maxZoom) {
            _availableZoomLevels.add(zoom);
          }
        }
        if (_availableZoomLevels.isEmpty) {
          _availableZoomLevels = [1.0];
        }
        _currentZoom = 1.0;

        setState(() => _isInitialized = true);
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      setState(() {
        _resolutionInfo = 'Neural Vision';
        _fpsInfo = 'Engaged';
      });
    }
  }

  void _toggleMaterialSelector() {
    HapticFeedback.lightImpact();
    setState(() => _materialSelectorExpanded = !_materialSelectorExpanded);
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    HapticFeedback.lightImpact();
    try {
      final newFlashMode = _flashOn ? FlashMode.off : FlashMode.torch;
      await _controller!.setFlashMode(newFlashMode);
      if (mounted) {
        setState(() => _flashOn = !_flashOn);
      }
    } catch (e) {
      debugPrint('Error toggling flash: $e');
    }
  }

  Future<void> _applyExposure() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      await _controller!.setExposureOffset(_exposureValue);
    } catch (e) {
      debugPrint('Error setting exposure: $e');
    }
  }

  Future<void> _setZoom(double targetZoom) async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      HapticFeedback.lightImpact();
      await _controller!.setZoomLevel(targetZoom);
      setState(() => _currentZoom = targetZoom);
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

  void _openCapturedImages() {
    if (_capturedImages.isEmpty) return;

    HapticFeedback.lightImpact();
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CapturedImagesScreen(imagePaths: _capturedImages),
      ),
    );
  }

  Future<void> _capturePhoto() async {
    if (_controller == null || !_controller!.value.isInitialized || _isCapturing) {
      return;
    }

    try {
      setState(() => _isCapturing = true);
      HapticFeedback.heavyImpact();

      _scanAnimationController.forward(from: 0.0);
      _captureController.forward();

      // Capture in background
      _controller!.takePicture().then((XFile image) {
        // Add to captured images (max 3)
        if (_capturedImages.length < 3 && mounted) {
          setState(() {
            _capturedImages.add(image.path);
          });
        }
      }).catchError((e) {
        debugPrint('Error saving photo: $e');
      });

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
    SystemChrome.setSystemUIOverlayStyle(
      const SystemUiOverlayStyle(
        systemNavigationBarColor: Color(0xFF000000),
        systemNavigationBarIconBrightness: Brightness.light,
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
      ),
    );

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A1A),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF2C2C2C),
              borderRadius: BorderRadius.circular(40),
              border: Border.all(color: const Color(0xFF444444), width: 1),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.4),
                  blurRadius: 12,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Stack(
                children: [
                  Column(
                    children: [
                      // Top controls row
                      _buildTopControls(),

                      const SizedBox(height: 12),

                      // Resolution and FPS info
                      _buildResolutionInfo(),

                      const SizedBox(height: 16),

                      // Camera viewfinder with aspect ratio frame
                      Expanded(
                        child: Stack(
                          children: [
                            _buildCameraView(),
                            _buildFlashAnimation(),
                          ],
                        ),
                      ),

                      const SizedBox(height: 20),

                      // Bottom controls
                      _buildBottomControls(),
                    ],
                  ),

                  // Material selector overlay
                  if (_materialSelectorExpanded)
                    Positioned.fill(
                      child: GestureDetector(
                        onTap: () {
                          setState(() => _materialSelectorExpanded = false);
                        },
                        child: Container(
                          color: Colors.transparent,
                          child: Center(
                            child: GestureDetector(
                              onTap: () {}, // Prevent clicks from closing when tapping menu
                              child: Container(
                                margin: const EdgeInsets.symmetric(horizontal: 40, vertical: 40),
                                padding: const EdgeInsets.all(20),
                                decoration: BoxDecoration(
                                  color: const Color(0xFF2C2C2C),
                                  borderRadius: BorderRadius.circular(20),
                                  border: Border.all(
                                    color: const Color(0xFF444444),
                                    width: 1,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.5),
                                      blurRadius: 20,
                                      spreadRadius: 5,
                                      offset: const Offset(0, 10),
                                    ),
                                  ],
                                ),
                                child: SingleChildScrollView(
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: _materialTypes.map((type) {
                                      final isSelected = _selectedMaterial == type;
                                      return GestureDetector(
                                        onTap: () {
                                          HapticFeedback.lightImpact();
                                          setState(() {
                                            _selectedMaterial = type;
                                            _materialSelectorExpanded = false;
                                          });

                                          // Show toast at top
                                          ScaffoldMessenger.of(context).showSnackBar(
                                            SnackBar(
                                              content: Row(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(
                                                    Icons.folder_outlined,
                                                    color: const Color(0xFFea580c),
                                                    size: 16,
                                                  ),
                                                  const SizedBox(width: 8),
                                                  Text(
                                                    'Saved under $type',
                                                    style: GoogleFonts.robotoMono(
                                                      fontWeight: FontWeight.w600,
                                                      fontSize: 11,
                                                      color: Colors.white,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                              duration: const Duration(milliseconds: 1500),
                                              backgroundColor: const Color(0xFF2C2C2C),
                                              behavior: SnackBarBehavior.floating,
                                              margin: EdgeInsets.only(
                                                top: MediaQuery.of(context).padding.top + 10,
                                                left: 40,
                                                right: 40,
                                                bottom: MediaQuery.of(context).size.height - 100,
                                              ),
                                              shape: RoundedRectangleBorder(
                                                borderRadius: BorderRadius.circular(20),
                                                side: BorderSide(
                                                  color: const Color(0xFFea580c),
                                                  width: 1.5,
                                                ),
                                              ),
                                              elevation: 8,
                                            ),
                                          );
                                        },
                                        child: Container(
                                          width: double.infinity,
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 20,
                                            vertical: 14,
                                          ),
                                          margin: const EdgeInsets.only(bottom: 8),
                                          decoration: BoxDecoration(
                                            color: isSelected
                                                ? const Color(0xFFea580c)
                                                : const Color(0xFF444444),
                                            borderRadius: BorderRadius.circular(12),
                                          ),
                                          child: Text(
                                            type,
                                            textAlign: TextAlign.center,
                                            style: GoogleFonts.robotoMono(
                                              color: Colors.white,
                                              fontSize: 14,
                                              fontWeight: FontWeight.w700,
                                              letterSpacing: 1.2,
                                            ),
                                          ),
                                        ),
                                      );
                                    }).toList(),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTopControls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // Material selector button
        Flexible(
          child: GestureDetector(
            onTap: _toggleMaterialSelector,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFF444444),
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: const Color(0xFF222222), width: 1),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.5),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.label_outline, color: Colors.white, size: 14),
                  const SizedBox(width: 6),
                  Text(
                    _selectedMaterial,
                    style: GoogleFonts.robotoMono(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        const SizedBox(width: 8),

        // Aspect ratio selector
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 3),
          decoration: BoxDecoration(
            color: const Color(0xFF444444),
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: const Color(0xFF222222), width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: ['3:4', '9:16', 'Full'].map((ratio) {
              final isSelected = _aspectRatio == ratio;
              return GestureDetector(
                onTap: () {
                  HapticFeedback.lightImpact();
                  setState(() => _aspectRatio = ratio);
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: isSelected ? const Color(0xFF666666) : Colors.transparent,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    ratio,
                    style: GoogleFonts.robotoMono(
                      color: Colors.white,
                      fontSize: 9,
                      fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ),

        const SizedBox(width: 8),

        // Flash toggle
        GestureDetector(
          onTap: _toggleFlash,
          child: Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: _flashOn ? const Color(0xFFea580c) : const Color(0xFF444444),
              shape: BoxShape.circle,
              border: Border.all(color: const Color(0xFF222222), width: 1),
            ),
            child: Icon(
              _flashOn ? Icons.flash_on : Icons.flash_off,
              color: Colors.white,
              size: 20,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCameraView() {
    double aspectRatioValue = 3 / 4; // Default 3:4
    if (_aspectRatio == '9:16') {
      aspectRatioValue = 9 / 16;
    }

    return Center(
      child: _aspectRatio == 'Full'
          ? _buildFullCameraView()
          : AspectRatio(
              aspectRatio: aspectRatioValue,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFF444444), width: 2),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: _isInitialized && _controller != null
                      ? Stack(
                          fit: StackFit.expand,
                          children: [
                            CameraPreview(_controller!),
                            _buildReticle(),
                          ],
                        )
                      : Center(
                          child: CircularProgressIndicator(
                            color: const Color(0xFFea580c),
                            strokeWidth: 2,
                          ),
                        ),
                ),
              ),
            ),
    );
  }

  Widget _buildFullCameraView() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF444444), width: 2),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(18),
        child: _isInitialized && _controller != null
            ? FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _controller!.value.previewSize!.height,
                  height: _controller!.value.previewSize!.width,
                  child: CameraPreview(_controller!),
                ),
              )
            : Center(
                child: CircularProgressIndicator(
                  color: const Color(0xFFea580c),
                  strokeWidth: 2,
                ),
              ),
      ),
    );
  }

  Widget _buildReticle() {
    return Center(
      child: AnimatedBuilder(
        animation: _focusAnimation,
        builder: (context, child) {
          return Opacity(
            opacity: _focusAnimation.value,
            child: Container(
              width: 200,
              height: 200,
              decoration: BoxDecoration(
                border: Border.all(
                  color: const Color(0xFFfbbf24),
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(4),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildFlashAnimation() {
    if (!_isCapturing) return const SizedBox.shrink();

    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _scanLineAnimation,
        builder: (context, child) {
          double opacity = 0.0;
          if (_scanLineAnimation.value < 0.1) {
            opacity = _scanLineAnimation.value / 0.1;
          } else if (_scanLineAnimation.value < 0.3) {
            opacity = (0.3 - _scanLineAnimation.value) / 0.2;
          }

          return IgnorePointer(
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(opacity),
                borderRadius: BorderRadius.circular(18),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildBottomControls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // Left side: Gallery and image progress
        Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildControlButton(
              icon: Icons.photo_library,
              onTap: _pickFromGallery,
            ),
            const SizedBox(height: 12),
            _buildCapturedImagesPreview(),
          ],
        ),

        // Shutter button
        GestureDetector(
          onTap: _capturePhoto,
          child: Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFFD32F2F),
              border: Border.all(color: const Color(0xFF8B0000), width: 2),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.5),
                  blurRadius: 6,
                  offset: const Offset(0, 4),
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

        // Right side: Zoom and EV
        Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Zoom selector
            if (_availableZoomLevels.length > 1)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF444444),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: const Color(0xFF222222), width: 1),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: _availableZoomLevels.map((zoom) {
                    final isSelected = _currentZoom == zoom;
                    return GestureDetector(
                      onTap: () => _setZoom(zoom),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: isSelected ? const Color(0xFF666666) : Colors.transparent,
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Text(
                          '${zoom.toInt()}×',
                          style: GoogleFonts.robotoMono(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              )
            else
              const SizedBox(width: 56),

            const SizedBox(height: 12),

            // EV control
            _buildEVControl(),
          ],
        ),
      ],
    );
  }

  Widget _buildEVControl() {
    final showValue = _exposureValue.abs() > 0.05; // Only show if not essentially zero
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          decoration: BoxDecoration(
            color: const Color(0xFF444444),
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: const Color(0xFF222222), width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              GestureDetector(
                onTap: () {
                  if (_exposureValue > _minExposure) {
                    HapticFeedback.selectionClick();
                    setState(() {
                      _exposureValue = (_exposureValue - 0.2).clamp(_minExposure, _maxExposure);
                    });
                    _applyExposure();
                  }
                },
                child: Container(
                  padding: const EdgeInsets.all(4),
                  child: Icon(Icons.remove, color: Colors.white, size: 14),
                ),
              ),
              const SizedBox(width: 4),
              Text(
                'EV',
                style: GoogleFonts.robotoMono(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(width: 4),
              GestureDetector(
                onTap: () {
                  if (_exposureValue < _maxExposure) {
                    HapticFeedback.selectionClick();
                    setState(() {
                      _exposureValue = (_exposureValue + 0.2).clamp(_minExposure, _maxExposure);
                    });
                    _applyExposure();
                  }
                },
                child: Container(
                  padding: const EdgeInsets.all(4),
                  child: Icon(Icons.add, color: Colors.white, size: 14),
                ),
              ),
            ],
          ),
        ),
        if (showValue) ...[
          const SizedBox(height: 4),
          Text(
            _exposureValue > 0 ? '+${_exposureValue.toStringAsFixed(1)}' : _exposureValue.toStringAsFixed(1),
            style: GoogleFonts.robotoMono(
              color: const Color(0xFF999999),
              fontSize: 9,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildResolutionInfo() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          _resolutionInfo,
          style: GoogleFonts.robotoMono(
            color: const Color(0xFFea580c),
            fontSize: 12,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.5,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '•',
          style: GoogleFonts.robotoMono(
            color: const Color(0xFF666666),
            fontSize: 12,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          _fpsInfo,
          style: GoogleFonts.robotoMono(
            color: const Color(0xFF999999),
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 1.2,
          ),
        ),
      ],
    );
  }

  Widget _buildCapturedImagesPreview() {
    final isComplete = _capturedImages.length == 3;

    return GestureDetector(
      onTap: _capturedImages.isNotEmpty ? _openCapturedImages : null,
      child: Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          color: const Color(0xFF444444),
          shape: BoxShape.circle,
          border: Border.all(
            color: isComplete ? const Color(0xFFea580c) : const Color(0xFF222222),
            width: isComplete ? 3 : 1,
          ),
          boxShadow: isComplete
              ? [
                  BoxShadow(
                    color: const Color(0xFFea580c).withOpacity(0.5),
                    blurRadius: 12,
                    spreadRadius: 2,
                  ),
                ]
              : [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.5),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(3, (index) {
              final isCaptured = index < _capturedImages.length;
              return Container(
                margin: const EdgeInsets.symmetric(vertical: 2),
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isCaptured
                      ? const Color(0xFFea580c)
                      : const Color(0xFF666666),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }

  Widget _buildControlButton({required IconData icon, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          color: const Color(0xFF444444),
          shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFF222222), width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.5),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Icon(icon, color: Colors.white, size: 24),
      ),
    );
  }
}
