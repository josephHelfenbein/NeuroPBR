import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import 'dart:io';
import 'captured_images_screen.dart';
import '../theme/theme_provider.dart';
import '../services/native_image_processor.dart';

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
  final String _aspectRatio = '1:1';
  List<String> _capturedImages = [];

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
    
    // Keep screen on while camera is active
    WakelockPlus.enable();

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
    // Allow screen to dim again
    WakelockPlus.disable();
    
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
        ResolutionPreset.ultraHigh,
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
      final List<XFile> images = await picker.pickMultiImage();
      if (images.isNotEmpty && mounted) {
        // Show loading indicator
        setState(() => _isCapturing = true);
        
        // Process each picked image using native Swift processor
        for (final XFile image in images) {
          try {
            // Use native Swift processor (much faster than Dart)
            final processedPath = await NativeImageProcessor.processImageFile(
              image.path,
              targetSize: 2048,
              quality: 0.92,
            );
            
            if (processedPath != null && mounted) {
              setState(() {
                _capturedImages.add(processedPath);
              });
            } else if (mounted) {
              // Fallback: use original if processing fails
              setState(() {
                _capturedImages.add(image.path);
              });
            }
          } catch (e) {
            debugPrint('Error processing gallery image: $e');
            if (mounted) {
              setState(() {
                _capturedImages.add(image.path);
              });
            }
          }
        }

        if (mounted) {
          setState(() => _isCapturing = false);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Added ${images.length} image${images.length > 1 ? 's' : ''}',
                style: GoogleFonts.robotoMono(
                  fontWeight: FontWeight.w500,
                  color: Colors.white,
                ),
              ),
              duration: const Duration(seconds: 2),
              backgroundColor: const Color(0xFF1a1a1a),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Error picking image: $e');
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _openCapturedImages() async {
    if (_capturedImages.isEmpty) return;

    HapticFeedback.lightImpact();

    // We use 'await' to pause here until the user comes back
    final updatedList = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CapturedImagesScreen(imagePaths: _capturedImages),
      ),
    );

    // When they return, we check if they sent back a list
    if (updatedList != null && updatedList is List<String> && mounted) {
      setState(() {
        _capturedImages = updatedList;
      });
    }
  }

  Future<void> _capturePhoto() async {
    // 1. Hardware Check & Initial Guard
    if (_controller == null || !_controller!.value.isInitialized || _controller!.value.isTakingPicture || _isCapturing) {
      return;
    }

    // 2. INSTANT FEEDBACK: Haptic + Animation immediately
    HapticFeedback.heavyImpact();
    setState(() => _isCapturing = true);
    _scanAnimationController.forward(from: 0.0);
    _captureController.forward();

    try {
      // 3. Take picture (hardware call)
      final XFile image = await _controller!.takePicture();

      // 4. Process using native Swift processor (fast!)
      try {
        final processedPath = await NativeImageProcessor.processImageFile(
          image.path,
          targetSize: 2048,
          quality: 0.92,
        );

        // Delete the original raw capture to save space
        final originalFile = File(image.path);
        if (await originalFile.exists()) {
          await originalFile.delete();
        }

        if (processedPath != null && mounted) {
          setState(() {
            _capturedImages.add(processedPath);
          });
        } else if (mounted) {
          // Fallback if processing fails - shouldn't happen
          debugPrint('Native processing returned null');
        }
      } catch (e) {
        debugPrint('Error processing image: $e');
        if (mounted) {
          setState(() {
            _capturedImages.add(image.path);
          });
        }
      }
    } catch (e) {
      debugPrint('Error capturing photo: $e');
    } finally {
      // 5. Reset UI State and Animations
      if (mounted) {
        _captureController.reverse();
        _scanAnimationController.reset();
        setState(() => _isCapturing = false);
      }
    }
  }

  void _showMaterialToast() {
    // Toast removed per user request
  }

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;

    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        systemNavigationBarColor: colors.background,
        systemNavigationBarIconBrightness: colors.statusBarBrightness,
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: colors.statusBarBrightness,
      ),
    );

    return Scaffold(
      backgroundColor: colors.background,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                TopControls(
                  onBack: () {
                    HapticFeedback.lightImpact();
                    Navigator.pop(context);
                  },
                  selectedMaterial: _selectedMaterial,
                  flashOn: _flashOn,
                  onMaterialSelector: _toggleMaterialSelector,
                  onFlashToggle: _toggleFlash,
                ),
                const SizedBox(height: 12),
                ResolutionInfo(resolutionInfo: _resolutionInfo, fpsInfo: _fpsInfo),
                const SizedBox(height: 16),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Stack(
                      children: [
                        CameraView(
                          isInitialized: _isInitialized,
                          controller: _controller,
                          focusAnimation: _focusAnimation,
                        ),
                        FlashAnimation(
                          isCapturing: _isCapturing,
                          scanLineAnimation: _scanLineAnimation,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                BottomControls(
                  availableZoomLevels: _availableZoomLevels,
                  currentZoom: _currentZoom,
                  exposureValue: _exposureValue,
                  minExposure: _minExposure,
                  maxExposure: _maxExposure,
                  capturedImages: _capturedImages,
                  onGalleryTap: _pickFromGallery,
                  onCaptureTap: _capturePhoto,
                  onZoomChange: _setZoom,
                  onExposureChange: (value) {
                    HapticFeedback.selectionClick();
                    setState(() => _exposureValue = value);
                    _applyExposure();
                  },
                  onImagesTap: _openCapturedImages,
                ),
                const SizedBox(height: 20),
              ],
            ),
            if (_materialSelectorExpanded)
              MaterialSelectorOverlay(
                materialTypes: _materialTypes,
                selectedMaterial: _selectedMaterial,
                onMaterialSelect: (type) {
                  HapticFeedback.lightImpact();
                  setState(() {
                    _selectedMaterial = type;
                    _materialSelectorExpanded = false;
                  });
                  _showMaterialToast();
                },
                onClose: () {
                  setState(() => _materialSelectorExpanded = false);
                },
              ),
          ],
        ),
      ),
    );
  }
}

// --- Component: Top Controls ---
class TopControls extends StatelessWidget {
  final VoidCallback onBack;
  final String selectedMaterial;
  final bool flashOn;
  final VoidCallback onMaterialSelector;
  final VoidCallback onFlashToggle;

  const TopControls({
    super.key,
    required this.onBack,
    required this.selectedMaterial,
    required this.flashOn,
    required this.onMaterialSelector,
    required this.onFlashToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      color: Colors.transparent,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: onBack,
            child: Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: const Color(0xFF262626),
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0xFF3A3A3A)),
              ),
              child: const Icon(Icons.arrow_back, size: 20, color: Colors.white),
            ),
          ),
          GestureDetector(
            onTap: onMaterialSelector,
            child: Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: const Color(0xFF262626),
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0xFF3A3A3A)),
              ),
              child: const Icon(Icons.label_outline, size: 20, color: Colors.white),
            ),
          ),
          FlashToggle(flashOn: flashOn, onToggle: onFlashToggle),
        ],
      ),
    );
  }
}

// --- Component: Flash Toggle ---
class FlashToggle extends StatelessWidget {
  final bool flashOn;
  final VoidCallback onToggle;

  const FlashToggle({
    super.key,
    required this.flashOn,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;
    return GestureDetector(
      onTap: onToggle,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: flashOn ? colors.accent : colors.surface,
          shape: BoxShape.circle,
          border: Border.all(
            color: flashOn
                ? colors.accent.withOpacity(0.3)
                : Colors.white.withOpacity(0.05),
            width: 1,
          ),
          boxShadow: flashOn
              ? [
            BoxShadow(
              color: colors.accent.withOpacity(0.5),
              blurRadius: 15,
              spreadRadius: 2,
            ),
          ]
              : null,
        ),
        child: Icon(
          flashOn ? Icons.flash_on : Icons.flash_off,
          color: Colors.white,
          size: 20,
        ),
      ),
    );
  }
}

// --- Component: Resolution Info ---
class ResolutionInfo extends StatelessWidget {
  final String resolutionInfo;
  final String fpsInfo;

  const ResolutionInfo({
    super.key,
    required this.resolutionInfo,
    required this.fpsInfo,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          resolutionInfo,
          style: GoogleFonts.robotoMono(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.5,
          ),
        ),
        SizedBox(width: 8),
        Text(
          '•',
          style: TextStyle(
            color: Colors.grey[600],
            fontSize: 12,
          ),
        ),
        SizedBox(width: 8),
        Text(
          fpsInfo,
          style: GoogleFonts.robotoMono(
            color: Colors.grey[600],
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 1.2,
          ),
        ),
      ],
    );
  }
}

// --- Component: Camera View ---
class CameraView extends StatelessWidget {
  final bool isInitialized;
  final CameraController? controller;
  final Animation<double> focusAnimation;

  const CameraView({
    super.key,
    required this.isInitialized,
    required this.controller,
    required this.focusAnimation,
  });

  @override
  Widget build(BuildContext context) {
    // Locked to 1:1 aspect ratio
    return Center(
      child: AspectRatio(
        aspectRatio: 1.0,
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black,
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: Colors.white.withOpacity(0.1),
              width: 2,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.6),
                blurRadius: 20,
                spreadRadius: 5,
              ),
            ],
          ),
          clipBehavior: Clip.antiAlias,
          child: _buildCameraContent(),
        ),
      ),
    );
  }

  Widget _buildCameraContent() {
    if (!isInitialized || controller == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(
              color: Colors.white.withOpacity(0.5),
              strokeWidth: 2,
            ),
            SizedBox(height: 16),
            Text(
              'Initializing camera...',
              style: GoogleFonts.robotoMono(
                color: Colors.grey[600],
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller!),
        Center(
          child: AnimatedBuilder(
            animation: focusAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: focusAnimation.value,
                child: Container(
                  width: 120,
                  height: 120,
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Colors.white.withOpacity(0.6),
                      width: 2,
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

// --- Component: Flash Animation ---
class FlashAnimation extends StatelessWidget {
  final bool isCapturing;
  final Animation<double> scanLineAnimation;

  const FlashAnimation({
    super.key,
    required this.isCapturing,
    required this.scanLineAnimation,
  });

  @override
  Widget build(BuildContext context) {
    if (!isCapturing) return SizedBox.shrink();

    return AnimatedBuilder(
      animation: scanLineAnimation,
      builder: (context, child) {
        double whiteOpacity;
        if (scanLineAnimation.value < 0.8) {
          whiteOpacity = 0.3;
        } else {
          // Normalize the last 20% (from 0.8 to 1.0) into a 0.0 to 1.0 scale
          double fadeProgress = (scanLineAnimation.value - 0.8) * 5.0;
          whiteOpacity = 0.3 * (1.0 - fadeProgress);
        }

        return Stack(
          children: [
            Container(
              color: Colors.white.withOpacity(whiteOpacity),
            ),
            Positioned(
              top: scanLineAnimation.value * MediaQuery.of(context).size.height * 0.6,
              left: 0,
              right: 0,
              child: Container(
                height: 2,
                decoration: BoxDecoration(
                  color: Colors.red,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red.withOpacity(0.8),
                      blurRadius: 10,
                      spreadRadius: 3,
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

// --- Component: Bottom Controls ---
class BottomControls extends StatelessWidget {
  final List<double> availableZoomLevels;
  final double currentZoom;
  final double exposureValue;
  final double minExposure;
  final double maxExposure;
  final List<String> capturedImages;
  final VoidCallback onGalleryTap;
  final VoidCallback onCaptureTap;
  final Function(double) onZoomChange;
  final Function(double) onExposureChange;
  final VoidCallback onImagesTap;

  const BottomControls({
    super.key,
    required this.availableZoomLevels,
    required this.currentZoom,
    required this.exposureValue,
    required this.minExposure,
    required this.maxExposure,
    required this.capturedImages,
    required this.onGalleryTap,
    required this.onCaptureTap,
    required this.onZoomChange,
    required this.onExposureChange,
    required this.onImagesTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        children: [
          ZoomSelector(
            availableZoomLevels: availableZoomLevels,
            currentZoom: currentZoom,
            onZoomChange: onZoomChange,
          ),
          const SizedBox(height: 16),
          ExposureControl(
            exposureValue: exposureValue,
            minExposure: minExposure,
            maxExposure: maxExposure,
            onExposureChange: onExposureChange,
          ),
          const SizedBox(height: 20),
          CaptureControls(
            capturedImages: capturedImages,
            onGalleryTap: onGalleryTap,
            onCaptureTap: onCaptureTap,
            onImagesTap: onImagesTap,
          ),
        ],
      ),
    );
  }
}

// --- Component: Zoom Selector ---
class ZoomSelector extends StatelessWidget {
  final List<double> availableZoomLevels;
  final double currentZoom;
  final Function(double) onZoomChange;

  const ZoomSelector({
    super.key,
    required this.availableZoomLevels,
    required this.currentZoom,
    required this.onZoomChange,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: const Color(0xFF262626),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.05), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: availableZoomLevels.map((zoom) {
          final isSelected = (currentZoom - zoom).abs() < 0.01;
          return GestureDetector(
            onTap: () => onZoomChange(zoom),
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              margin: EdgeInsets.symmetric(horizontal: 2),
              decoration: BoxDecoration(
                color: isSelected ? Colors.white.withOpacity(0.1) : Colors.transparent,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                '${zoom.toStringAsFixed(0)}x',
                style: GoogleFonts.robotoMono(
                  color: isSelected ? Colors.white : Colors.grey[600],
                  fontSize: 12,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// --- Component: Exposure Control ---
class ExposureControl extends StatelessWidget {
  final double exposureValue;
  final double minExposure;
  final double maxExposure;
  final Function(double) onExposureChange;

  const ExposureControl({
    super.key,
    required this.exposureValue,
    required this.minExposure,
    required this.maxExposure,
    required this.onExposureChange,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(Icons.brightness_6, color: Colors.grey[600], size: 18),
        SizedBox(width: 12),
        Expanded(
          child: SliderTheme(
            data: SliderThemeData(
              activeTrackColor: Colors.white,
              inactiveTrackColor: Colors.white.withOpacity(0.1),
              thumbColor: Colors.white,
              overlayColor: Colors.white.withOpacity(0.2),
              trackHeight: 3,
              thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
            ),
            child: Slider(
              value: exposureValue,
              min: minExposure,
              max: maxExposure,
              onChanged: onExposureChange,
            ),
          ),
        ),
        SizedBox(width: 12),
        SizedBox(
          width: 40,
          child: Text(
            exposureValue.toStringAsFixed(1),
            style: GoogleFonts.robotoMono(
              color: Colors.grey[600],
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
            textAlign: TextAlign.end,
          ),
        ),
      ],
    );
  }
}

// --- Component: Capture Controls ---
class CaptureControls extends StatelessWidget {
  final List<String> capturedImages;
  final VoidCallback onGalleryTap;
  final VoidCallback onCaptureTap;
  final VoidCallback onImagesTap;

  const CaptureControls({
    super.key,
    required this.capturedImages,
    required this.onGalleryTap,
    required this.onCaptureTap,
    required this.onImagesTap,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        GestureDetector(
          onTap: onGalleryTap,
          child: Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: colors.surface,
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white.withOpacity(0.1), width: 2),
            ),
            child: Icon(Icons.photo_library, color: Colors.white, size: 24),
          ),
        ),
        GestureDetector(
          onTap: onCaptureTap,
          child: Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: colors.accent,
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white.withOpacity(0.2), width: 4),
              boxShadow: [
                BoxShadow(
                  color: colors.accent.withOpacity(0.5),
                  blurRadius: 20,
                  spreadRadius: 3,
                ),
              ],
            ),
            child: Icon(Icons.camera_alt, color: Colors.white, size: 32),
          ),
        ),
        GestureDetector(
          onTap: onImagesTap,
          child: Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: colors.surface,
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white.withOpacity(0.1), width: 2),
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                Icon(Icons.photo, color: Colors.white, size: 24),
                if (capturedImages.isNotEmpty)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: EdgeInsets.all(4),
                      decoration: BoxDecoration(
                        color: colors.accent,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: colors.accent.withOpacity(0.5),
                            blurRadius: 8,
                            spreadRadius: 1,
                          ),
                        ],
                      ),
                      constraints: BoxConstraints(minWidth: 18, minHeight: 18),
                      child: Center(
                        child: Text(
                          '${capturedImages.length}',
                          style: GoogleFonts.robotoMono(
                            color: Colors.white,
                            fontSize: 9,
                            fontWeight: FontWeight.bold,
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
    );
  }
}

// --- Component: Material Selector Overlay ---
class MaterialSelectorOverlay extends StatelessWidget {
  final List<String> materialTypes;
  final String selectedMaterial;
  final Function(String) onMaterialSelect;
  final VoidCallback onClose;

  const MaterialSelectorOverlay({
    super.key,
    required this.materialTypes,
    required this.selectedMaterial,
    required this.onMaterialSelect,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;
    return GestureDetector(
      onTap: onClose,
      child: Container(
        color: Colors.black.withOpacity(0.8),
        child: Center(
          child: GestureDetector(
            onTap: () {},
            child: Container(
              margin: EdgeInsets.symmetric(horizontal: 32),
              padding: EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: colors.surface,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white.withOpacity(0.1), width: 1),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.6),
                    blurRadius: 30,
                    spreadRadius: 10,
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'SELECT MATERIAL',
                        style: GoogleFonts.robotoMono(
                          color: Colors.grey[600],
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 2,
                        ),
                      ),
                      GestureDetector(
                        onTap: onClose,
                        child: Icon(Icons.close, color: Colors.grey[600], size: 20),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                  ...materialTypes.map((material) {
                    final isSelected = material == selectedMaterial;
                    return GestureDetector(
                      onTap: () => onMaterialSelect(material),
                      child: Container(
                        margin: EdgeInsets.only(bottom: 12),
                        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? Colors.white.withOpacity(0.1)
                              : Colors.transparent,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: isSelected
                                ? Colors.white.withOpacity(0.2)
                                : Colors.transparent,
                            width: 1,
                          ),
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              decoration: BoxDecoration(
                                color: isSelected ? colors.accent : Colors.grey[700],
                                shape: BoxShape.circle,
                                boxShadow: isSelected
                                    ? [
                                  BoxShadow(
                                    color: colors.accent.withOpacity(0.5),
                                    blurRadius: 8,
                                    spreadRadius: 1,
                                  ),
                                ]
                                    : null,
                              ),
                            ),
                            SizedBox(width: 16),
                            Text(
                              material,
                              style: GoogleFonts.robotoMono(
                                color: isSelected ? Colors.white : Colors.grey[600],
                                fontSize: 14,
                                fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                                letterSpacing: 1.2,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

