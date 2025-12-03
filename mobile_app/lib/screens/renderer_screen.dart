import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import '../neuropbr_plugin.dart';

class RendererScreen extends StatefulWidget {
  final String? materialPath;
  final bool isAsset;

  const RendererScreen({
    super.key,
    this.materialPath,
    this.isAsset = false,
  });

  @override
  State<RendererScreen> createState() => _RendererScreenState();
}

class _RendererScreenState extends State<RendererScreen> {
  final NeuropbrRenderer _renderer = NeuropbrRenderer.instance;
  bool _isInitialized = false;
  String? _errorMessage;

  // Camera State
  double _cameraTheta = 0.0; // Horizontal angle
  double _cameraPhi = math.pi / 2; // Vertical angle
  double _cameraDistance = 8.0;
  double _baseScale = 1.0;
  double _lastScale = 1.0;

  // Scene State
  NeuropbrModelType _currentModel = NeuropbrModelType.sphere;
  String _currentEnvMap = 'sunny_rose_garden_8k';
  
  // Available environment maps (base names without extension)
  final List<String> _availableEnvMaps = [
    'furry_clouds_8k',
    'large_corridor_8k',
    'sunny_rose_garden_8k',
    'the_sky_is_on_fire_8k',
  ];

  // Cache for copied asset paths
  final Map<String, Map<String, String>> _envMapPaths = {};
  
  final List<String> _defaultTextures = [
    'albedo.png',
    'metallic.png',
    'normal.png',
    'roughness.png',
  ];
  final Map<String, String> _texturePaths = {};
  
  // Shared BRDF LUT path
  String? _brdfLutPath;

  @override
  void initState() {
    super.initState();
    _initializeRenderer();
  }

  Future<void> _initializeRenderer() async {
    try {
      // 1. Initialize Renderer
      // Get the physical size of the screen to avoid stretching
      final view = ui.PlatformDispatcher.instance.views.first;
      final physicalSize = view.physicalSize;
      
      await _renderer.initRenderer(
        width: physicalSize.width.toInt(), 
        height: physicalSize.height.toInt()
      );

      // 2. Setup Initial Camera
      _updateCamera();

      // 3. Setup Initial Lighting/Preview
      await _renderer.setLighting(const NeuropbrLighting(
        exposure: 1.0,
        intensity: 0.1,
        rotation: 0.0,
      ));

      await _renderer.setPreviewControls(NeuropbrPreviewControls(
        tint: const [1.0, 1.0, 1.0],
        roughnessMultiplier: 1.0,
        metallicMultiplier: 1.0,
        toneMapping: NeuropbrToneMapping.filmic,
        modelType: _currentModel,
        zoom: 1.0,
      ));

      // 4. Load Material
      if (widget.materialPath != null) {
        await _loadMaterialFromPath(widget.materialPath!, widget.isAsset);
      } else {
        await _loadDefaultMaterial();
      }

      // 5. Prepare and Load Initial Environment
      await _prepareEnvMaps();
      await _loadEnvironment(_currentEnvMap);

      setState(() {
        _isInitialized = true;
      });

      // Start rendering loop or trigger initial frame
      _renderer.renderFrame('default_mat');

    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
      });
      debugPrint('Renderer Initialization Error: $e');
    }
  }

  Future<void> _prepareDefaultTextures() async {
    final tempDir = await getTemporaryDirectory();
    for (final texName in _defaultTextures) {
      try {
        final file = File('${tempDir.path}/$texName');
        // Always overwrite to ensure fresh assets during dev, or check exists for prod
        // For now, we'll overwrite if it doesn't exist or just overwrite.
        // Let's overwrite to be safe.
        final byteData = await rootBundle.load('assets/default_tex/$texName');
        await file.writeAsBytes(byteData.buffer.asUint8List());
        _texturePaths[texName] = file.path;
      } catch (e) {
        debugPrint('Failed to copy texture $texName: $e');
      }
    }
  }

  Future<void> _loadMaterialFromPath(String path, bool isAsset) async {
    try {
      // Helper to find file with allowed extensions
      String? findFile(String basePath, String name) {
        const extensions = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG'];
        for (final ext in extensions) {
          final filePath = '$basePath/$name.$ext';
          if (File(filePath).existsSync()) {
            return filePath;
          }
        }
        return null;
      }

      String? albedoPath, normalPath, roughnessPath, metallicPath;

      if (isAsset) {
        if (path.contains('default_tex')) {
          await _prepareDefaultTextures();
          albedoPath = _texturePaths['albedo.png'];
          normalPath = _texturePaths['normal.png'];
          roughnessPath = _texturePaths['roughness.png'];
          metallicPath = _texturePaths['metallic.png'];
        } else {
          // Handle other assets if needed
          debugPrint('Unsupported asset path: $path');
          return;
        }
      } else {
        // Filesystem path - search for files with allowed extensions
        albedoPath = findFile(path, 'albedo');
        normalPath = findFile(path, 'normal');
        roughnessPath = findFile(path, 'roughness');
        metallicPath = findFile(path, 'metallic');
      }

      // Helper to create payload if path is valid
      NeuropbrTexturePayload? createPayload(String? filePath, {int channels = 4}) {
        if (filePath == null) return null;
        
        // We don't need actual dimensions for file paths, the native loader handles it.
        // Passing 1x1 to satisfy the constructor.
        return NeuropbrTexturePayload.fromFile(
          filePath,
          width: 1,
          height: 1,
          format: 'rgba8unorm',
          channels: channels,
        );
      }

      final albedoPayload = createPayload(albedoPath);
      final normalPayload = createPayload(normalPath);
      final roughnessPayload = createPayload(roughnessPath, channels: 1);
      final metallicPayload = createPayload(metallicPath, channels: 1);

      // If no textures found at all, fallback to default
      if (albedoPayload == null && normalPayload == null && 
          roughnessPayload == null && metallicPayload == null) {
        debugPrint('No textures found in $path, loading default material');
        await _loadDefaultMaterial();
        return;
      }

      await _renderer.loadMaterial('default_mat', NeuropbrMaterialTextures(
        albedo: albedoPayload,
        normal: normalPayload,
        roughness: roughnessPayload,
        metallic: metallicPayload,
      ));

    } catch (e) {
      debugPrint('Error loading material from path $path: $e');
      // Fallback to default
      await _loadDefaultMaterial();
    }
  }

  Future<void> _loadDefaultMaterial() async {
    // Try loading from assets first
    try {
      await _prepareDefaultTextures();
      
      final albedoPath = _texturePaths['albedo.png'];
      final normalPath = _texturePaths['normal.png'];
      final roughnessPath = _texturePaths['roughness.png'];
      final metallicPath = _texturePaths['metallic.png'];

      if (albedoPath != null && normalPath != null && roughnessPath != null && metallicPath != null) {
        await _renderer.loadMaterial('default_mat', NeuropbrMaterialTextures(
          albedo: NeuropbrTexturePayload.fromFile(
            albedoPath,
            width: 1,
            height: 1,
            format: 'rgba8unorm',
          ),
          normal: NeuropbrTexturePayload.fromFile(
            normalPath,
            width: 1,
            height: 1,
            format: 'rgba8unorm',
          ),
          roughness: NeuropbrTexturePayload.fromFile(
            roughnessPath,
            width: 1,
            height: 1,
            format: 'rgba8unorm',
            channels: 1,
          ),
          metallic: NeuropbrTexturePayload.fromFile(
            metallicPath,
            width: 1,
            height: 1,
            format: 'rgba8unorm',
            channels: 1,
          ),
        ));
        return;
      }
    } catch (e) {
      debugPrint('Error loading default textures from assets: $e');
      // Fallthrough to generated textures
    }

    // Create a simple default material (grey-ish)
    final size = 256;
    final pixelCount = size * size;
    
    // Helper to create a solid color texture
    Uint8List createSolidTexture(double r, double g, double b, double a) {
      final list = Float32List(pixelCount * 4);
      for (int i = 0; i < pixelCount * 4; i += 4) {
        list[i] = r;
        list[i + 1] = g;
        list[i + 2] = b;
        list[i + 3] = a;
      }
      return Uint8List.view(list.buffer);
    }

    // Helper to create a single channel texture
    Uint8List createSingleChannelTexture(double value) {
      final list = Float32List(pixelCount);
      for (int i = 0; i < pixelCount; i++) {
        list[i] = value;
      }
      return Uint8List.view(list.buffer);
    }

    await _renderer.loadMaterial('default_mat', NeuropbrMaterialTextures(
      albedo: NeuropbrTexturePayload.fromBytes(
        createSolidTexture(0.7, 0.7, 0.7, 1.0),
        width: size,
        height: size,
        format: 'rgba32float',
      ),
      normal: NeuropbrTexturePayload.fromBytes(
        createSolidTexture(0.5, 0.5, 1.0, 1.0), // Flat normal
        width: size,
        height: size,
        format: 'rgba32float',
      ),
      roughness: NeuropbrTexturePayload.fromBytes(
        createSingleChannelTexture(0.5),
        width: size,
        height: size,
        format: 'r32float',
        channels: 1,
      ),
      metallic: NeuropbrTexturePayload.fromBytes(
        createSingleChannelTexture(0.0),
        width: size,
        height: size,
        format: 'r32float',
        channels: 1,
      ),
    ));
  }

  Future<void> _prepareEnvMaps() async {
    final tempDir = await getTemporaryDirectory();
    
    // Copy BRDF LUT first (shared across all environments)
    try {
      final brdfByteData = await rootBundle.load('assets/env_maps/brdf_lut.ktx');
      final brdfFile = File('${tempDir.path}/brdf_lut.ktx');
      await brdfFile.writeAsBytes(brdfByteData.buffer.asUint8List());
      _brdfLutPath = brdfFile.path;
    } catch (e) {
      debugPrint('Failed to copy BRDF LUT: $e');
    }
    
    // Copy environment map files for each environment
    for (final envMap in _availableEnvMaps) {
      try {
        final paths = <String, String>{};
        
        // Environment cubemap
        final envByteData = await rootBundle.load('assets/env_maps/${envMap}_env.ktx');
        final envFile = File('${tempDir.path}/${envMap}_env.ktx');
        await envFile.writeAsBytes(envByteData.buffer.asUint8List());
        paths['environment'] = envFile.path;
        
        // Irradiance cubemap
        final irrByteData = await rootBundle.load('assets/env_maps/${envMap}_irradiance.ktx');
        final irrFile = File('${tempDir.path}/${envMap}_irradiance.ktx');
        await irrFile.writeAsBytes(irrByteData.buffer.asUint8List());
        paths['irradiance'] = irrFile.path;
        
        // Prefiltered cubemap
        final pfByteData = await rootBundle.load('assets/env_maps/${envMap}_prefiltered.ktx');
        final pfFile = File('${tempDir.path}/${envMap}_prefiltered.ktx');
        await pfFile.writeAsBytes(pfByteData.buffer.asUint8List());
        paths['prefiltered'] = pfFile.path;
        
        _envMapPaths[envMap] = paths;
      } catch (e) {
        debugPrint('Failed to copy environment maps for $envMap: $e');
      }
    }
  }

  Future<void> _loadEnvironment(String envMapName) async {
    final paths = _envMapPaths[envMapName];
    if (paths == null) {
      debugPrint('Environment maps not found for $envMapName');
      return;
    }
    
    await _renderer.setEnvironment(NeuropbrEnvironment(
      environmentPath: paths['environment'],
      irradiancePath: paths['irradiance'],
      prefilteredPath: paths['prefiltered'],
      brdfPath: _brdfLutPath,
    ));
    
    if (_isInitialized) {
      _renderer.renderFrame('default_mat');
    }
  }

  void _updateCamera() {
    // Convert spherical coordinates to Cartesian
    final x = _cameraDistance * math.sin(_cameraPhi) * math.sin(_cameraTheta);
    final y = _cameraDistance * math.cos(_cameraPhi);
    final z = _cameraDistance * math.sin(_cameraPhi) * math.cos(_cameraTheta);

    _renderer.setCamera(NeuropbrCamera(
      position: [x, y, z],
      target: [0.0, 0.0, 0.0],
      up: [0.0, 1.0, 0.0],
      fov: 45.0,
    ));
    
    if (_isInitialized) {
      _renderer.renderFrame('default_mat');
    }
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _baseScale = _cameraDistance;
    _lastScale = 1.0;
  }

  void _handleScaleUpdate(ScaleUpdateDetails details) {
    // Orbit (Rotation)
    // Adjust sensitivity as needed
    final sensitivity = 0.01;
    setState(() {
      _cameraTheta -= details.focalPointDelta.dx * sensitivity;
      _cameraPhi -= details.focalPointDelta.dy * sensitivity;

      // Clamp vertical angle to avoid flipping
      _cameraPhi = _cameraPhi.clamp(0.1, math.pi - 0.1);
    });

    // Zoom (Pinch)
    if (details.scale != 1.0) {
      // Calculate scale delta relative to the last frame of the gesture
      // This prevents jumps when switching from pan to pinch
      // Simple approach: use total scale from start
      
      // Inverted scale for intuitive zoom (pinch out -> zoom in/closer)
      // Actually, usually pinch out (scale > 1) means zoom in (make object larger), 
      // which means decreasing distance.
      
      double newDistance = _baseScale / details.scale;
      newDistance = newDistance.clamp(1.5, 10.0); // Clamp zoom range
      
      setState(() {
        _cameraDistance = newDistance;
      });
    }

    _updateCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // 1. Renderer View
          if (_isInitialized)
            Positioned.fill(
              child: GestureDetector(
                onScaleStart: _handleScaleStart,
                onScaleUpdate: _handleScaleUpdate,
                child: _renderer.buildPreviewTexture(),
              ),
            )
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.orange),
            ),

          if (_errorMessage != null)
            Center(child: Text('Error: $_errorMessage', style: const TextStyle(color: Colors.red))),

          // 2. UI Overlays
          // Top Bar: Back & Title
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(
                        Icons.arrow_back, 
                        color: Colors.white,
                        shadows: [
                          Shadow(
                            color: Colors.black,
                            blurRadius: 4.0,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      '3D Preview',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            color: Colors.black,
                            blurRadius: 4.0,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Bottom Controls
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withOpacity(0.8),
                  ],
                ),
              ),
              child: SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Model Selection
                      const Text(
                        'MODEL',
                        style: TextStyle(
                          color: Colors.grey,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          _buildModelBtn(NeuropbrModelType.sphere, Icons.circle, 'Sphere'),
                          const SizedBox(width: 10),
                          _buildModelBtn(NeuropbrModelType.cube, Icons.square, 'Cube'),
                          const SizedBox(width: 10),
                          _buildModelBtn(NeuropbrModelType.plane, Icons.layers, 'Plane'),
                        ],
                      ),

                      const SizedBox(height: 20),

                      // HDRI Selection
                      const Text(
                        'ENVIRONMENT',
                        style: TextStyle(
                          color: Colors.grey,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 10),
                      SizedBox(
                        height: 80,
                        child: ListView.builder(
                          scrollDirection: Axis.horizontal,
                          itemCount: _availableEnvMaps.length,
                          itemBuilder: (context, index) {
                            final envMap = _availableEnvMaps[index];
                            final isSelected = _currentEnvMap == envMap;
                            // Clean up name for display
                            final displayName = envMap
                                .replaceAll('_8k', '')
                                .replaceAll('_', ' ')
                                .toUpperCase();

                            return GestureDetector(
                              onTap: () {
                                setState(() => _currentEnvMap = envMap);
                                _loadEnvironment(envMap);
                              },
                              child: Container(
                                width: 100,
                                margin: const EdgeInsets.only(right: 10),
                                decoration: BoxDecoration(
                                  color: isSelected ? Colors.orange : Colors.grey[900],
                                  borderRadius: BorderRadius.circular(12),
                                  border: isSelected 
                                    ? Border.all(color: Colors.white, width: 2)
                                    : Border.all(color: Colors.white24),
                                ),
                                padding: const EdgeInsets.all(8),
                                child: Center(
                                  child: Text(
                                    displayName,
                                    textAlign: TextAlign.center,
                                    style: TextStyle(
                                      color: isSelected ? Colors.white : Colors.grey[400],
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModelBtn(NeuropbrModelType type, IconData icon, String label) {
    final isSelected = _currentModel == type;
    return Expanded(
      child: GestureDetector(
        onTap: () {
          if (!_isInitialized) return;
          setState(() => _currentModel = type);
          _renderer.setModelType(type);
          _renderer.renderFrame('default_mat');
        },
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? Colors.white : Colors.white.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              Icon(
                icon,
                color: isSelected ? Colors.black : Colors.white,
                size: 20,
              ),
              const SizedBox(height: 4),
              Text(
                label,
                style: TextStyle(
                  color: isSelected ? Colors.black : Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    // _renderer.dispose(); // Keep renderer alive if we want to reuse it, or dispose if this is the only screen
    super.dispose();
  }
}
