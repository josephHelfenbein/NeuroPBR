import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'package:google_fonts/google_fonts.dart';
import 'package:neuro_pbr/screens/main_tab_screen.dart';
import 'package:provider/provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import '../theme/theme_provider.dart';
import '../theme/app_theme.dart';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import '../services/pbr_service.dart';
import 'dart:convert';

class CapturedImagesScreen extends StatefulWidget {
  final List<String> imagePaths;

  const CapturedImagesScreen({super.key, required this.imagePaths});

  @override
  State<CapturedImagesScreen> createState() => _CapturedImagesScreenState();
}

class _CapturedImagesScreenState extends State<CapturedImagesScreen> {
  // Logic State
  late List<String> _images;
  Set<int> _selectedIndices = {};
  Map<int, String> _imageTags = {};
  String _activeFilter = 'ALL';
  final PBRService _pbrService = PBRService();
  bool _isLoading = false; // To show loading spinner

  // Drag Select State
  final ScrollController _scrollController = ScrollController();
  final GlobalKey _gridKey = GlobalKey();
  bool _isDragging = false;
  int? _startDragIndex;
  Set<int> _initialSelectedIndices = {};

  // Auto-scroll constants
  static const double _scrollThreshold = 50.0;
  static const double _scrollSpeed = 15.0;

  @override
  void initState() {
    super.initState();
    _images = List.from(widget.imagePaths);
    for (int i = 0; i < _images.length; i++) {
      _imageTags[i] = 'MISC';
    }
  }

  // --- Logic Getters ---
  bool get _canProcess => _selectedIndices.length == 3 && _validateSelectionTags();

  bool _validateSelectionTags() {
    if (_selectedIndices.isEmpty) return false;
    if (_selectedIndices.first >= _images.length) return false;
    String firstTag = _imageTags[_selectedIndices.first] ?? 'MISC';
    return _selectedIndices.every((index) => _imageTags[index] == firstTag);
  }

  List<String> get _uniqueTags => ['ALL', ..._imageTags.values.toSet().toList()];

  List<int> get _filteredIndices {
    List<int> indices = List.generate(_images.length, (i) => i);
    if (_activeFilter == 'ALL') return indices;
    return indices.where((i) => _imageTags[i] == _activeFilter).toList();
  }

  // NEW: Auto-scrolling logic during drag
  void _scrollGridOnDrag(Offset localPosition) {
    if (!_isDragging) return;

    final RenderBox? box = _gridKey.currentContext?.findRenderObject() as RenderBox?;
    if (box == null) return;

    final size = box.size;
    double offset = 0.0;

    // Check if near top edge
    if (localPosition.dy < _scrollThreshold) {
      offset = -(_scrollThreshold - localPosition.dy) / _scrollThreshold * _scrollSpeed;
    }
    // Check if near bottom edge
    else if (localPosition.dy > size.height - _scrollThreshold) {
      offset = (localPosition.dy - (size.height - _scrollThreshold)) / _scrollThreshold * _scrollSpeed;
    }

    if (offset != 0.0) {
      final newOffset = (_scrollController.offset + offset).clamp(
        _scrollController.position.minScrollExtent,
        _scrollController.position.maxScrollExtent,
      );

      _scrollController.jumpTo(newOffset);

      // Update the selection immediately after scrolling
      int? index = _hitTest(localPosition, _filteredIndices);
      if (index != null) {
        _updateDragSelection(index);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // 2. Access ThemeProvider colors in build
    final themeProvider = Provider.of<ThemeProvider>(context);
    final AppColors colors = themeProvider.colors; // Get AppColors object
    final bool isDark = themeProvider.isDarkMode;

    // Local constants for cleaner code, using theme colors
    final Color accent = colors.accent;
    final Color surface = colors.surface;
    final Color background = colors.background;
    final Color primaryText = colors.textPrimary;
    final Color secondaryText = colors.textSecondary;

    // Determine the color for the background elements based on the theme
    final Color iconColor = isDark ? Colors.white : colors.textPrimary;
    final Color backIconColor = isDark ? Colors.white : Colors.black;
    final Color backButtonSurface = isDark ? surface : background;

    return PopScope(
      canPop: false,
      onPopInvoked: (didPop) {
        if (didPop) return;
        if (!_isLoading) Navigator.pop(context, _images); // Prevent back while loading
      },
      child: Scaffold(
        backgroundColor: background,
        body: Stack( // <--- CHANGED TO STACK
          children: [
            SafeArea(
              child: Column(
                children: [
                  _buildTopBar(surface, backButtonSurface, backIconColor, primaryText),
                  _buildFilterRow(primaryText, secondaryText, surface, isDark),
                  _buildStatusInfo(primaryText, secondaryText, accent),
                  Expanded(child: _buildGridWithDrag(secondaryText)),
                  _buildBottomControls(surface, accent, iconColor, secondaryText),
                ],
              ),
            ),

            // --- NEW LOADING OVERLAY ---
            if (_isLoading)
              Container(
                color: Colors.black.withOpacity(0.7),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: accent),
                      const SizedBox(height: 20),
                      Text(
                        "GENERATING PBR MAPS...",
                        style: GoogleFonts.robotoMono(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5
                        ),
                      ),
                      Text(
                        "This may take a moment",
                        style: GoogleFonts.robotoMono(
                          color: Colors.white70,
                          fontSize: 10,
                        ),
                      )
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // --- Widget Builders (Updated to pass colors) ---

  // _buildTopBar, _buildFilterRow, _buildStatusInfo, _buildBottomControls, _buildImageCard, _showTagModal...
  // (These methods are lengthy, keeping them abbreviated but noting the color arguments they use)

  Widget _buildTopBar(Color surface, Color backButtonSurface, Color iconColor, Color primaryText) {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    final accent = themeProvider.colors.accent;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context, _images),
            child: Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: backButtonSurface,
                shape: BoxShape.circle,
                border: Border.all(color: primaryText.withOpacity(0.1)),
              ),
              child: Icon(Icons.arrow_back, size: 20, color: iconColor),
            ),
          ),
          Text(
            'CAPTURED STACKS',
            style: GoogleFonts.robotoMono(
              color: primaryText,
              fontSize: 14,
              fontWeight: FontWeight.bold,
              letterSpacing: 2.0,
            ),
          ),
          GestureDetector(
            onTap: () {
              if (_selectedIndices.isNotEmpty) {
                HapticFeedback.mediumImpact();
                setState(() => _selectedIndices.clear());
              }
            },
            child: Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: _selectedIndices.isNotEmpty ? accent.withOpacity(0.2) : surface,
                shape: BoxShape.circle,
                border: Border.all(
                    color: _selectedIndices.isNotEmpty ? accent : primaryText.withOpacity(0.1)
                ),
              ),
              child: Icon(
                  Icons.layers_clear,
                  size: 20,
                  color: _selectedIndices.isNotEmpty ? accent : Colors.grey
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterRow(Color primaryText, Color secondaryText, Color surface, bool isDark) {
    return Container(
      height: 40,
      margin: const EdgeInsets.only(bottom: 12),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: _uniqueTags.length,
        itemBuilder: (context, index) {
          final tag = _uniqueTags[index];
          final isSelected = _activeFilter == tag;

          // Use primaryText for dark mode secondary text, or secondaryText for light mode
          final Color inactiveTextColor = isDark ? Colors.grey[600]! : secondaryText;
          final Color activeFilterBackground = isDark ? Colors.white.withOpacity(0.1) : surface;

          return GestureDetector(
            onTap: () {
              HapticFeedback.lightImpact();
              setState(() => _activeFilter = tag);
            },
            child: Container(
              margin: const EdgeInsets.only(right: 8),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected ? activeFilterBackground : Colors.transparent,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                    color: isSelected ? primaryText : primaryText.withOpacity(0.1)
                ),
              ),
              child: Text(
                tag,
                style: GoogleFonts.robotoMono(
                  color: isSelected ? primaryText : inactiveTextColor,
                  fontSize: 11,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildStatusInfo(Color primaryText, Color secondaryText, Color accent) {
    String statusText;
    Color statusColor;

    if (_selectedIndices.length == 3) {
      if (_validateSelectionTags()) {
        statusText = "READY TO PROCESS";
        statusColor = accent;
      } else {
        statusText = "TAG MISMATCH";
        statusColor = Colors.red;
      }
    } else {
      statusText = "${_selectedIndices.length} SELECTED";
      statusColor = primaryText;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            '${_images.length} FILES',
            style: GoogleFonts.robotoMono(
              color: primaryText,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(width: 8),
          Text('•', style: TextStyle(color: secondaryText, fontSize: 12)),
          const SizedBox(width: 8),
          Text(
            statusText,
            style: GoogleFonts.robotoMono(
              color: statusColor,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }


  // UPDATED: Now uses Listener for auto-scroll and drag detection
  Widget _buildGridWithDrag(Color secondaryText) {
    final indices = _filteredIndices;

    if (indices.isEmpty) {
      return Center(
        child: Text(
          'NO DATA',
          style: GoogleFonts.robotoMono(color: secondaryText),
        ),
      );
    }

    return Listener(
      onPointerMove: (details) {
        if (_isDragging) {
          // 1. Auto-Scroll on drag move
          _scrollGridOnDrag(details.localPosition);
        }
      },
      child: GestureDetector(
        onLongPressStart: (details) {
          HapticFeedback.selectionClick();
          setState(() {
            _isDragging = true;
            _initialSelectedIndices = Set.from(_selectedIndices);
          });
          int? index = _hitTest(details.localPosition, indices);
          if (index != null) {
            _startDragIndex = index;
            _updateDragSelection(index);
          }
        },
        onLongPressMoveUpdate: (details) {
          if (!_isDragging) return; // Only process if drag was started
          int? index = _hitTest(details.localPosition, indices);
          if (index != null) {
            _updateDragSelection(index);
          }
        },
        onLongPressEnd: (details) {
          setState(() {
            _isDragging = false;
            _startDragIndex = null;
          });
        },
        child: GridView.builder(
          key: _gridKey,
          controller: _scrollController,
          padding: const EdgeInsets.symmetric(horizontal: 20),
          physics: const AlwaysScrollableScrollPhysics(),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            childAspectRatio: 0.8,
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
          ),
          itemCount: indices.length,
          itemBuilder: (context, i) {
            return _buildImageCard(indices[i]);
          },
        ),
      ),
    );
  }

  // --- Hit Test and Drag Logic ---

  int? _hitTest(Offset localPosition, List<int> visibleIndices) {
    final RenderBox? box = _gridKey.currentContext?.findRenderObject() as RenderBox?;
    if (box == null) return null;

    final double gridWidth = box.size.width;
    const double crossAxisSpacing = 12.0;
    const double mainAxisSpacing = 12.0;
    const double horizontalPadding = 20.0;

    final double contentWidth = gridWidth - 40;
    final double itemWidth = (contentWidth - crossAxisSpacing) / 2;
    final double itemHeight = itemWidth / 0.8;

    final double dy = localPosition.dy + _scrollController.offset;
    final double dx = localPosition.dx;

    if (dx < 20 || dx > gridWidth - 20) return null;

    int col = ((dx - 20) / (itemWidth + crossAxisSpacing)).floor();
    int row = (dy / (itemHeight + mainAxisSpacing)).floor();

    if (col < 0) col = 0;
    if (col > 1) col = 1;

    int gridIndex = (row * 2) + col;

    if (gridIndex >= 0 && gridIndex < visibleIndices.length) {
      return visibleIndices[gridIndex];
    }
    return null;
  }

  void _updateDragSelection(int currentIndex) {
    if (_startDragIndex == null) return;

    final start = _startDragIndex! < currentIndex ? _startDragIndex! : currentIndex;
    final end = _startDragIndex! > currentIndex ? _startDragIndex! : currentIndex;

    Set<int> newSelection = Set.from(_initialSelectedIndices);

    final visible = _filteredIndices;

    int visibleStartIndex = visible.indexOf(start);
    int visibleEndIndex = visible.indexOf(end);

    if (visibleStartIndex == -1 || visibleEndIndex == -1) return;

    if (visibleStartIndex > visibleEndIndex) {
      final temp = visibleStartIndex;
      visibleStartIndex = visibleEndIndex;
      visibleEndIndex = temp;
    }

    for (int i = visibleStartIndex; i <= visibleEndIndex; i++) {
      newSelection.add(visible[i]);
    }

    setState(() {
      _selectedIndices = newSelection;
    });
  }

  Widget _buildImageCard(int index) {
    if (index >= _images.length) return const SizedBox();

    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    final AppColors colors = themeProvider.colors;
    final Color accent = colors.accent;
    final Color primaryText = colors.textPrimary;

    final isSelected = _selectedIndices.contains(index);
    final tag = _imageTags[index] ?? 'MISC';

    return GestureDetector(
      onTap: () => _handleSelection(index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? accent : primaryText.withOpacity(0.1),
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Stack(
          fit: StackFit.expand,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: Image.file(
                File(_images[index]),
                fit: BoxFit.cover,
                errorBuilder: (c, o, s) => Container(color: colors.surfaceVariant),
              ),
            ),
            if (isSelected)
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(14),
                  color: accent.withOpacity(0.1),
                ),
              ),
            Positioned(
              top: 8,
              right: 8,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 20,
                height: 20,
                decoration: BoxDecoration(
                  color: isSelected ? accent : Colors.black.withOpacity(0.5),
                  shape: BoxShape.circle,
                  border: Border.all(color: isSelected ? accent : Colors.white.withOpacity(0.5)),
                ),
                child: isSelected
                    ? const Center(child: Icon(Icons.check, size: 12, color: Colors.white))
                    : null,
              ),
            ),
            Positioned(
              bottom: 8,
              left: 8,
              child: GestureDetector(
                onTap: () => _showTagSelector(index),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.white.withOpacity(0.1)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        tag,
                        style: GoogleFonts.robotoMono(
                          color: Colors.white,
                          fontSize: 9,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 4),
                      Icon(Icons.edit, size: 8, color: Colors.grey[400]),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomControls(Color surface, Color accent, Color iconColor, Color secondaryText) {
    final Color inactiveColor = secondaryText.withOpacity(0.3);

    return Container(
      padding: const EdgeInsets.all(20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // Bulk Tag Selector
          GestureDetector(
            onTap: _selectedIndices.isNotEmpty ? () => _showBulkTagSelector() : null,
            child: Opacity(
              opacity: _selectedIndices.isNotEmpty ? 1.0 : 0.3,
              child: Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: surface,
                  shape: BoxShape.circle,
                  border: Border.all(color: iconColor.withOpacity(0.1), width: 2),
                ),
                child: Icon(Icons.label_outline, color: iconColor, size: 24),
              ),
            ),
          ),

          // Process Button
          GestureDetector(
            onTap: _canProcess ? _showNameDialog : null,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: _canProcess ? accent : Colors.transparent,
                shape: BoxShape.circle,
                border: Border.all(
                    color: _canProcess ? accent : iconColor.withOpacity(0.2),
                    width: 4
                ),
                boxShadow: _canProcess ? [
                  BoxShadow(color: accent.withOpacity(0.5), blurRadius: 20, spreadRadius: 3),
                ] : [],
              ),
              child: Icon(
                  Icons.auto_awesome,
                  color: _canProcess ? Colors.white : inactiveColor,
                  size: 32
              ),
            ),
          ),

          // Delete Button
          GestureDetector(
            onTap: _selectedIndices.isNotEmpty ? _deleteSelected : null,
            child: Opacity(
              opacity: _selectedIndices.isNotEmpty ? 1.0 : 0.3,
              child: Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: surface,
                  shape: BoxShape.circle,
                  border: Border.all(color: iconColor.withOpacity(0.1), width: 2),
                ),
                child: Icon(Icons.delete_outline, color: iconColor, size: 24),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // --- Other Logic Methods ---

  void _deleteSelected() {
    HapticFeedback.heavyImpact();

    List<String> keptImages = [];
    Map<int, String> newTags = {};
    int newIndexCounter = 0;

    for (int i = 0; i < _images.length; i++) {
      if (!_selectedIndices.contains(i)) {
        keptImages.add(_images[i]);
        newTags[newIndexCounter] = _imageTags[i] ?? 'MISC';
        newIndexCounter++;
      }
    }

    setState(() {
      _images = keptImages;
      _imageTags = newTags;
      _selectedIndices.clear();
    });
  }

  void _handleSelection(int index) {
    HapticFeedback.lightImpact();
    setState(() {
      if (_selectedIndices.contains(index)) {
        _selectedIndices.remove(index);
      } else {
        _selectedIndices.add(index);
        if (_selectedIndices.length == 3 && _canProcess) {
          HapticFeedback.mediumImpact();
        }
      }
    });
  }

  Future<void> _processImages(String customName) async {
    HapticFeedback.heavyImpact();

    // Safety check
    if (_selectedIndices.length != 3) return;

    setState(() {
      _isLoading = true;
    });
    
    // Keep screen on during generation
    await WakelockPlus.enable();

    try {
      // 1. Convert selected indices to File objects
      final List<int> sortedIndices = _selectedIndices.toList()..sort();

      final File v1 = File(_images[sortedIndices[0]]);
      final File v2 = File(_images[sortedIndices[1]]);
      final File v3 = File(_images[sortedIndices[2]]);

      // 2. Prepare output directory BEFORE calling AI
      final Directory appDocDir = await getApplicationDocumentsDirectory();
      final Directory materialsBaseDir = Directory('${appDocDir.path}/Materials');
      if (!await materialsBaseDir.exists()) {
        await materialsBaseDir.create(recursive: true);
      }
      
      String safeName = customName.replaceAll(RegExp(r'[<>:"/\\|?*]'), '_');
      Directory newMaterialDir = Directory('${materialsBaseDir.path}/$safeName');
      
      if (await newMaterialDir.exists()) {
        await newMaterialDir.delete(recursive: true);
      }
      await newMaterialDir.create(recursive: true);

      // 3. Call AI - it writes directly to disk (no memory round-trip!)
      await _pbrService.generatePBRMaps(
        view1: v1,
        view2: v2,
        view3: v3,
        outputDir: newMaterialDir.path,
      );

      // 4. Create info.json (textures are already saved by native code)
      String materialTag = "Misc";
      if (_selectedIndices.isNotEmpty) {
        materialTag = _imageTags[_selectedIndices.first] ?? "Misc";
      }
      
      final File infoFile = File('${newMaterialDir.path}/info.json');
      final Map<String, String> infoData = {
        'name': customName,
        'tag': materialTag,
        'date_created': DateTime.now().toIso8601String(),
      };
      await infoFile.writeAsString(jsonEncode(infoData));

      // 5. Success Feedback
      if (mounted) {
        HapticFeedback.mediumImpact();
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const MainTabScreen()),
              (Route<dynamic> route) => false,
        );
      }

    } catch (e) {
      if (mounted) {
        HapticFeedback.vibrate();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Error: $e", style: const TextStyle(color: Colors.white)),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      // Allow screen to dim again
      await WakelockPlus.disable();
      
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<bool> _doesNameExist(String rawName) async {
    final Directory appDocDir = await getApplicationDocumentsDirectory();
    final String safeName = rawName.replaceAll(RegExp(r'[<>:"/\\|?*]'), '_');
    final Directory materialDir = Directory('${appDocDir.path}/Materials/$safeName');

    return await materialDir.exists();
  }

  Future<void> _showNameDialog() async {
    HapticFeedback.selectionClick();

    final TextEditingController controller = TextEditingController(text: "");

    final String? name = await showDialog<String>(
      context: context,
      // barrierDismissible: false, // Optional: Force them to click Cancel or Create
      builder: (context) {
        final theme = Provider.of<ThemeProvider>(context);

        // Define local variables for state
        bool isChecking = false; // To show a mini spinner on the button
        String? errorMessage;    // To show "Name already exists" in red

        return StatefulBuilder(
            builder: (context, setDialogState) {
              final bool isValid = controller.text.trim().isNotEmpty;

              return AlertDialog(
                backgroundColor: theme.colors.surface,
                title: Text(
                    "Name Material",
                    style: GoogleFonts.robotoMono(
                        color: theme.colors.textPrimary,
                        fontWeight: FontWeight.bold
                    )
                ),
                content: Column(
                  mainAxisSize: MainAxisSize.min, // Keep dialog compact
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      controller: controller,
                      autofocus: true,
                      style: TextStyle(color: theme.colors.textPrimary),
                      onChanged: (value) {
                        // Rebuild to update isValid and clear any error
                        setDialogState(() {
                          errorMessage = null;
                        });
                      },
                      decoration: InputDecoration(
                        hintText: "Enter unique name...",
                        hintStyle: TextStyle(color: theme.colors.textSecondary.withOpacity(0.5)),
                        errorText: errorMessage, // This displays the red text if set
                        enabledBorder: UnderlineInputBorder(
                          borderSide: BorderSide(color: theme.colors.accent),
                        ),
                        focusedBorder: UnderlineInputBorder(
                          borderSide: BorderSide(color: theme.colors.accent, width: 2),
                        ),
                      ),
                    ),
                  ],
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context, null),
                    child: Text("Cancel", style: TextStyle(color: theme.colors.textSecondary)),
                  ),
                  TextButton(
                    // Disable button if text is empty OR if we are currently checking
                    onPressed: (isValid && !isChecking) ? () async {
                      // 1. Start Checking
                      setDialogState(() { isChecking = true; });

                      final String inputName = controller.text.trim();
                      final bool exists = await _doesNameExist(inputName);

                      // 2. Handle Result
                      if (exists) {
                        // Stop spinner, show error, DON'T close dialog
                        setDialogState(() {
                          isChecking = false;
                          errorMessage = "Material name already exists";
                        });
                        HapticFeedback.mediumImpact(); // Error vibration
                      } else {
                        // Success! Close dialog and pass the name back
                        if (context.mounted) {
                          Navigator.pop(context, inputName);
                        }
                      }
                    } : null,
                    child: isChecking
                        ? SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: theme.colors.accent))
                        : Text(
                        "CREATE",
                        style: TextStyle(
                            color: isValid ? theme.colors.accent : theme.colors.textSecondary.withOpacity(0.3),
                            fontWeight: FontWeight.bold
                        )
                    ),
                  ),
                ],
              );
            }
        );
      },
    );

    if (name != null && name.isNotEmpty) {
      _processImages(name);
    }
  }

  void _showTagSelector(int index) {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    _showTagModal(themeProvider.colors.surface, themeProvider.colors.textSecondary, (tag) {
      setState(() {
        _imageTags[index] = tag;
      });
    });
  }

  void _showBulkTagSelector() {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    _showTagModal(themeProvider.colors.surface, themeProvider.colors.textSecondary, (tag) {
      setState(() {
        for (var idx in _selectedIndices) {
          _imageTags[idx] = tag;
        }
      });
    });
  }

  void _showTagModal(Color modalColor, Color secondaryTextColor, Function(String) onSelect) {
    HapticFeedback.selectionClick();
    final tags = ['MISC', 'WOOD', 'METAL', 'FABRIC', 'STONE', 'PLASTIC'];

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: BoxDecoration(
          color: modalColor,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          border: Border.all(color: secondaryTextColor.withOpacity(0.1)),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'SELECT MATERIAL',
              style: GoogleFonts.robotoMono(
                color: secondaryTextColor,
                fontSize: 10,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.5,
              ),
            ),
            const SizedBox(height: 20),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              alignment: WrapAlignment.center,
              children: tags.map((tag) => GestureDetector(
                onTap: () {
                  HapticFeedback.lightImpact();
                  onSelect(tag);
                  Navigator.pop(context);
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.black,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white.withOpacity(0.1)),
                  ),
                  child: Text(
                    tag,
                    style: GoogleFonts.robotoMono(
                      color: Colors.white,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              )).toList(),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}