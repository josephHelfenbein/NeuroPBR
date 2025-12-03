import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:neuro_pbr/providers/preferences_provider.dart';
import 'dart:math' as math;
import 'dart:io';
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'scan_screen.dart';
import 'renderer_screen.dart';
import '../theme/theme_provider.dart';

// --- Data Models ---
enum ItemType { folder, file }

class FileSystemItem {
  final String id;
  final String name;
  final ItemType type;
  final Color? mainColor;
  final IconData? iconShape;
  final List<String> previewImages;
  final List<FileSystemItem> items;
  final String? tag;
  final Color? tagColor;
  final String? path; // Path to the material folder (asset or filesystem)
  final bool isAsset; // True if it's a bundled asset

  const FileSystemItem({
    required this.id,
    required this.name,
    required this.type,
    this.mainColor,
    this.iconShape,
    this.previewImages = const [],
    this.items = const [],
    this.tag,
    this.tagColor,
    this.path,
    this.isAsset = false,
  });

  FileSystemItem copyWith({
    String? id,
    String? name,
    ItemType? type,
    Color? mainColor,
    IconData? iconShape,
    List<String>? previewImages,
    List<FileSystemItem>? items,
    String? tag,
    Color? tagColor,
    String? path,
    bool? isAsset,
  }) {
    return FileSystemItem(
      id: id ?? this.id,
      name: name ?? this.name,
      type: type ?? this.type,
      mainColor: mainColor ?? this.mainColor,
      iconShape: iconShape ?? this.iconShape,
      previewImages: previewImages ?? this.previewImages,
      items: items ?? this.items,
      tag: tag ?? this.tag,
      tagColor: tagColor ?? this.tagColor,
      path: path ?? this.path,
      isAsset: isAsset ?? this.isAsset,
    );
  }
}

// --- Mock Data ---
// (Replaced by dynamic loading)

// --- Utility Functions ---
// (None currently)

List<FileSystemItem> getCurrentContent(List<String> pathIds, List<FileSystemItem> rootItems) {
  // For now, since we are flattening everything, we might not need deep folder navigation 
  // unless we re-implement folder structure. 
  // But let's keep it simple and just return the root items if path is empty.
  if (pathIds.isEmpty) return rootItems;
  
  // If we had nested folders, we would traverse here.
  return [];
}

// --- Tags Screen ---
class TagsScreen extends StatefulWidget {
  const TagsScreen({super.key});

  @override
  State<TagsScreen> createState() => _TagsScreenState();
}

class _TagsScreenState extends State<TagsScreen> {
  List<String> path = [];
  
  // Flattened list of all materials
  List<FileSystemItem> _allMaterials = [];
  String _selectedTag = 'All';

  bool get isRoot => path.isEmpty;
  List<FileSystemItem> get currentContent => getCurrentContent(path, _allMaterials);

  @override
  void initState() {
    super.initState();
    _loadMaterials();
  }
  
  Future<void> _loadMaterials() async {
    List<FileSystemItem> materials = [];

    try {
      final directory = await getApplicationDocumentsDirectory();
      final materialsDir = Directory('${directory.path}/Materials');

      if (!await materialsDir.exists()) {
        await materialsDir.create(recursive: true);
        debugPrint('Created Materials directory at ${materialsDir.path}');
      }

      // Check and create Example Material if needed
      final exampleDir = Directory('${materialsDir.path}/Example Material');
      if (!await exampleDir.exists()) {
        await exampleDir.create();
        
        // Copy assets
        final textures = ['albedo.png', 'metallic.png', 'normal.png', 'roughness.png'];
        for (final tex in textures) {
          try {
            final data = await rootBundle.load('assets/default_tex/$tex');
            final file = File('${exampleDir.path}/$tex');
            await file.writeAsBytes(data.buffer.asUint8List());
          } catch (e) {
            debugPrint('Error copying asset $tex: $e');
          }
        }

        // Create info.json
        final infoFile = File('${exampleDir.path}/info.json');
        await infoFile.writeAsString(jsonEncode({
          'name': 'Example Material',
          'tag': 'Stone'
        }));
        
        debugPrint('Created Example Material at ${exampleDir.path}');
      }

      // Load all materials from filesystem
      final entities = materialsDir.listSync();
      for (var entity in entities) {
        if (entity is Directory) {
          // Check for info.json
          final infoFile = File('${entity.path}/info.json');
          if (await infoFile.exists()) {
            try {
              final infoContent = await infoFile.readAsString();
              final info = jsonDecode(infoContent);
              
              final name = info['name'] as String? ?? 'Unknown Material';
              final tag = info['tag'] as String? ?? 'Misc';
              
              materials.add(FileSystemItem(
                id: entity.path, // Use path as ID
                name: name,
                type: ItemType.file,
                tag: tag,
                tagColor: _getColorForTag(tag),
                path: entity.path,
                isAsset: false,
              ));
            } catch (e) {
              debugPrint('Error parsing info.json for ${entity.path}: $e');
            }
          }
        }
      }
    } catch (e) {
      debugPrint('Error loading user materials: $e');
    }

    if (mounted) {
      setState(() {
        _allMaterials = materials;
      });
    }
  }

  Color _getColorForTag(String tag) {
    switch (tag.toLowerCase()) {
      case 'wood': return const Color(0xFFA67C52);
      case 'metal': return const Color(0xFF94A3B8);
      case 'fabric': return const Color(0xFFE63946);
      case 'stone': return const Color(0xFF6C757D);
      case 'plastic': return const Color(0xFF457B9D);
      default: return const Color(0xFFF4A261); // Misc color
    }
  }

  @override
  void dispose() {
    super.dispose();
  }

  void handleOpen(FileSystemItem item) {
    if (item.type == ItemType.folder) {
      setState(() {
        path.add(item.id);
      });
    } else {
      debugPrint('Opening file: ${item.name} at ${item.path}');
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => RendererScreen(
          materialPath: item.path,
          isAsset: item.isAsset,
        )),
      );
    }
  }

  void handleBack() {
    if (path.isNotEmpty) {
      setState(() {
        path.removeLast();
      });
    }
  }

  Future<void> _renameMaterial(FileSystemItem item, String newName) async {
    if (item.path == null) return;
    try {
      final infoFile = File('${item.path}/info.json');
      if (await infoFile.exists()) {
        final content = await infoFile.readAsString();
        final Map<String, dynamic> info = jsonDecode(content);
        info['name'] = newName;
        await infoFile.writeAsString(jsonEncode(info));
        
        if (mounted) {
          setState(() {
            final index = _allMaterials.indexWhere((m) => m.id == item.id);
            if (index != -1) {
              _allMaterials[index] = item.copyWith(name: newName);
            }
          });
        }
      }
    } catch (e) {
      debugPrint('Error renaming material: $e');
    }
  }

  Future<void> _deleteMaterial(FileSystemItem item) async {
    if (item.path == null) return;
    
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Material'),
        content: Text('Are you sure you want to delete "${item.name}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      try {
        final dir = Directory(item.path!);
        if (await dir.exists()) {
          await dir.delete(recursive: true);
          
          if (mounted) {
            setState(() {
              _allMaterials.removeWhere((m) => m.id == item.id);
            });
          }
        }
      } catch (e) {
        debugPrint('Error deleting material: $e');
      }
    }
  }

  Future<void> _showRenameDialog(BuildContext context, FileSystemItem item) async {
    final TextEditingController controller = TextEditingController(text: item.name);
    return showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Rename Material'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(hintText: "Enter new name"),
            autofocus: true,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                if (controller.text.isNotEmpty && controller.text != item.name) {
                  _renameMaterial(item, controller.text);
                }
              },
              child: const Text('Rename'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _changeTag(FileSystemItem item, String newTag) async {
    if (item.path == null) return;
    try {
      final infoFile = File('${item.path}/info.json');
      if (await infoFile.exists()) {
        final content = await infoFile.readAsString();
        final Map<String, dynamic> info = jsonDecode(content);
        info['tag'] = newTag;
        await infoFile.writeAsString(jsonEncode(info));
        
        if (mounted) {
          setState(() {
            final index = _allMaterials.indexWhere((m) => m.id == item.id);
            if (index != -1) {
              _allMaterials[index] = item.copyWith(
                tag: newTag,
                tagColor: _getColorForTag(newTag),
              );
            }
          });
        }
      }
    } catch (e) {
      debugPrint('Error changing tag: $e');
    }
  }

  Future<void> _showChangeTagDialog(BuildContext context, FileSystemItem item) async {
    final predefinedTags = ['Wood', 'Metal', 'Fabric', 'Stone', 'Plastic', 'Misc'];
    
    return showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Change Tag'),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: predefinedTags.length,
              itemBuilder: (context, index) {
                final tag = predefinedTags[index];
                final isSelected = item.tag == tag;
                return ListTile(
                  title: Text(tag),
                  leading: CircleAvatar(
                    backgroundColor: _getColorForTag(tag),
                    radius: 8,
                  ),
                  trailing: isSelected ? const Icon(Icons.check, color: Colors.blue) : null,
                  onTap: () {
                    Navigator.pop(context);
                    if (tag != item.tag) {
                      _changeTag(item, tag);
                    }
                  },
                );
              },
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final colors = themeProvider.colors;

    return Scaffold(
      backgroundColor: colors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(colors),
            Expanded(child: _buildBody(colors)),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(dynamic colors){
    return Container(
      padding: const EdgeInsets.all(20),
      color: Colors.transparent,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const SizedBox(width: 44),
          GestureDetector(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const ScanScreenNew()),
              );
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.4),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: colors.border.withOpacity(0.5)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.camera_alt, size: 16, color: colors.textPrimary),
                  const SizedBox(width: 8),
                  Text(
                    'CAMERA',
                    style: TextStyle(
                      color: colors.textPrimary,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody(dynamic colors) {
    if (!isRoot) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
            child: Row(
              children: [
                _IconButton(
                  icon: Icons.chevron_left,
                  onTap: handleBack,
                  backgroundColor: colors.surface,
                  hasBorder: true,
                  borderColor: colors.border,
                ),
                const SizedBox(width: 8),
                Text(
                  'Assets',
                  style: TextStyle(
                    color: colors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: colors.surface.withOpacity(0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: currentContent.isEmpty
                  ? Center(
                child: Text(
                  'No assets found.',
                  style: TextStyle(color: colors.textSecondary),
                ),
              )
                  : ListView.builder(
                padding: EdgeInsets.zero,
                itemCount: currentContent.length,
                itemBuilder: (context, index) {
                  return FileSystemItemWidget(
                    item: currentContent[index],
                    onOpen: handleOpen,
                  );
                },
              ),
            ),
          ),
        ],
      );
    }

    // Get unique tags from _allMaterials
    final tags = ['All', ..._allMaterials.map((e) => e.tag).where((t) => t != null).toSet().cast<String>().toList()];
    
    // Filter materials
    final filteredMaterials = _selectedTag == 'All' 
        ? _allMaterials 
        : _allMaterials.where((m) => m.tag == _selectedTag).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Filter Bar
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Row(
            children: tags.map((tag) {
              final isSelected = _selectedTag == tag;
              return Padding(
                padding: const EdgeInsets.only(right: 12),
                child: GestureDetector(
                  onTap: () => setState(() => _selectedTag = tag),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: isSelected ? Colors.white : Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      tag,
                      style: TextStyle(
                        color: isSelected ? Colors.black : Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ),

        // List View
        Expanded(
          child: RefreshIndicator(
            onRefresh: _loadMaterials,
            child: ListView.builder(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.symmetric(horizontal: 24),
              itemCount: filteredMaterials.length,
              itemBuilder: (context, index) {
                final item = filteredMaterials[index];
                return _buildMaterialListItem(item, colors);
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMaterialListItem(FileSystemItem item, dynamic colors) {
    return InkWell(
      onTap: () => handleOpen(item),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: colors.surface.withOpacity(0.5),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: colors.border.withOpacity(0.3),
          ),
        ),
        child: Row(
          children: [
            // Icon or Preview
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: Colors.grey[800],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(Icons.texture, color: Colors.white54),
            ),
            const SizedBox(width: 16),
            
            // Name and Tag
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.name,
                    style: TextStyle(
                      color: colors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  // Tag Label
                  if (item.tag != null)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: item.tagColor?.withOpacity(0.2) ?? Colors.grey.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 6,
                            height: 6,
                            decoration: BoxDecoration(
                              color: item.tagColor ?? Colors.grey,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            item.tag!,
                            style: TextStyle(
                              color: item.tagColor ?? Colors.grey,
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
            
            // Action Button
            PopupMenuButton<String>(
              icon: Icon(Icons.more_vert, color: colors.textSecondary),
              onSelected: (value) {
                if (value == 'rename') {
                  _showRenameDialog(context, item);
                } else if (value == 'change_tag') {
                  _showChangeTagDialog(context, item);
                } else if (value == 'delete') {
                  _deleteMaterial(item);
                }
              },
              itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
                const PopupMenuItem<String>(
                  value: 'rename',
                  child: Text('Rename'),
                ),
                const PopupMenuItem<String>(
                  value: 'change_tag',
                  child: Text('Change Tag'),
                ),
                const PopupMenuItem<String>(
                  value: 'delete',
                  child: Text('Delete', style: TextStyle(color: Colors.red)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// --- Icon Button Widget ---
class _IconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final Color backgroundColor;
  final bool hasBorder;
  final bool hasShadow;
  final Color? borderColor;

  const _IconButton({
    required this.icon,
    required this.onTap,
    required this.backgroundColor,
    this.hasBorder = false,
    this.hasShadow = false,
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;

    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: backgroundColor,
          shape: BoxShape.circle,
          border: hasBorder ? Border.all(color: borderColor ?? colors.border) : null,
          boxShadow: hasShadow
              ? [
            BoxShadow(
              color: backgroundColor.withOpacity(0.5),
              blurRadius: 15,
            ),
          ]
              : null,
        ),
        child: Icon(icon, size: 20, color: Colors.white),
      ),
    );
  }
}

// --- File System Item Widget ---
class FileSystemItemWidget extends StatelessWidget {
  final FileSystemItem item;
  final Function(FileSystemItem) onOpen;

  const FileSystemItemWidget({
    super.key,
    required this.item,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;
    final bool isFolder = item.type == ItemType.folder;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => onOpen(item),
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Icon(
                isFolder ? Icons.folder : Icons.insert_drive_file_outlined,
                size: 24,
                color: isFolder ? const Color(0xFFFBBF24) : colors.textSecondary,
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  item.name,
                  style: TextStyle(
                    color: colors.textPrimary,
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              if (isFolder)
                Text(
                  '${item.items.length} items',
                  style: TextStyle(color: colors.textSecondary, fontSize: 14),
                ),
            ],
          ),
        ),
      ),
    );
  }
}