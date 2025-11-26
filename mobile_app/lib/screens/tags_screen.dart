import 'package:flutter/material.dart';
import 'package:neuro_pbr/providers/preferences_provider.dart';
import 'dart:math' as math;
import 'package:provider/provider.dart';
import 'scan_screen_new.dart';
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
    );
  }
}

// --- Mock Data ---
final List<FileSystemItem> initialFilesystem = [
  const FileSystemItem(
    id: '1',
    name: 'Misc',
    type: ItemType.folder,
    mainColor: Color(0xFFF4A261),
    iconShape: Icons.circle,
    previewImages: [
      'https://placehold.co/400x500/E76F51/FFFFFF/png?text=Misc+1',
      'https://placehold.co/400x500/2A9D8F/FFFFFF/png?text=Misc+2'
    ],
    items: [
      FileSystemItem(id: '1-1', name: 'Random Asset 01', type: ItemType.file),
      FileSystemItem(id: '1-2', name: 'Scratch Pad', type: ItemType.file),
      FileSystemItem(id: '1-3', name: 'Reference', type: ItemType.file),
    ],
  ),
  const FileSystemItem(
    id: '2',
    name: 'Wood',
    type: ItemType.folder,
    mainColor: Color(0xFFA67C52),
    iconShape: Icons.square,
    previewImages: [
      'https://placehold.co/400x500/8B4513/FFFFFF/png?text=Oak+Texture',
      'https://placehold.co/400x500/DEB887/FFFFFF/png?text=Pine+Grain'
    ],
    items: [
      FileSystemItem(id: '2-1', name: 'Oak_Albedo.png', type: ItemType.file),
    ],
  ),
  const FileSystemItem(
    id: '3',
    name: 'Metal',
    type: ItemType.folder,
    mainColor: Color(0xFF94A3B8),
    iconShape: Icons.hexagon_outlined,
    previewImages: [
      'https://placehold.co/400x500/475569/FFFFFF/png?text=Brushed+Steel',
      'https://placehold.co/400x500/CBD5E1/FFFFFF/png?text=Gold+Leaf'
    ],
    items: [
      FileSystemItem(id: '3-1', name: 'Rust Map', type: ItemType.file),
      FileSystemItem(id: '3-2', name: 'Scratches', type: ItemType.file),
      FileSystemItem(id: '3-3', name: 'Iron.obj', type: ItemType.file),
      FileSystemItem(id: '3-4', name: 'Copper.mat', type: ItemType.file),
    ],
  ),
  const FileSystemItem(
    id: '4',
    name: 'Fabric',
    type: ItemType.folder,
    mainColor: Color(0xFFE63946),
    iconShape: Icons.change_history,
    items: [
      FileSystemItem(id: '4-1', name: 'Silk Pattern', type: ItemType.file),
      FileSystemItem(id: '4-2', name: 'Wool.norm', type: ItemType.file),
      FileSystemItem(id: '4-3', name: 'Denim.diff', type: ItemType.file),
    ],
  ),
  const FileSystemItem(
    id: '5',
    name: 'Stone',
    type: ItemType.folder,
    mainColor: Color(0xFF6C757D),
    iconShape: Icons.album,
    items: [
      FileSystemItem(id: '5-1', name: 'Marble Tile', type: ItemType.file),
      FileSystemItem(id: '5-2', name: 'Granite.rough', type: ItemType.file),
      FileSystemItem(id: '5-3', name: 'Pavement.disp', type: ItemType.file),
      FileSystemItem(id: '5-4', name: 'Slate.norm', type: ItemType.file),
    ],
  ),
  const FileSystemItem(
    id: '6',
    name: 'Plastic',
    type: ItemType.folder,
    mainColor: Color(0xFF457B9D),
    iconShape: Icons.square_outlined,
    items: [
      FileSystemItem(id: '6-1', name: 'Shiny Red', type: ItemType.file),
      FileSystemItem(id: '6-2', name: 'Matte Black', type: ItemType.file),
      FileSystemItem(id: '6-3', name: 'Clearcoat', type: ItemType.file),
      FileSystemItem(id: '6-4', name: 'PVC Pipe', type: ItemType.file),
    ],
  ),
];

// --- Utility Functions ---
// (None currently)

List<FileSystemItem> getCurrentContent(List<String> pathIds) {
  List<FileSystemItem> current = initialFilesystem;
  for (final id in pathIds) {
    final folder = current.firstWhere(
          (item) => item.id == id && item.type == ItemType.folder,
      orElse: () =>
          FileSystemItem(id: 'null', name: 'null', type: ItemType.folder),
    );
    if (folder.id == 'null') return [];
    current = folder.items;
  }
  return current;
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
  List<FileSystemItem> get currentContent => getCurrentContent(path);

  @override
  void initState() {
    super.initState();
    _flattenMaterials();
  }
  
  void _flattenMaterials() {
    _allMaterials = [];
    for (var folder in initialFilesystem) {
      if (folder.type == ItemType.folder) {
        for (var item in folder.items) {
          _allMaterials.add(item.copyWith(
            tag: folder.name,
            tagColor: folder.mainColor,
          ));
        }
      }
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
      debugPrint('Opening file: ${item.name}');
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const RendererScreen()),
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

    // Get unique tags from initialFilesystem
    final tags = ['All', ...initialFilesystem.where((e) => e.type == ItemType.folder).map((e) => e.name)];
    
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
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            itemCount: filteredMaterials.length,
            itemBuilder: (context, index) {
              final item = filteredMaterials[index];
              return _buildMaterialListItem(item, colors);
            },
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
            IconButton(
              icon: Icon(Icons.more_vert, color: colors.textSecondary),
              onPressed: () {},
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