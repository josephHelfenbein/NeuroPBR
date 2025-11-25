import 'package:flutter/material.dart';
import 'package:neuro_pbr/providers/preferences_provider.dart';
import 'dart:math' as math;
import 'package:provider/provider.dart';
import 'scan_screen_new.dart';
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

  const FileSystemItem({
    required this.id,
    required this.name,
    required this.type,
    this.mainColor,
    this.iconShape,
    this.previewImages = const [],
    this.items = const [],
  });
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
Color getDarkerColor(Color color) {
  final red = math.max(0, color.red - 40);
  final green = math.max(0, color.green - 40);
  final blue = math.max(0, color.blue - 40);
  return Color.fromARGB(color.alpha, red, green, blue);
}

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

  // Local view mode state (not persisted) - null means use preference
  String? _localViewModeOverride;
  final ScrollController _cardsScrollController = ScrollController();

  // Constants for card dimensions
  static const double _cardWidth = 288.0;

  bool get isRoot => path.isEmpty;
  List<FileSystemItem> get currentContent => getCurrentContent(path);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final prefs = Provider.of<PreferencesProvider>(context, listen: false);
      final viewMode = _localViewModeOverride ?? prefs.tagsViewMode;
      // Calculate initial scroll offset to center card at index 0 (Misc)
      if (_cardsScrollController.hasClients && viewMode == 'cards') {
        final double initialOffset = 0.0;
        _cardsScrollController.jumpTo(initialOffset);
      }
    });
  }

  @override
  void dispose() {
    _cardsScrollController.dispose();
    super.dispose();
  }

  void handleOpen(FileSystemItem item) {
    if (item.type == ItemType.folder) {
      setState(() {
        path.add(item.id);
      });
    } else {
      debugPrint('Opening file: ${item.name}');
    }
  }

  void handleBack() {
    if (path.isNotEmpty) {
      setState(() {
        path.removeLast();
      });
    }
  }

  void toggleViewMode() {
    final prefs = Provider.of<PreferencesProvider>(context, listen: false);
    final currentMode = _localViewModeOverride ?? prefs.tagsViewMode;
    setState(() {
      _localViewModeOverride = currentMode == 'cards' ? 'list' : 'cards';
    });
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final prefs = Provider.of<PreferencesProvider>(context);

    final colors = themeProvider.colors;

    // Use local override if set, otherwise use preference (this allows settings changes to affect tags screen)
    final isCards = (_localViewModeOverride ?? prefs.tagsViewMode) == 'cards';

    return Scaffold(
      backgroundColor: colors.background,
      body: SafeArea(
        child: Column(
          children: [
            // Pass local toggle to header
            _buildHeader(colors, isCards),
            // Pass the boolean to body
            Expanded(child: _buildBody(colors, isCards)),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(dynamic colors, bool isCards){
    final showToggle = isRoot;

    return Container(
      padding: const EdgeInsets.all(20),
      color: Colors.transparent,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          if (showToggle)
            _IconButton(
              icon: isCards
                  ? Icons.format_list_bulleted
                  : Icons.grid_view_rounded,
              onTap: toggleViewMode,
              backgroundColor: colors.surface,
              hasBorder: true,
              borderColor: colors.border,
            )
          else
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
          _IconButton(
            icon: Icons.view_in_ar,
            onTap: () => debugPrint('View Renders'),
            backgroundColor: colors.accent,
            hasShadow: true,
            borderColor: colors.border,
          ),
        ],
      ),
    );
  }

  Widget _buildBody(dynamic colors, bool isCards) {
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

    return Column(
      children: [
        const SizedBox(height: 8),
        Text(
          isCards ? 'RECENT STACKS' : 'TAG LIBRARY',
          style: TextStyle(
            color: colors.textSecondary,
            fontSize: 11,
            fontWeight: FontWeight.bold,
            letterSpacing: 3.0,
          ),
        ),
        const SizedBox(height: 16),
        Expanded(
          child: isCards ? _buildCardsView() : _buildListView(),
        ),
      ],
    );
  }

  Widget _buildCardsView() {
    return Center(
      child: SizedBox(
        height: 400,
        child: ListView.builder(
          controller: _cardsScrollController,
          scrollDirection: Axis.horizontal,
          padding: EdgeInsets.symmetric(horizontal: (_cardWidth / 2) - 144),
          clipBehavior: Clip.none,
          cacheExtent: 3000,
          physics: const BouncingScrollPhysics(),
          itemCount: initialFilesystem.length,
          itemBuilder: (context, index) {
            return ProjectCard(
              item: initialFilesystem[index],
              onOpen: handleOpen,
              index: index,
            );
          },
        ),
      ),
    );
  }

  Widget _buildListView() {
    final colors = Provider.of<ThemeProvider>(context, listen: false).colors;

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      children: [
        ...initialFilesystem.map<Widget>(
              (FileSystemItem item) => TagListViewItem(
            item: item,
            onOpen: handleOpen,
          ),
        ),
        const SizedBox(height: 32),
        Text(
          'Material Library',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontFamily: 'Serif',
            fontSize: 24,
            color: colors.textPrimary,
          ),
        ),
        const SizedBox(height: 8),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Text(
            'Organize your PBR assets by physical properties.',
            textAlign: TextAlign.center,
            style: TextStyle(color: colors.textSecondary, fontSize: 14),
          ),
        ),
        const SizedBox(height: 80),
      ],
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

class ProjectCard extends StatelessWidget {
  final FileSystemItem item;
  final Function(FileSystemItem) onOpen;
  final int index;

  const ProjectCard({
    super.key,
    required this.item,
    required this.onOpen,
    required this.index,
  });

  @override
  Widget build(BuildContext context) {
    final mainColor = item.mainColor ?? Colors.grey;
    final darkerColor = getDarkerColor(mainColor);
    final darkestColor = getDarkerColor(darkerColor);

    return GestureDetector(
      onTap: () => onOpen(item),
      child: Container(
        width: _TagsScreenState._cardWidth,
        height: 360,
        margin: const EdgeInsets.symmetric(horizontal: 8),
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center,
          children: [
            // Back card 2 - furthest back
            Positioned(
              top: -20,
              left: 0,
              right: 0,
              child: Container(
                width: 256,
                height: 320,
                decoration: BoxDecoration(
                  color: item.previewImages.length > 1
                      ? Colors.transparent
                      : darkestColor,
                  borderRadius: BorderRadius.circular(20),
                  image: item.previewImages.length > 1
                      ? DecorationImage(
                    image: NetworkImage(item.previewImages[1]),
                    fit: BoxFit.cover,
                  )
                      : null,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.4),
                      blurRadius: 15,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: item.previewImages.length > 1
                    ? Container(
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(20),
                  ),
                )
                    : null,
              ),
            ),

            // Back card 1 - middle layer
            Positioned(
              top: -10,
              left: 0,
              right: 0,
              child: Container(
                width: 256,
                height: 320,
                decoration: BoxDecoration(
                  color: item.previewImages.isNotEmpty
                      ? Colors.transparent
                      : darkerColor,
                  borderRadius: BorderRadius.circular(20),
                  image: item.previewImages.isNotEmpty
                      ? DecorationImage(
                    image: NetworkImage(item.previewImages[0]),
                    fit: BoxFit.cover,
                  )
                      : null,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.4),
                      blurRadius: 15,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: item.previewImages.isNotEmpty
                    ? Container(
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(20),
                  ),
                )
                    : null,
              ),
            ),

            // Front card - main card with content
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: Container(
                width: 256,
                height: 320,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: mainColor,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.9),
                      blurRadius: 50,
                      offset: const Offset(0, 25),
                    ),
                  ],
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.name,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.w900,
                        height: 1.1,
                        shadows: [
                          Shadow(
                            color: Colors.black38,
                            blurRadius: 4,
                            offset: Offset(0, 2),
                          )
                        ],
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${item.items.length} renders',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text(
                        'View',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
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

// --- Tag List View Item Widget ---
class TagListViewItem extends StatelessWidget {
  final FileSystemItem item;
  final Function(FileSystemItem) onOpen;

  const TagListViewItem({super.key, required this.item, required this.onOpen});

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;
    final color = item.mainColor ?? colors.textPrimary;
    return GestureDetector(
      onTap: () => onOpen(item),
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: colors.surface.withOpacity(0.5),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: colors.border.withOpacity(0.3)),
        ),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              alignment: Alignment.center,
              child: Icon(
                item.iconShape ?? Icons.circle,
                size: 32,
                color: color,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                item.name,
                style: TextStyle(
                  color: colors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ),
            Text(
              '${item.items.length}',
              style: TextStyle(
                color: color,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(width: 8),
          ],
        ),
      ),
    );
  }
}