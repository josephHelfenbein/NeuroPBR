import 'package:flutter/material.dart';
import 'dart:math' as math;
import 'package:provider/provider.dart';
import 'scan_screen_new.dart';
import 'renderer_screen.dart';
import '../theme/theme_provider.dart';

// --- Data Models ---
class MaterialItem {
  final String id;
  final String name;
  final String type;
  final Color? mainColor;
  final IconData? iconShape;
  final List<String> previewImages;
  final List<MaterialItem> items;

  MaterialItem({
    required this.id,
    required this.name,
    required this.type,
    this.mainColor,
    this.iconShape,
    this.previewImages = const [],
    this.items = const [],
  });
}

// --- Initial Data ---
final List<MaterialItem> initialFilesystem = [
  MaterialItem(
    id: '1',
    name: 'Misc',
    type: 'folder',
    mainColor: const Color(0xFFF4A261),
    iconShape: Icons.circle,
    previewImages: const [
      'https://placehold.co/400x500/E76F51/FFFFFF?text=Misc+1',
      'https://placehold.co/400x500/2A9D8F/FFFFFF?text=Misc+2',
    ],
    items: [
      MaterialItem(id: '1-1', name: 'Random Asset 01', type: 'file'),
      MaterialItem(id: '1-2', name: 'Scratch Pad', type: 'file'),
      MaterialItem(id: '1-3', name: 'Reference', type: 'file'),
    ],
  ),
  MaterialItem(
    id: '2',
    name: 'Wood',
    type: 'folder',
    mainColor: const Color(0xFFA67C52),
    iconShape: Icons.square,
    previewImages: const [
      'https://placehold.co/400x500/8B4513/FFFFFF?text=Oak+Texture',
      'https://placehold.co/400x500/DEB887/FFFFFF?text=Pine+Grain',
    ],
    items: [
      MaterialItem(id: '2-1', name: 'Oak_Albedo.png', type: 'file'),
    ],
  ),
  MaterialItem(
    id: '3',
    name: 'Metal',
    type: 'folder',
    mainColor: const Color(0xFF94A3B8),
    iconShape: Icons.hexagon,
    previewImages: const [
      'https://placehold.co/400x500/475569/FFFFFF?text=Brushed+Steel',
      'https://placehold.co/400x500/CBD5E1/FFFFFF?text=Gold+Leaf',
    ],
    items: [
      MaterialItem(id: '3-1', name: 'Rust Map', type: 'file'),
      MaterialItem(id: '3-2', name: 'Scratches', type: 'file'),
      MaterialItem(id: '3-3', name: 'Iron.obj', type: 'file'),
      MaterialItem(id: '3-4', name: 'Copper.mat', type: 'file'),
    ],
  ),
  MaterialItem(
    id: '4',
    name: 'Fabric',
    type: 'folder',
    mainColor: const Color(0xFFE63946),
    iconShape: Icons.change_history,
    items: [
      MaterialItem(id: '4-1', name: 'Silk Pattern', type: 'file'),
      MaterialItem(id: '4-2', name: 'Wool.norm', type: 'file'),
      MaterialItem(id: '4-3', name: 'Denim.diff', type: 'file'),
    ],
  ),
  MaterialItem(
    id: '5',
    name: 'Stone',
    type: 'folder',
    mainColor: const Color(0xFF6C757D),
    iconShape: Icons.album,
    items: [
      MaterialItem(id: '5-1', name: 'Marble Tile', type: 'file'),
      MaterialItem(id: '5-2', name: 'Granite.rough', type: 'file'),
      MaterialItem(id: '5-3', name: 'Pavement.disp', type: 'file'),
      MaterialItem(id: '5-4', name: 'Slate.norm', type: 'file'),
    ],
  ),
  MaterialItem(
    id: '6',
    name: 'Plastic',
    type: 'folder',
    mainColor: const Color(0xFF457B9D),
    iconShape: Icons.square,
    items: [
      MaterialItem(id: '6-1', name: 'Shiny Red', type: 'file'),
      MaterialItem(id: '6-2', name: 'Matte Black', type: 'file'),
      MaterialItem(id: '6-3', name: 'Clearcoat', type: 'file'),
      MaterialItem(id: '6-4', name: 'PVC Pipe', type: 'file'),
    ],
  ),
];

// --- Utility Functions ---
Color getDarkerColor(Color color) {
  final r = math.max(0, color.red - 40);
  final g = math.max(0, color.green - 40);
  final b = math.max(0, color.blue - 40);
  return Color.fromARGB(color.alpha, r, g, b);
}

List<MaterialItem> getCurrentContent(List<String> pathIds) {
  List<MaterialItem> current = initialFilesystem;
  for (final id in pathIds) {
    final folder = current.firstWhere(
      (item) => item.id == id && item.type == 'folder',
      orElse: () => MaterialItem(id: '', name: '', type: ''),
    );
    if (folder.id.isEmpty) return [];
    current = folder.items;
  }
  return current;
}

// --- Main Screen ---
class MaterialLibraryScreen extends StatefulWidget {
  const MaterialLibraryScreen({super.key});

  @override
  State<MaterialLibraryScreen> createState() => _MaterialLibraryScreenState();
}

class _MaterialLibraryScreenState extends State<MaterialLibraryScreen> {
  List<String> _path = [];
  String _activeTab = 'Tags';
  String _viewMode = 'list';

  List<MaterialItem> get currentContent => getCurrentContent(_path);
  bool get isRoot => _path.isEmpty;

  void _handleOpen(MaterialItem item) {
    if (item.type == 'folder') {
      setState(() {
        _path = [..._path, item.id];
      });
    } else {
      debugPrint('Opening file: ${item.name}');
    }
  }

  void _handleBack() {
    setState(() {
      _path = _path.sublist(0, _path.length - 1);
    });
  }

  void _handleTabChange(String tab) {
    setState(() {
      _activeTab = tab;
      _path = [];
    });
  }

  void _toggleViewMode() {
    setState(() {
      _viewMode = _viewMode == 'cards' ? 'list' : 'cards';
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Container(
          margin: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: colors.background,
            borderRadius: BorderRadius.circular(48),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.5),
                blurRadius: 40,
                spreadRadius: 10,
              ),
            ],
          ),
          child: Column(
            children: [
              // Header
              _buildHeader(colors),

              // Main Content
              Expanded(
                child: _buildMainContent(colors),
              ),

              // Bottom Navigation
              _buildBottomNav(colors),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(colors) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Left Button
          if (_activeTab == 'Tags' && isRoot)
            _buildIconButton(
              icon: _viewMode == 'cards' ? Icons.view_list : Icons.grid_view,
              onPressed: _toggleViewMode,
              backgroundColor: colors.surface,
              colors: colors,
            )
          else
            const SizedBox(width: 40),

          // Center Label
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
                      letterSpacing: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Right Button
          _buildIconButton(
            icon: Icons.view_in_ar,
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const RendererScreen()),
              );
            },
            backgroundColor: Colors.red,
            shadow: BoxShadow(
              color: Colors.red.withOpacity(0.5),
              blurRadius: 15,
              spreadRadius: 2,
            ),
            colors: colors,
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required VoidCallback onPressed,
    required Color backgroundColor,
    BoxShadow? shadow,
    required colors,
  }) {
    return Container(
      decoration: shadow != null
          ? BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              boxShadow: [shadow],
            )
          : null,
      child: Material(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(20),
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(20),
          child: Container(
            width: 40,
            height: 40,
            alignment: Alignment.center,
            child: Icon(icon, size: 20, color: Colors.white),
          ),
        ),
      ),
    );
  }

  Widget _buildMainContent(colors) {
    if (_activeTab == 'Stats') {
      return _buildStatsView(colors);
    } else if (_activeTab == 'Settings') {
      return _buildSettingsView(colors);
    } else {
      return _buildTagsView(colors);
    }
  }

  Widget _buildStatsView(colors) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Overview',
            style: TextStyle(
              color: colors.textPrimary,
              fontSize: 30,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 24),

          // Summary Cards
          Row(
            children: [
              Expanded(
                child: _buildStatCard(
                  icon: Icons.view_in_ar,
                  iconColor: Colors.blue,
                  value: '1,240',
                  label: 'TOTAL RENDERS',
                  colors: colors,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildStatCard(
                  icon: Icons.bolt,
                  iconColor: Colors.purple,
                  value: '85%',
                  label: 'EFFICIENCY',
                  colors: colors,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Material Usage
          _buildMaterialUsage(colors),
          const SizedBox(height: 24),

          // Cloud Storage
          _buildCloudStorage(colors),
        ],
      ),
    );
  }

  Widget _buildStatCard({
    required IconData icon,
    required Color iconColor,
    required String value,
    required String label,
    required colors,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colors.surface.withOpacity(0.8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, size: 20, color: iconColor),
          ),
          const SizedBox(height: 12),
          Text(
            value,
            style: TextStyle(
              color: colors.textPrimary,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(
              color: colors.textSecondary,
              fontSize: 10,
              fontWeight: FontWeight.w500,
              letterSpacing: 1.2,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMaterialUsage(colors) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: colors.surface.withOpacity(0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Material Usage',
                style: TextStyle(
                  color: colors.textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Icon(Icons.pie_chart, size: 16, color: colors.textSecondary),
            ],
          ),
          const SizedBox(height: 16),
          _buildUsageBar('Metal', 0.45, const Color(0xFF9CA3AF), colors),
          const SizedBox(height: 12),
          _buildUsageBar('Wood', 0.30, const Color(0xFFA67C52), colors),
          const SizedBox(height: 12),
          _buildUsageBar('Plastic', 0.15, const Color(0xFF457B9D), colors),
        ],
      ),
    );
  }

  Widget _buildUsageBar(String label, double percentage, Color color, colors) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: TextStyle(color: colors.textSecondary, fontSize: 12),
            ),
            Text(
              '${(percentage * 100).toInt()}%',
              style: TextStyle(color: colors.textSecondary, fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: percentage,
            backgroundColor: colors.border,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 8,
          ),
        ),
      ],
    );
  }

  Widget _buildCloudStorage(colors) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: colors.surface.withOpacity(0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Cloud Storage',
                style: TextStyle(
                  color: colors.textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'Pro',
                  style: TextStyle(
                    color: Colors.green[400],
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '45.2',
                style: TextStyle(
                  color: colors.textPrimary,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 4),
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text(
                  'GB / 100 GB',
                  style: TextStyle(color: colors.textSecondary, fontSize: 14),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: Container(
              height: 12,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.blue, Colors.purple],
                  stops: const [0.0, 0.45],
                ),
              ),
              child: FractionallySizedBox(
                alignment: Alignment.centerLeft,
                widthFactor: 0.45,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.blue, Colors.purple],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSettingsView(colors) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Settings',
            style: TextStyle(
              color: colors.textPrimary,
              fontSize: 30,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 24),

          // Account Section
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: colors.surface.withOpacity(0.5),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: colors.border.withOpacity(0.3)),
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.orange, Colors.red],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      'JD',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'John Doe',
                        style: TextStyle(
                          color: colors.textPrimary,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        'Pro Member',
                        style: TextStyle(
                          color: colors.textSecondary,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(Icons.chevron_right, color: colors.textSecondary),
              ],
            ),
          ),
          const SizedBox(height: 32),

          // General Section
          Text(
            'GENERAL',
            style: TextStyle(
              color: colors.textSecondary,
              fontSize: 10,
              fontWeight: FontWeight.bold,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 16),
          _buildSettingItem(
            icon: Icons.cloud,
            iconColor: Colors.blue,
            label: 'Cloud Sync',
            enabled: true,
            colors: colors,
          ),
          const SizedBox(height: 12),
          _buildSettingItem(
            icon: Icons.notifications,
            iconColor: Colors.purple,
            label: 'Notifications',
            enabled: true,
            colors: colors,
          ),
          const SizedBox(height: 32),

          // Rendering Section
          Text(
            'RENDERING',
            style: TextStyle(
              color: colors.textSecondary,
              fontSize: 10,
              fontWeight: FontWeight.bold,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 16),
          _buildSettingItem(
            icon: Icons.bolt,
            iconColor: Colors.orange,
            label: 'High Quality Previews',
            enabled: false,
            colors: colors,
          ),
          const SizedBox(height: 32),

          // Log Out Button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () {},
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red.withOpacity(0.1),
                foregroundColor: Colors.red[400],
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 0,
              ),
              child: Text(
                'Log Out',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSettingItem({
    required IconData icon,
    required Color iconColor,
    required String label,
    required bool enabled,
    required colors,
  }) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: colors.surfaceVariant.withOpacity(0.5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, size: 18, color: iconColor),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                color: colors.textPrimary,
                fontSize: 15,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Icon(
            Icons.toggle_on,
            size: 32,
            color: enabled ? Colors.green : colors.textSecondary,
          ),
        ],
      ),
    );
  }

  Widget _buildTagsView() {
    if (isRoot) {
      return Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: Text(
              _viewMode == 'cards' ? 'RECENT STACKS' : 'TAG LIBRARY',
              style: TextStyle(
                color: Colors.grey[600],
                fontSize: 10,
                fontWeight: FontWeight.bold,
                letterSpacing: 2,
              ),
            ),
          ),
          Expanded(
            child: _viewMode == 'cards'
                ? _buildCardsView()
                : _buildListView(),
          ),
        ],
      );
    } else {
      return _buildAssetsView();
    }
  }

  Widget _buildCardsView() {
    return PageView.builder(
      itemCount: initialFilesystem.length,
      padEnds: false,
      controller: PageController(viewportFraction: 0.85),
      itemBuilder: (context, index) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: ProjectCard(
            project: initialFilesystem[index],
            onOpen: _handleOpen,
          ),
        );
      },
    );
  }

  Widget _buildListView() {
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      children: [
        ...initialFilesystem.map((project) => TagListViewItem(
              project: project,
              onOpen: _handleOpen,
            )),
        const SizedBox(height: 32),
        Column(
          children: [
            Text(
              'Material Library',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontFamily: 'serif',
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                'Organize your PBR assets by physical properties.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 14,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 80),
      ],
    );
  }

  Widget _buildAssetsView() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Row(
            children: [
              _buildIconButton(
                icon: Icons.chevron_left,
                onPressed: _handleBack,
                backgroundColor: const Color(0xFF262626),
              ),
              const SizedBox(width: 12),
              Text(
                'Assets',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF262626).withOpacity(0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: currentContent.isEmpty
                  ? Center(
                      child: Text(
                        'No assets found.',
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 16,
                        ),
                      ),
                    )
                  : ListView.builder(
                      itemCount: currentContent.length,
                      itemBuilder: (context, index) {
                        return FileSystemItem(
                          item: currentContent[index],
                          onOpen: _handleOpen,
                        );
                      },
                    ),
            ),
          ),
        ],
      ),
    );
  }

  // In MaterialLibraryScreen - add to your existing build method
  Widget _buildBottomNav(colors) {
    return Container(
      decoration: BoxDecoration(
        border: Border(
          top: BorderSide(color: colors.border.withOpacity(0.3)),
        ),
      ),
      padding: const EdgeInsets.only(bottom: 16, top: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          NavItem(
            icon: Icons.bar_chart,
            label: 'Stats',
            active: _activeTab == 'Stats',
            onTap: () => _handleTabChange('Stats'),
          ),
          NavItem(
            icon: Icons.local_offer,
            label: 'Tags',
            active: _activeTab == 'Tags',
            onTap: () => _handleTabChange('Tags'),
          ),
          NavItem(
            icon: Icons.settings,
            label: 'Settings',
            active: _activeTab == 'Settings',
            onTap: () => _handleTabChange('Settings'),
          ),
        ],
      ),
    );
  }
}

// --- Components ---

class FileSystemItem extends StatelessWidget {
  final MaterialItem item;
  final Function(MaterialItem) onOpen;

  const FileSystemItem({
    super.key,
    required this.item,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final isFolder = item.type == 'folder';

    return InkWell(
      onTap: () => onOpen(item),
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(12),
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Icon(
              isFolder ? Icons.folder : Icons.insert_drive_file,
              color: isFolder ? Colors.yellow[700] : Colors.grey[600],
              size: 24,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                item.name,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (isFolder)
              Text(
                '${item.items.length} items',
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 14,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class ProjectCard extends StatelessWidget {
  final MaterialItem project;
  final Function(MaterialItem) onOpen;

  const ProjectCard({
    super.key,
    required this.project,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => onOpen(project),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 20),
        child: Stack(
          children: [
            // Background layers
            if (project.previewImages.isNotEmpty || project.mainColor != null) ...[
              Positioned.fill(
                child: Transform.translate(
                  offset: const Offset(0, -20),
                  child: Transform.rotate(
                    angle: 0.087,
                    child: Container(
                      decoration: BoxDecoration(
                        color: project.mainColor != null
                            ? getDarkerColor(getDarkerColor(project.mainColor!))
                            : Colors.grey[800],
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.4),
                            blurRadius: 15,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              Positioned.fill(
                child: Transform.translate(
                  offset: const Offset(0, -10),
                  child: Transform.rotate(
                    angle: -0.052,
                    child: Container(
                      decoration: BoxDecoration(
                        color: project.mainColor != null
                            ? getDarkerColor(project.mainColor!)
                            : Colors.grey[700],
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.4),
                            blurRadius: 15,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],

            // Front card
            Transform.rotate(
              angle: -0.017,
              child: Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: project.mainColor ?? Colors.grey[600],
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.9),
                      blurRadius: 50,
                      spreadRadius: 10,
                    ),
                  ],
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      project.name,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 28,
                        fontWeight: FontWeight.w900,
                        height: 1.2,
                        shadows: [
                          Shadow(
                            color: Colors.black.withOpacity(0.5),
                            blurRadius: 3,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${project.items.length} renders',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () => onOpen(project),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.black.withOpacity(0.2),
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        'View',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
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

class TagListViewItem extends StatelessWidget {
  final MaterialItem project;
  final Function(MaterialItem) onOpen;

  const TagListViewItem({
    super.key,
    required this.project,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => onOpen(project),
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.all(12),
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF262626).withOpacity(0.5),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: const Color(0xFF404040).withOpacity(0.3),
          ),
        ),
        child: Row(
          children: [
            SizedBox(
              width: 40,
              height: 40,
              child: Icon(
                project.iconShape ?? Icons.circle,
                size: 32,
                color: project.mainColor,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                project.name,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ),
            Text(
              '${project.items.length}',
              style: TextStyle(
                color: project.mainColor,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class NavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  const NavItem({
    super.key,
    required this.icon,
    required this.label,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        width: 64,
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: active ? Colors.white.withOpacity(0.1) : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                icon,
                size: 22,
                color: active ? Colors.white : Colors.grey[600],
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: active ? Colors.white : Colors.grey[600],
                fontSize: 10,
                fontWeight: FontWeight.w500,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
