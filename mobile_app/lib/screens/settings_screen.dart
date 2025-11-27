import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'scan_screen.dart';
import '../theme/theme_provider.dart';
import '../providers/preferences_provider.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final prefs = Provider.of<PreferencesProvider>(context);
    final colors = themeProvider.colors;

    return Scaffold(
      backgroundColor: colors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context, colors),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(24),
                children: [
                  Text(
                    'Settings',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: colors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 24),

                  _buildSectionHeader('DISPLAY', colors),
                  _buildSelectionTile(
                    'Default View Mode',
                    // Icon changes based on mode
                    prefs.tagsViewMode == 'cards' ? Icons.grid_view_rounded : Icons.format_list_bulleted,
                    colors.accent,
                    // Text shows current mode
                    prefs.tagsViewMode == 'cards' ? 'CARDS' : 'LIST',
                    // Tap triggers the provider toggle
                        () => prefs.toggleViewMode(),
                    colors,
                  ),
                  const SizedBox(height: 24),

                  // colors just don't look good
                  // _buildSectionHeader('APPEARANCE', colors),
                  // _buildToggleTile(
                  //   'Dark Mode',
                  //   Icons.dark_mode,
                  //   colors.accent,
                  //   themeProvider.isDarkMode,
                  //       () => themeProvider.toggleTheme(),
                  //   colors,
                  // ),
                  // const SizedBox(height: 24),

                  _buildSectionHeader('GENERAL', colors),
                  _buildToggleTile(
                    'Notifications',
                    Icons.notifications,
                    colors.accent,
                    true,
                        () {},
                    colors,
                  ),
                  const SizedBox(height: 24),

                  _buildSectionHeader('RENDERING', colors),
                  _buildToggleTile(
                    'High Quality Previews',
                    Icons.bolt,
                    colors.accent,
                    false,
                        () {},
                    colors,
                  ),
                ],
              ),
            ),
            // Bottom nav is now handled by MainTabScreen
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, colors) {
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

  Widget _buildSectionHeader(String title, colors) {
    return Padding(
      padding: const EdgeInsets.only(left: 8, bottom: 12),
      child: Text(
        title,
        style: TextStyle(
          color: colors.textSecondary,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.5,
        ),
      ),
    );
  }

  Widget _buildToggleTile(
      String title,
      IconData icon,
      Color iconColor,
      bool value,
      VoidCallback onTap,
      colors,
      ) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: colors.surfaceVariant,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
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
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  color: colors.textPrimary,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            Icon(
              value ? Icons.toggle_on : Icons.toggle_off,
              size: 32,
              color: value ? Colors.greenAccent : colors.textSecondary,
            ),
          ],
        ),
      ),
    );
  }

  // Helper widget for settings that cycle through options (like View Mode)
  Widget _buildSelectionTile(
      String title,
      IconData icon,
      Color iconColor,
      String statusText,
      VoidCallback onTap,
      dynamic colors,
      ) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: colors.surfaceVariant,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
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
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  color: colors.textPrimary,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: colors.background,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: colors.border.withOpacity(0.5)),
              ),
              child: Text(
                statusText,
                style: TextStyle(
                  color: colors.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
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