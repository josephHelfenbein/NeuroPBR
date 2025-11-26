import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'scan_screen_new.dart';
import '../theme/theme_provider.dart';
import 'tags_screen.dart'; // ADD THIS
import 'settings_screen.dart'; // ADD THIS
import 'nav_item.dart'; // ADD THIS

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  @override
  Widget build(BuildContext context) {
    final colors = Provider.of<ThemeProvider>(context).colors;

    return Scaffold(
      backgroundColor: colors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context, colors),
            Expanded(child: _buildBody(colors)),
            // REMOVE: _buildBottomNav(colors),
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

  Widget _buildBody(colors) {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text(
          'Overview',
          style: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.bold,
            color: colors.textPrimary,
          ),
        ),
        const SizedBox(height: 24),

        // Total Renders - Clean and balanced
        _buildTotalRendersCard(colors),
        const SizedBox(height: 20),

        // Material Usage
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: colors.surface.withOpacity(0.6),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: colors.border.withOpacity(0.5)),
          ),
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Material Usage',
                    style: TextStyle(
                      color: colors.textPrimary,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Icon(Icons.pie_chart, size: 16, color: colors.textSecondary),
                ],
              ),
              const SizedBox(height: 20),
              _buildProgressBar('Metal', 0.45, const Color(0xFF94A3B8), colors),
              const SizedBox(height: 12),
              _buildProgressBar('Wood', 0.30, const Color(0xFFA67C52), colors),
              const SizedBox(height: 12),
              _buildProgressBar('Plastic', 0.15, const Color(0xFF457B9D), colors),
              const SizedBox(height: 12),
              _buildProgressBar('Fabric', 0.25, const Color(0xFFE63946), colors),
              const SizedBox(height: 12),
              _buildProgressBar('Stone', 0.20, const Color(0xFF6C757D), colors),
              const SizedBox(height: 12),
              _buildProgressBar('Misc', 0.35, const Color(0xFFF4A261), colors),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildTotalRendersCard(colors) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: colors.surface.withOpacity(0.8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.border.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(
            '1,240',
            style: TextStyle(
              color: colors.textPrimary,
              fontSize: 36,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'TOTAL RENDERS',
            style: TextStyle(
              color: colors.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar(String label, double value, Color color, colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: TextStyle(color: colors.textSecondary, fontSize: 12),
            ),
            Text(
              '${(value * 100).toInt()}%',
              style: TextStyle(color: colors.textSecondary, fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value,
            minHeight: 6,
            backgroundColor: colors.border,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ],
    );
  }
}

// --- Icon Button Widget --- (FIXED - remove the duplicate build method)
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