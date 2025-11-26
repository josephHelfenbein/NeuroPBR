// main_tab_screen.dart
import 'package:flutter/material.dart';
import 'stats_screen.dart';
import 'tags_screen.dart';
import 'settings_screen.dart';
import 'nav_item.dart';

class MainTabScreen extends StatefulWidget {
  const MainTabScreen({super.key});

  @override
  State<MainTabScreen> createState() => _MainTabScreenState();
}

class _MainTabScreenState extends State<MainTabScreen> {
  int _currentIndex = 1;
  final GlobalKey<StatsScreenState> _statsKey = GlobalKey();
  late final List<Widget> _screens;

  @override
  void initState() {
    super.initState();
    _screens = [
      StatsScreen(key: _statsKey),
      const TagsScreen(),
      const SettingsScreen(),
    ];
  }

  void _onTabTapped(int index) {
    setState(() {
      _currentIndex = index;
    });
    if (index == 0) {
      _statsKey.currentState?.refresh();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        border: Border(
          top: BorderSide(color: Colors.grey[800]!),
        ),
      ),
      padding: const EdgeInsets.only(bottom: 16, top: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          NavItem(
            icon: Icons.bar_chart,
            label: 'Stats',
            active: _currentIndex == 0,
            onTap: () => _onTabTapped(0),
          ),
          NavItem(
            icon: Icons.local_offer,
            label: 'Tags',
            active: _currentIndex == 1,
            onTap: () => _onTabTapped(1),
          ),
          NavItem(
            icon: Icons.settings,
            label: 'Settings',
            active: _currentIndex == 2,
            onTap: () => _onTabTapped(2),
          ),
        ],
      ),
    );
  }
}