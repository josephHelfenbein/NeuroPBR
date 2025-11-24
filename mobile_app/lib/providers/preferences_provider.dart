import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class PreferencesProvider extends ChangeNotifier {
  // Default to list
  String _tagsViewMode = 'list';

  String get tagsViewMode => _tagsViewMode;

  PreferencesProvider() {
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    _tagsViewMode = prefs.getString('tagsViewMode') ?? 'list';
    notifyListeners();
  }

  Future<void> toggleViewMode() async {
    final prefs = await SharedPreferences.getInstance();
    if (_tagsViewMode == 'cards') {
      _tagsViewMode = 'list';
    } else {
      _tagsViewMode = 'cards';
    }
    await prefs.setString('tagsViewMode', _tagsViewMode);
    notifyListeners();
  }
}