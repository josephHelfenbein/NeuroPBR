import 'package:flutter/material.dart';

class AppTheme {
  // Dark theme colors
  static const Color darkBackground = Color(0xFF171717);
  static const Color darkSurface = Color(0xFF262626);
  static const Color darkBorder = Color(0xFF3A3A3A);

  static const Color lightBackground = Color(0xFFE9F2FF); // New Background (From Cube Demo)
  static const Color lightSurface = Color(0xFFFAFBFF);    // Cooler White Surface
  static const Color lightBorder = Color(0xFFD1DDEB);     // Muted Blue-Grey Border

  // Shared colors
  static const Color accentRed = Color(0xFFEF4444);

  static ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: darkBackground,
    colorScheme: const ColorScheme.dark(
      primary: accentRed,
      surface: darkSurface,
      background: darkBackground,
    ),
  );

  static ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    scaffoldBackgroundColor: lightBackground,
    colorScheme: const ColorScheme.light(
      primary: accentRed,
      surface: lightSurface,
      background: lightBackground,
    ),
  );
}

class AppColors {
  final Color background;
  final Color surface;
  final Color surfaceVariant;
  final Color border;
  final Color textPrimary;
  final Color textSecondary;
  final Color accent;
  final Color cardBackground;
  final Brightness statusBarBrightness;

  const AppColors({
    required this.background,
    required this.surface,
    required this.surfaceVariant,
    required this.border,
    required this.textPrimary,
    required this.textSecondary,
    required this.accent,
    required this.cardBackground,
    required this.statusBarBrightness,
  });

  static const AppColors dark = AppColors(
    background: Color(0xFF171717),
    surface: Color(0xFF262626),
    surfaceVariant: Color(0xFF1a1a1a),
    border: Color(0xFF3A3A3A),
    textPrimary: Colors.white,
    textSecondary: Color(0xFF9CA3AF),
    accent: Color(0xFFEF4444),
    cardBackground: Color(0xFF262626),
    statusBarBrightness: Brightness.light,
  );

  static const AppColors light = AppColors(
    background: Color(0xFFE9F2FF),     // Light Blue/Grey from Cube
    surface: Color(0xFFFAFBFF),        // Cooler White Surface
    surfaceVariant: Color(0xFFE0E9F4), // Muted light blue for distinction
    border: Color(0xFFD1DDEB),         // Muted Blue-Grey Border
    textPrimary: Color(0xFF1C2738),    // Dark, readable text
    textSecondary: Color(0xFF6A7E9A),  // Muted secondary text
    accent: Color(0xFFEF4444),         // Red Accent
    cardBackground: Color(0xFFFAFBFF), // Cooler White Surface
    statusBarBrightness: Brightness.dark,
  );
}
