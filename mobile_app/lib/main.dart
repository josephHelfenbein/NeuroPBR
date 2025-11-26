// lib/main.dart

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:neuro_pbr/screens/main_tab_screen.dart';
import 'package:neuro_pbr/screens/animated_splash_screen.dart';
import 'package:neuro_pbr/providers/preferences_provider.dart';
import 'package:provider/provider.dart';
import 'screens/carousel_screen.dart';
import 'theme/theme_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ChangeNotifierProvider(create: (_) => PreferencesProvider()),
      ],
        child: const NeuroPBRApp(),
    ),
  );
}

class NeuroPBRApp extends StatelessWidget {
  const NeuroPBRApp({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    return MaterialApp(
      title: 'NeuroPBR',
      debugShowCheckedModeBanner: false,
      theme: themeProvider.themeData,
      home: AnimatedSplashScreen(
        nextScreen: const MainTabScreen(),
      ),
    );
  }
}
