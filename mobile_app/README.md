# NeuroPBR Mobile App

A Flutter-based iOS application designed to capture real-world surfaces, reconstruct their PBR material properties on-device, and render them in real-time using a custom Metal backend.

## Overview

The mobile app serves as the primary interface for the NeuroPBR pipeline in the field. It guides users to capture three specific images of a surface, processes them using a lightweight "Student" model (distilled from the larger server-side teacher model), and immediately visualizes the resulting Albedo, Normal, Roughness, and Metallic maps.

## Key Features

- **Multi-View Capture**: Guided UI for taking the three required input images for material reconstruction.
- **On-Device Inference**: Runs a Core ML version of the Student model to generate PBR maps locally without needing a server connection.
- **Real-Time Metal Renderer**:
  - Uses a custom C++ and Metal PBR IBL renderer.
  - Exposed to Flutter via an Objective-C++ bridge (`MetalBridge.mm`).
  - Supports high-fidelity previewing with environment lighting.
- **Material Export**: Save or share the generated PBR texture maps.

## Architecture

The app is built with **Flutter** for the UI and logic, but relies heavily on native iOS technologies for performance:

1.  **Flutter Layer (`lib/`)**: Handles the UI, camera capture flow, and coordinates the native plugins.
2.  **Dart Plugin (`neuropbr_plugin.dart`)**: Communicates with the native iOS code via MethodChannels.
3.  **Native iOS Layer (`ios/`)**:
    -   **Runner**: Contains the application logic and the Core ML model (`pbr_model.mlpackage`).
    -   **Metal Renderer**: A real-time renderer sharing core logic with the desktop CUDA renderer but optimized for Apple Silicon. See [MetalRenderer/README.md](ios/MetalRenderer/README.md) for deep technical details.

## Prerequisites

- **macOS** with Xcode installed.
- **Flutter SDK** (latest stable channel recommended).
- **CocoaPods** for iOS dependency management.
- An iOS device (simulator support may be limited for Metal/Camera features).

## Getting Started

1.  **Install Dependencies**:
    ```bash
    cd mobile_app
    flutter pub get
    ```

2.  **Setup iOS Project**:
    *   The repository includes a pre-compiled Core ML model at `ios/Runner/pbr_model.mlpackage`, so no extra setup is needed to run the default model. If you trained a custom model, ensure it replaces this file.
    *   Open `ios/Runner.xcworkspace` in Xcode.
    *   Ensure `pbr_model.mlpackage` is added to the "Runner" target (drag and drop it into the project navigator if missing).
    *   **Important**: Ensure `PBRModelHandler.swift` is added to the "Runner" target in Xcode. If you see "Cannot find type 'PBRModelHandler'", right-click the Runner group in Xcode -> "Add Files to 'Runner'..." and select `ios/Runner/PBRModelHandler.swift`.
    ```bash
    cd ios
    pod install
    cd ..
    ```

3.  **Run the App**:
    Connect your iOS device and run:
    ```bash
    flutter run
    ```
