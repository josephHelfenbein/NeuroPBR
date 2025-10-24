# MetalRenderer

C++ and Metal-based renderer for real-time visualization inside the iOS app.

- Written in C++ (render logic) and Metal (shaders) with an Objective-C++ bridge for iOS integration.
- Displays Core ML–generated PBR textures (albedo, roughness, metallic, normal, AO) in real time.
- Uses IBL lighting for photorealistic previews and consistency with the dataset renderer.
- Integrated with Flutter through Objective-C++ platform channels for seamless UI interaction.
