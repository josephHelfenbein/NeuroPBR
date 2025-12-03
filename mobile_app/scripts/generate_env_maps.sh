#!/bin/bash
# Generate environment maps using the Metal prefilter
# This script compiles and runs a macOS command-line tool that uses the same
# Metal shaders as the iOS app to precompute environment maps from HDRIs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/GenerateEnvMaps/build"
METAL_RENDERER_DIR="$PROJECT_DIR/ios/MetalRenderer"

# Default paths
INPUT_DIR="$PROJECT_DIR/assets/hdris"
OUTPUT_DIR="$PROJECT_DIR/assets/env_maps"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -i, --input <dir>   Input directory containing HDR files"
            echo "  -o, --output <dir>  Output directory for generated maps"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "=== NeuroPBR Environment Map Generator ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Compiling Metal shaders and tool..."

# Compile Metal shaders to metallib
xcrun -sdk macosx metal -c "$METAL_RENDERER_DIR/EnvironmentPrefilter.metal" -o "$BUILD_DIR/EnvironmentPrefilter.air"
xcrun -sdk macosx metallib "$BUILD_DIR/EnvironmentPrefilter.air" -o "$BUILD_DIR/default.metallib"

# Compile the Objective-C++ files
clang++ -std=c++17 -fobjc-arc -O2 \
    -framework Foundation \
    -framework Metal \
    -framework MetalKit \
    -framework CoreGraphics \
    -I"$METAL_RENDERER_DIR" \
    "$METAL_RENDERER_DIR/EnvironmentPrefilter.mm" \
    "$SCRIPT_DIR/GenerateEnvMaps/main.mm" \
    -o "$BUILD_DIR/generate_env_maps"

echo "Compilation successful!"
echo ""

# Run the tool
echo "Generating environment maps..."
cd "$BUILD_DIR"
# Filter out AGX driver warnings (harmless Metal driver messages on Apple Silicon)
./generate_env_maps -i "$INPUT_DIR" -o "$OUTPUT_DIR" "$@" 2>&1 | grep -v "AGX: Texture read/write assertion failed"

echo ""
echo "Done! Environment maps saved to: $OUTPUT_DIR"
