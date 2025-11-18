#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace neuropbr {

enum class ToneMapping : uint32_t {
    ACES = 0,
    Filmic = 1
};

enum class PreviewChannel : uint32_t {
    Final = 0,
    Albedo = 1,
    Roughness = 2,
    Metallic = 3,
    Normal = 4
};

enum class MaterialSlot : uint32_t {
    Albedo = 0,
    Normal = 1,
    Roughness = 2,
    Metallic = 3,
    COUNT
};

struct TextureHandle {
    uint64_t value = 0;
    constexpr explicit operator bool() const { return value != 0; }
};

struct TextureInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    bool isHDR = false;
};

struct TextureBinding {
    TextureHandle handle;
    TextureInfo info;
};

struct CameraParameters {
    float position[3] = {0.0f, 0.0f, 5.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float fovY = 45.0f;
    float nearZ = 0.01f;
    float farZ = 100.0f;
};

struct LightingControls {
    float exposure = 0.0f;
    float intensity = 1.0f;
    float rotation = 0.0f;
};

struct PreviewControls {
    float baseColorTint[3] = {1.0f, 1.0f, 1.0f};
    float roughnessMultiplier = 1.0f;
    float metallicMultiplier = 1.0f;
    bool enableNormalMap = true;
    bool showNormals = false;
    bool showWireframe = false;
    PreviewChannel channel = PreviewChannel::Final;
    ToneMapping toneMapping = ToneMapping::ACES;
};

struct EnvironmentControls {
    uint32_t environmentId = 0;
    TextureHandle environmentHandle{};
    TextureHandle irradianceHandle{};
    TextureHandle prefilteredHandle{};
    TextureHandle brdfHandle{};
    float lodCount = 1.0f;
};

struct RendererConfig {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct FrameUniforms {
    std::array<float, 16> viewProjection{};
    std::array<float, 16> invView{};
    std::array<float, 4> cameraPosTime{};
    std::array<float, 4> resolutionExposure{};
    std::array<float, 4> iblParams{};
    std::array<float, 4> toneMapping{};
};

struct MaterialUniforms {
    std::array<float, 4> baseTint{};
    std::array<float, 4> scalars{}; // x roughness multiplier, y metallic multiplier, z unused, w channel code
    std::array<float, 4> featureToggles{}; // x normal, y unused, z show normals, w wireframe
};

struct FrameDescriptor {
    FrameUniforms frame;
    MaterialUniforms material;
    std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> textures;
    EnvironmentControls environment;
};

class Renderer {
  public:
    explicit Renderer(const RendererConfig &config);
    ~Renderer();

    void setCamera(const CameraParameters &camera);
    void setLighting(const LightingControls &lighting);
    void setPreviewControls(const PreviewControls &preview);
    void setEnvironment(const EnvironmentControls &environment);
    void setOutputSize(uint32_t width, uint32_t height);

    void upsertMaterial(uint32_t materialId, const std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> &bindings);
    void updateMaterialTexture(uint32_t materialId, MaterialSlot slot, const TextureBinding &binding);

    bool buildFrameDescriptor(uint32_t materialId, FrameDescriptor &outDescriptor) const;

  private:
    struct MaterialState {
        uint32_t id = 0;
        std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> textures{};
    };

    RendererConfig config_{};
    CameraParameters camera_{};
    LightingControls lighting_{};
    PreviewControls preview_{};
    EnvironmentControls environment_{};

    std::unordered_map<uint32_t, MaterialState> materials_{};

    void computeMatrices(FrameUniforms &uniforms) const;
};

extern "C" {
Renderer *npbrCreateRenderer(const RendererConfig &config);
void npbrDestroyRenderer(Renderer *renderer);
void npbrSetCamera(Renderer *renderer, const CameraParameters &camera);
void npbrSetLighting(Renderer *renderer, const LightingControls &lighting);
void npbrSetPreview(Renderer *renderer, const PreviewControls &preview);
void npbrSetEnvironment(Renderer *renderer, const EnvironmentControls &environment);
void npbrSetOutputSize(Renderer *renderer, uint32_t width, uint32_t height);
void npbrUpsertMaterial(Renderer *renderer, uint32_t materialId, const std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> &bindings);
void npbrUpdateMaterialTexture(Renderer *renderer, uint32_t materialId, MaterialSlot slot, const TextureBinding &binding);
bool npbrBuildFrame(Renderer *renderer, uint32_t materialId, FrameDescriptor &outDescriptor);
}

} // namespace neuropbr
