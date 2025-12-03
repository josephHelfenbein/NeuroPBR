#include "Renderer.hpp"

#include <cmath>
#include <cstring>

namespace {
constexpr float kPi = 3.14159265359f;

void normalize3(float v[3]) {
    const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 0.0f) {
        const float inv = 1.0f / len;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    }
}

void cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void subtract(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

void identity(std::array<float, 16> &m) {
    m = {1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 1.0f};
}

} // namespace

namespace neuropbr {

Renderer::Renderer(const RendererConfig &config) : config_(config) {}

Renderer::~Renderer() = default;

void Renderer::setCamera(const CameraParameters &camera) { camera_ = camera; }

void Renderer::setLighting(const LightingControls &lighting) { lighting_ = lighting; }

void Renderer::setPreviewControls(const PreviewControls &preview) { preview_ = preview; }

void Renderer::setEnvironment(const EnvironmentControls &environment) { environment_ = environment; }

void Renderer::setOutputSize(uint32_t width, uint32_t height) {
    config_.width = width;
    config_.height = height;
}

void Renderer::upsertMaterial(uint32_t materialId, const std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> &bindings) {
    MaterialState &state = materials_[materialId];
    state.id = materialId;
    state.textures = bindings;
}

void Renderer::updateMaterialTexture(uint32_t materialId, MaterialSlot slot, const TextureBinding &binding) {
    auto it = materials_.find(materialId);
    if (it == materials_.end()) {
        MaterialState newState;
        newState.id = materialId;
        newState.textures[static_cast<size_t>(slot)] = binding;
        materials_[materialId] = newState;
        return;
    }
    it->second.textures[static_cast<size_t>(slot)] = binding;
}

bool Renderer::buildFrameDescriptor(uint32_t materialId, FrameDescriptor &outDescriptor) const {
    auto it = materials_.find(materialId);
    if (it == materials_.end()) {
        return false;
    }

    outDescriptor.textures = it->second.textures;
    outDescriptor.environment = environment_;
    computeMatrices(outDescriptor.frame);

    outDescriptor.material.baseTint = {preview_.baseColorTint[0], preview_.baseColorTint[1], preview_.baseColorTint[2], 1.0f};
    outDescriptor.material.scalars = {preview_.roughnessMultiplier, preview_.metallicMultiplier, 0.0f, static_cast<float>(preview_.channel)};
    outDescriptor.material.featureToggles = {preview_.enableNormalMap ? 1.0f : 0.0f,
                                             0.0f,
                                             preview_.showNormals ? 1.0f : 0.0f,
                                             preview_.showWireframe ? 1.0f : 0.0f};

    outDescriptor.frame.iblParams = {lighting_.intensity, lighting_.rotation, environment_.lodCount, 0.0f};
    outDescriptor.frame.toneMapping = {static_cast<float>(preview_.toneMapping), lighting_.exposure, 0.0f, 0.0f};

    return true;
}

void Renderer::computeMatrices(FrameUniforms &uniforms) const {
    identity(uniforms.cameraToWorld);
    identity(uniforms.worldToCamera);

    // Calculate effective position based on zoom
    float diff[3];
    subtract(camera_.position, camera_.target, diff);
    
    float scale = 1.0f / std::max(0.001f, preview_.zoom);
    diff[0] *= scale;
    diff[1] *= scale;
    diff[2] *= scale;

    float effectivePos[3];
    effectivePos[0] = camera_.target[0] + diff[0];
    effectivePos[1] = camera_.target[1] + diff[1];
    effectivePos[2] = camera_.target[2] + diff[2];

    float forward[3];
    subtract(camera_.target, effectivePos, forward);
    normalize3(forward);

    float up[3] = {camera_.up[0], camera_.up[1], camera_.up[2]};
    normalize3(up);

    float side[3];
    cross(forward, up, side);
    normalize3(side);

    float trueUp[3];
    cross(side, forward, trueUp);

    // World To Camera (View Matrix)
    auto &view = uniforms.worldToCamera;
    view[0] = side[0];
    view[1] = trueUp[0];
    view[2] = -forward[0];
    view[3] = 0.0f;

    view[4] = side[1];
    view[5] = trueUp[1];
    view[6] = -forward[1];
    view[7] = 0.0f;

    view[8] = side[2];
    view[9] = trueUp[2];
    view[10] = -forward[2];
    view[11] = 0.0f;

    view[12] = -(side[0] * effectivePos[0] + side[1] * effectivePos[1] + side[2] * effectivePos[2]);
    view[13] = -(trueUp[0] * effectivePos[0] + trueUp[1] * effectivePos[1] + trueUp[2] * effectivePos[2]);
    view[14] = forward[0] * effectivePos[0] + forward[1] * effectivePos[1] + forward[2] * effectivePos[2];
    view[15] = 1.0f;

    // Camera To World (Inverse View Matrix)
    auto &invView = uniforms.cameraToWorld;
    invView[0] = side[0];
    invView[1] = side[1];
    invView[2] = side[2];
    invView[3] = 0.0f;

    invView[4] = trueUp[0];
    invView[5] = trueUp[1];
    invView[6] = trueUp[2];
    invView[7] = 0.0f;

    invView[8] = -forward[0];
    invView[9] = -forward[1];
    invView[10] = -forward[2];
    invView[11] = 0.0f;

    invView[12] = effectivePos[0];
    invView[13] = effectivePos[1];
    invView[14] = effectivePos[2];
    invView[15] = 1.0f;

    // Projection Matrix
    const float aspect = config_.height == 0 ? 1.0f : static_cast<float>(config_.width) / static_cast<float>(config_.height);
    const float fovRad = camera_.fovY * (kPi / 180.0f);
    const float f = 1.0f / std::tan(fovRad * 0.5f);
    const float nearZ = camera_.nearZ;
    const float farZ = camera_.farZ;

    auto &proj = uniforms.projection;
    proj[0] = f / aspect;
    proj[1] = 0.0f;
    proj[2] = 0.0f;
    proj[3] = 0.0f;

    proj[4] = 0.0f;
    proj[5] = f;
    proj[6] = 0.0f;
    proj[7] = 0.0f;

    proj[8] = 0.0f;
    proj[9] = 0.0f;
    proj[10] = (farZ + nearZ) / (nearZ - farZ);
    proj[11] = -1.0f;

    proj[12] = 0.0f;
    proj[13] = 0.0f;
    proj[14] = (2.0f * farZ * nearZ) / (nearZ - farZ);
    proj[15] = 0.0f;

    uniforms.cameraPosFov = {effectivePos[0], effectivePos[1], effectivePos[2], fovRad};
    uniforms.resolutionExposure = {static_cast<float>(config_.width), static_cast<float>(config_.height), lighting_.exposure, 0.0f};
}

Renderer *npbrCreateRenderer(const RendererConfig &config) { return new Renderer(config); }

void npbrDestroyRenderer(Renderer *renderer) { delete renderer; }

void npbrSetCamera(Renderer *renderer, const CameraParameters &camera) {
    if (renderer) {
        renderer->setCamera(camera);
    }
}

void npbrSetLighting(Renderer *renderer, const LightingControls &lighting) {
    if (renderer) {
        renderer->setLighting(lighting);
    }
}

void npbrSetPreview(Renderer *renderer, const PreviewControls &preview) {
    if (renderer) {
        renderer->setPreviewControls(preview);
    }
}

void npbrSetEnvironment(Renderer *renderer, const EnvironmentControls &environment) {
    if (renderer) {
        renderer->setEnvironment(environment);
    }
}

void npbrSetOutputSize(Renderer *renderer, uint32_t width, uint32_t height) {
    if (renderer) {
        renderer->setOutputSize(width, height);
    }
}

void npbrUpsertMaterial(Renderer *renderer, uint32_t materialId, const std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> &bindings) {
    if (renderer) {
        renderer->upsertMaterial(materialId, bindings);
    }
}

void npbrUpdateMaterialTexture(Renderer *renderer, uint32_t materialId, MaterialSlot slot, const TextureBinding &binding) {
    if (renderer) {
        renderer->updateMaterialTexture(materialId, slot, binding);
    }
}

bool npbrBuildFrame(Renderer *renderer, uint32_t materialId, FrameDescriptor &outDescriptor) {
    return renderer && renderer->buildFrameDescriptor(materialId, outDescriptor);
}

} // namespace neuropbr
