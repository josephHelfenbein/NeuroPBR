"""

Data format:
  - albedo: (B, 3, H, W) RGB float linear
  - roughness: (B, 1, H, W) float [0, 1]
  - metallic: (B, 1, H, W) float [0, 1]
  - normal: (B, 3, H, W) normalized XYZ

Ground truth includes:
  - PBR maps (albedo, roughness, metallic, normal)
  - Rendered RGB images from external renderer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class WeightedL1Loss(nn.Module):
    """Weighted L1 loss across multiple prediction targets."""
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        loss_dict = {}
        
        for name, weight in self.weights.items():
            if weight > 0 and name in pred and name in target:
                l1 = F.l1_loss(pred[name], target[name])
                total_loss += weight * l1
                loss_dict[f"l1_{name}"] = l1.item()
        
        return total_loss, loss_dict


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss. Computes 1 - SSIM with 11x11 Gaussian window."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.window = None

    def _create_window(self, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()
        
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window_2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window.to(dtype=dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        _, channel, _, _ = pred.shape
        dtype = pred.dtype
        
        if (self.window is None or self.window.shape[0] != channel or
                self.window.device != pred.device or self.window.dtype != dtype):
            self.window = self._create_window(channel, pred.device, dtype)
        
        if target.dtype != dtype:
            target = target.to(dtype=dtype)
        
        C1 = torch.tensor((0.01) ** 2, device=pred.device, dtype=dtype)
        C2 = torch.tensor((0.03) ** 2, device=pred.device, dtype=dtype)
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
        
        return 1.0 - ssim_map.mean()


class NormalConsistencyLoss(nn.Module):
    """Angular error loss for normal maps. Computes 1 - cos(angle)."""
    def __init__(self, normalize_pred: bool = True):
        super().__init__()
        self.normalize_pred = normalize_pred

    def forward(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, float]:
        if self.normalize_pred:
            pred = F.normalize(pred, p=2, dim=1, eps=eps)
        
        target = F.normalize(target, p=2, dim=1, eps=eps)
        
        cos_sim = (pred * target).sum(dim=1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
        
        angle_rad = torch.acos(cos_sim)
        angle_deg = (angle_rad * 180.0 / np.pi).mean().item()
        
        loss = (1.0 - cos_sim).mean()
        
        return loss, angle_deg


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, loss_type: str = "hinge") -> torch.Tensor:
    if loss_type == "hinge":
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss
    elif loss_type == "bce":
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real_loss + fake_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def generator_gan_loss(fake_logits: torch.Tensor, loss_type: str = "hinge") -> torch.Tensor:
    if loss_type == "hinge":
        return -fake_logits.mean()
    elif loss_type == "bce":
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator. 70x70 receptive field, outputs (B, 1, H/16, W/16) logits."""
    def __init__(self, in_channels: int = 8, ndf: int = 64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class HybridLoss(nn.Module):
    """Unified loss function for PBR reconstruction combining L1, SSIM, normal consistency, and GAN losses."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        l1_weights = {
            "albedo": config.get("w_albedo", 1.0),
            "roughness": config.get("w_roughness", 1.0),
            "metallic": config.get("w_metallic", 1.0),
            "normal": config.get("w_normal_map", 1.0)
        }
        self.l1_loss = WeightedL1Loss(l1_weights)
        self.ssim_loss = SSIMLoss()
        self.normal_loss = NormalConsistencyLoss()
        
        self.gan_loss_type = config.get("gan_loss_type", "hinge")

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
                discriminator: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict]:
        info = {}
        total_loss = 0.0
        
        w_l1 = self.config.get("w_l1", 1.0)
        if w_l1 > 0:
            l1_loss, l1_dict = self.l1_loss(pred, target)
            total_loss += w_l1 * l1_loss
            info.update(l1_dict)
            info["loss_l1_total"] = l1_loss.item()
        
        w_ssim = self.config.get("w_ssim", 0.3)
        if w_ssim > 0 and "albedo" in pred and "albedo" in target:
            ssim_loss = self.ssim_loss(pred["albedo"], target["albedo"])
            total_loss += w_ssim * ssim_loss
            info["loss_ssim"] = ssim_loss.item()
        
        w_normal = self.config.get("w_normal", 0.5)
        if w_normal > 0 and "normal" in pred and "normal" in target:
            normal_loss, angle_deg = self.normal_loss(pred["normal"], target["normal"])
            total_loss += w_normal * normal_loss
            info["loss_normal"] = normal_loss.item()
            info["normal_angle_deg"] = angle_deg
        
        w_gan = self.config.get("w_gan", 0.0)
        if w_gan > 0 and discriminator is not None:
            pred_concat = torch.cat([
                pred["albedo"],
                pred["roughness"],
                pred["metallic"],
                pred["normal"]
            ], dim=1)
            
            fake_logits = discriminator(pred_concat)
            gan_loss = generator_gan_loss(fake_logits, self.gan_loss_type)
            total_loss += w_gan * gan_loss
            info["loss_gan_g"] = gan_loss.item()
        
        if "albedo" in pred and "albedo" in target:
            mae = F.l1_loss(pred["albedo"], target["albedo"]).item()
            rmse = torch.sqrt(F.mse_loss(pred["albedo"], target["albedo"])).item()
            info["mae_albedo"] = mae
            info["rmse_albedo"] = rmse
        
        info["loss_total"] = total_loss.item()
        
        return total_loss, info

def run_tests():
    print("=" * 80)
    print("Running NeuroPBR Loss Function Unit Tests")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Test 1: WeightedL1Loss")
    weights = {"albedo": 1.0, "roughness": 0.5}
    l1_loss = WeightedL1Loss(weights).to(device)
    
    pred = {
        "albedo": torch.rand(2, 3, 64, 64, device=device),
        "roughness": torch.rand(2, 1, 64, 64, device=device)
    }
    target = {
        "albedo": torch.rand(2, 3, 64, 64, device=device),
        "roughness": torch.rand(2, 1, 64, 64, device=device)
    }
    
    loss, loss_dict = l1_loss(pred, target)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Individual losses: {loss_dict}")
    assert torch.isfinite(loss), "L1 loss is not finite!"
    assert loss > 0, "L1 loss should be positive!"
    print("  ✓ Passed\n")
    
    print("Test 2: SSIMLoss (identical inputs)")
    ssim_loss = SSIMLoss().to(device)
    img = torch.rand(2, 3, 128, 128, device=device)
    loss = ssim_loss(img, img)
    print(f"  Loss (identical): {loss.item():.6f}")
    assert loss < 0.01, f"SSIM loss should be near zero for identical inputs! Got {loss.item()}"
    print("  ✓ Passed\n")
    
    print("Test 3: SSIMLoss (different inputs)")
    img1 = torch.rand(2, 3, 128, 128, device=device)
    img2 = torch.rand(2, 3, 128, 128, device=device)
    loss = ssim_loss(img1, img2)
    print(f"  Loss (different): {loss.item():.6f}")
    assert loss > 0, "SSIM loss should be positive for different inputs!"
    print("  ✓ Passed\n")
    
    print("Test 4: NormalConsistencyLoss (identical normals)")
    normal_loss = NormalConsistencyLoss().to(device)
    normals = F.normalize(torch.rand(2, 3, 64, 64, device=device), p=2, dim=1)
    loss, angle_deg = normal_loss(normals, normals)
    print(f"  Loss (identical): {loss.item():.6f}, Angle: {angle_deg:.2f}°")
    assert loss < 0.01, f"Normal loss should be near zero for identical normals! Got {loss.item()}"
    assert angle_deg < 1.0, f"Angle should be near zero! Got {angle_deg}°"
    print("  ✓ Passed\n")
    
    print("Test 5: NormalConsistencyLoss (opposite normals)")
    normals1 = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                              [[0.0, 0.0], [0.0, 0.0]],
                              [[0.0, 0.0], [0.0, 0.0]]]], device=device)
    normals2 = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]],
                              [[0.0, 0.0], [0.0, 0.0]],
                              [[0.0, 0.0], [0.0, 0.0]]]], device=device)
    loss, angle_deg = normal_loss(normals1, normals2)
    print(f"  Loss (opposite): {loss.item():.6f}, Angle: {angle_deg:.2f}°")
    assert angle_deg > 170.0, f"Angle should be ~180° for opposite normals! Got {angle_deg}°"
    print("  ✓ Passed\n")
    
    print("Test 6: PatchGANDiscriminator")
    disc = PatchGANDiscriminator(in_channels=8, ndf=32).to(device)
    pbr_input = torch.rand(2, 8, 256, 256, device=device)
    logits = disc(pbr_input)
    print(f"  Input shape: {pbr_input.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape[1] == 1, "Discriminator should output 1 channel!"
    assert torch.isfinite(logits).all(), "Discriminator output contains NaN/Inf!"
    print("  ✓ Passed\n")
    
    print("Test 7: GAN losses")
    real_logits = torch.randn(2, 1, 16, 16, device=device)
    fake_logits = torch.randn(2, 1, 16, 16, device=device)
    
    d_loss_hinge = discriminator_loss(real_logits, fake_logits, "hinge")
    g_loss_hinge = generator_gan_loss(fake_logits, "hinge")
    
    d_loss_bce = discriminator_loss(real_logits, fake_logits, "bce")
    g_loss_bce = generator_gan_loss(fake_logits, "bce")
    
    print(f"  D loss (hinge): {d_loss_hinge.item():.6f}")
    print(f"  G loss (hinge): {g_loss_hinge.item():.6f}")
    print(f"  D loss (BCE): {d_loss_bce.item():.6f}")
    print(f"  G loss (BCE): {g_loss_bce.item():.6f}")
    
    assert torch.isfinite(d_loss_hinge), "D loss (hinge) is not finite!"
    assert torch.isfinite(g_loss_hinge), "G loss (hinge) is not finite!"
    assert torch.isfinite(d_loss_bce), "D loss (BCE) is not finite!"
    assert torch.isfinite(g_loss_bce), "G loss (BCE) is not finite!"
    print("  ✓ Passed\n")
    
    print("Test 8: HybridLoss weight sensitivity")
    
    config_with_ssim = {
        "w_l1": 1.0, "w_ssim": 0.5, "w_normal": 0.5, "w_gan": 0.0,
        "w_albedo": 1.0, "w_roughness": 1.0, "w_metallic": 1.0, "w_normal_map": 1.0
    }
    
    config_no_ssim = {
        "w_l1": 1.0, "w_ssim": 0.0, "w_normal": 0.5, "w_gan": 0.0,
        "w_albedo": 1.0, "w_roughness": 1.0, "w_metallic": 1.0, "w_normal_map": 1.0
    }
    
    hybrid_loss_with = HybridLoss(config_with_ssim).to(device)
    hybrid_loss_without = HybridLoss(config_no_ssim).to(device)
    
    pred_dict = {
        "albedo": torch.rand(2, 3, 128, 128, device=device),
        "roughness": torch.rand(2, 1, 128, 128, device=device),
        "metallic": torch.rand(2, 1, 128, 128, device=device),
        "normal": F.normalize(torch.rand(2, 3, 128, 128, device=device), p=2, dim=1)
    }
    
    target_dict = {
        "albedo": torch.rand(2, 3, 128, 128, device=device),
        "roughness": torch.rand(2, 1, 128, 128, device=device),
        "metallic": torch.rand(2, 1, 128, 128, device=device),
        "normal": F.normalize(torch.rand(2, 3, 128, 128, device=device), p=2, dim=1)
    }
    
    loss_with, info_with = hybrid_loss_with(pred_dict, target_dict)
    loss_without, info_without = hybrid_loss_without(pred_dict, target_dict)
    
    print(f"  Loss with SSIM: {loss_with.item():.6f}")
    print(f"  Loss without SSIM: {loss_without.item():.6f}")
    
    assert "loss_ssim" in info_with, "SSIM loss should be in info when enabled!"
    assert "loss_ssim" not in info_without, "SSIM loss should not be in info when disabled!"
    print("  ✓ Passed\n")
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

def example_training_step():
    print("\n" + "=" * 80)
    print("Example Training Step")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    config = {
        "w_l1": 1.0,
        "w_ssim": 0.3,
        "w_normal": 0.5,
        "w_gan": 0.05,
        "w_albedo": 1.0,
        "w_roughness": 1.0,
        "w_metallic": 1.0,
        "w_normal_map": 1.0,
        "gan_loss_type": "hinge"
    }
    
    hybrid_loss = HybridLoss(config).to(device)
    discriminator = PatchGANDiscriminator(in_channels=8, ndf=64).to(device)
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    B, H, W = 2, 256, 256
    
    pred = {
        "albedo": torch.rand(B, 3, H, W, device=device),
        "roughness": torch.rand(B, 1, H, W, device=device),
        "metallic": torch.rand(B, 1, H, W, device=device),
        "normal": F.normalize(torch.randn(B, 3, H, W, device=device), p=2, dim=1)
    }
    
    target = {
        "albedo": torch.rand(B, 3, H, W, device=device),
        "roughness": torch.rand(B, 1, H, W, device=device),
        "metallic": torch.rand(B, 1, H, W, device=device),
        "normal": F.normalize(torch.randn(B, 3, H, W, device=device), p=2, dim=1)
    }
    
    print(f"Batch size: {B}, Resolution: {H}x{W}\n")
    
    print("Generator Training Step:")
    print("-" * 40)
    
    g_loss, g_info = hybrid_loss(pred, target, discriminator=discriminator)
    
    print(f"Total Loss: {g_info['loss_total']:.6f}")
    print("\nComponent Losses:")
    for key, val in g_info.items():
        if key.startswith("loss_"):
            print(f"  {key}: {val:.6f}")
    
    print("\nMetrics:")
    for key, val in g_info.items():
        if not key.startswith("loss_"):
            print(f"  {key}: {val:.6f}")
    
    print("\n" + "=" * 40)
    
    print("Discriminator Training Step:")
    print("-" * 40)
    
    pred_concat = torch.cat([pred["albedo"], pred["roughness"], pred["metallic"], pred["normal"]], dim=1)
    target_concat = torch.cat([target["albedo"], target["roughness"], target["metallic"], target["normal"]], dim=1)
    
    real_logits = discriminator(target_concat.detach())
    fake_logits = discriminator(pred_concat.detach())
    
    d_loss = discriminator_loss(real_logits, fake_logits, config["gan_loss_type"])
    
    print(f"Discriminator Loss: {d_loss.item():.6f}")
    print(f"Real logits mean: {real_logits.mean().item():.6f}")
    print(f"Fake logits mean: {fake_logits.mean().item():.6f}")
    
    print("\n" + "=" * 80)
    print("Example training step completed!")
    print("=" * 80)


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    run_tests()
    example_training_step()
    
    print("\n✓ All demonstrations completed successfully!")
