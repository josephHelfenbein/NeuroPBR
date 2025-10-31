import pytest
import torch

from training.models.encoders.unet import UNetEncoder
from training.models.encoders.unet import UNetStrideEncoder
from training.models.encoders.unet import UNetResNetEncoder

def test_no_skips():
    encoder1 = UNetEncoder(in_channels=3, skip=False)
    encoder2 = UNetResNetEncoder(in_channels=3, skip=False)
    encoder3 = UNetStrideEncoder(in_channels=3, skip=False)

    x = torch.randn(1, 3, 1024, 1024)

    feature1, skips1 = encoder1(x)
    feature2, skips2 = encoder2(x)
    feature3, skips3 = encoder3(x)

    assert skips1 == skips2 == skips3 is None


def test_encoder_shapes_1():
    """Test encoder output shapes"""
    encoder = UNetEncoder(in_channels=3, channel_list=[64, 128, 256, 512, 1024, 2048])

    x = torch.randn(1, 3, 1024, 1024) # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 1024, 1024])
    assert skips[1].shape == torch.Size([1, 128, 1024, 1024])
    assert skips[2].shape == torch.Size([1, 256, 512, 512])
    assert skips[3].shape == torch.Size([1, 512, 256, 256])
    assert skips[4].shape == torch.Size([1, 1024, 128, 128])
    assert skips[5].shape == torch.Size([1, 2048, 64, 64])
    assert feature.shape == torch.Size([1, 2048, 32, 32])

def test_encoder_shapes_2():
    """Test encoder output shapes"""
    encoder = UNetStrideEncoder(in_channels=3, channel_list=[64, 128, 256, 512, 1024, 2048])

    x = torch.randn(1, 3, 1024, 1024)  # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 1024, 1024])
    assert skips[1].shape == torch.Size([1, 128, 1024, 1024])
    assert skips[2].shape == torch.Size([1, 256, 512, 512])
    assert skips[3].shape == torch.Size([1, 512, 256, 256])
    assert skips[4].shape == torch.Size([1, 1024, 128, 128])
    assert skips[5].shape == torch.Size([1, 2048, 64, 64])
    assert feature.shape == torch.Size([1, 2048, 32, 32])

def test_encoder_shapes_3():
    """Test encoder output shapes"""
    encoder = UNetResNetEncoder(in_channels=3, stride=1)

    x = torch.randn(1, 3, 1024, 1024)  # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 1024, 1024])
    assert skips[1].shape == torch.Size([1, 256, 512, 512])
    assert skips[2].shape == torch.Size([1, 512, 256, 256])
    assert skips[3].shape == torch.Size([1, 1024, 128, 128])
    assert feature.shape == torch.Size([1, 2048, 64, 64])

def test_encoder_shapes_4():
    """Test encoder output shapes"""
    encoder = UNetResNetEncoder(in_channels=3, stride=2)

    x = torch.randn(1, 3, 1024, 1024)  # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 512, 512])
    assert skips[1].shape == torch.Size([1, 256, 256, 256])
    assert skips[2].shape == torch.Size([1, 512, 128, 128])
    assert skips[3].shape == torch.Size([1, 1024, 64, 64])
    assert feature.shape == torch.Size([1, 2048, 32, 32])

def test_encoder_shapes_5():
    """Test encoder output shapes"""
    encoder = UNetResNetEncoder(in_channels=9, stride=1)

    x = torch.randn(1, 9, 1024, 1024)  # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 1024, 1024])
    assert skips[1].shape == torch.Size([1, 256, 512, 512])
    assert skips[2].shape == torch.Size([1, 512, 256, 256])
    assert skips[3].shape == torch.Size([1, 1024, 128, 128])
    assert feature.shape == torch.Size([1, 2048, 64, 64])

def test_encoder_shapes_6():
    """Test encoder output shapes"""
    encoder = UNetResNetEncoder(in_channels=9, stride=2)

    x = torch.randn(1, 9, 1024, 1024)  # (batch, channels, height, width)

    # Forward pass
    feature, skips = encoder(x)

    assert skips[0].shape == torch.Size([1, 64, 512, 512])
    assert skips[1].shape == torch.Size([1, 256, 256, 256])
    assert skips[2].shape == torch.Size([1, 512, 128, 128])
    assert skips[3].shape == torch.Size([1, 1024, 64, 64])
    assert feature.shape == torch.Size([1, 2048, 32, 32])

def test_unet_encoder_gradient_flow_1():
    encoder = UNetEncoder(in_channels=9)
    encoder.train()

    x = torch.randn(1, 9, 1024, 1024, requires_grad=True)
    feature, skips = encoder(x)

    loss = feature.sum()
    loss.backward()

    # All encoder parameters should have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} missing gradients"

def test_unet_encoder_gradient_flow_2():
    encoder = UNetStrideEncoder(in_channels=3)
    encoder.train()

    x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
    feature, skips = encoder(x)

    loss = feature.sum()
    loss.backward()

    # All encoder parameters should have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} missing gradients"

def test_gradient_flow():
    """Test that gradients flow through the encoder properly"""
    encoder = UNetResNetEncoder(in_channels=3)
    encoder.train()

    x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
    feature, skips = encoder(x)

    # Create a dummy loss
    loss = feature.sum() + sum(skip.sum() for skip in skips)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"

    # Check that encoder parameters have gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"


def test_frozen_backbone_no_gradients():
    """Test that frozen backbone (layers 1 and 2) doesn't compute gradients"""
    encoder = UNetResNetEncoder(in_channels=3, freeze_backbone=True, freeze_bn=True)
    encoder.train()  # train mode, but layers 1&2 frozen

    x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
    feature, skips = encoder(x)

    loss = feature.sum()
    loss.backward()

    # Check that frozen layers don't have gradients
    frozen_indices = [0,1,2,3,4,5]  # layer1 and layer2

    for name, param in encoder.named_parameters():
        for idx in frozen_indices:
            if f'encoder_stack.{idx}.' in name:
                assert param.grad is None, f"Frozen parameter {name} should not have gradients"
                break

    # later layers shouldn't be frozen
    unfrozen_indices = [6, 7]
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            for idx in unfrozen_indices:
                if f'encoder_stack.{idx}.' in name:
                    assert param.grad is not None, f"Unfrozen parameter {name} should have gradients"
                    break

def test_frozen_vs_unfrozen_params():
    """Test that freezing reduces trainable parameters"""
    # Unfrozen encoder
    encoder_unfrozen = UNetResNetEncoder(in_channels=3, freeze_backbone=False, freeze_bn=False)
    trainable_unfrozen = sum(p.numel() for p in encoder_unfrozen.parameters() if p.requires_grad)
    total_unfrozen = sum(p.numel() for p in encoder_unfrozen.parameters())

    # Frozen backbone only
    encoder_frozen_backbone = UNetResNetEncoder(in_channels=3, freeze_backbone=True, freeze_bn=False)
    trainable_backbone = sum(p.numel() for p in encoder_frozen_backbone.parameters() if p.requires_grad)
    total_backbone = sum(p.numel() for p in encoder_frozen_backbone.parameters())

    # Frozen backbone + BN
    encoder_frozen_both = UNetResNetEncoder(in_channels=3, freeze_backbone=True, freeze_bn=True)
    trainable_both = sum(p.numel() for p in encoder_frozen_both.parameters() if p.requires_grad)
    total_both = sum(p.numel() for p in encoder_frozen_both.parameters())

    # Assertions
    assert total_unfrozen == total_backbone == total_both, "Total params should be the same"
    assert trainable_unfrozen > trainable_backbone, "Frozen backbone should have fewer trainable params"
    assert trainable_backbone > trainable_both, "Freezing BN should further reduce trainable params"
    assert trainable_both < total_both, "Some params should be frozen"


if __name__ == "__main__":
    pytest.main([__file__])