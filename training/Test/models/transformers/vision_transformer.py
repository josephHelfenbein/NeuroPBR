import torch
import pytest
from training.models.transformers.vision_transformer import ViT, ViTCrossViewFusion

def test_basic_forward_pass():
    """Test basic forward pass with default parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(dim=2048, num_heads=32, depth=4).to(device)
    x = torch.randn(2, 2048, 8, 8)

    output = model(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

def test_different_spatial_dimensions_2():
    """Test with different spatial dimensions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTCrossViewFusion(dim=2048, num_views=3, num_heads=32, depth=4).to(device)

    test_cases = [
        (2, 2048, 4, 4),
        (2, 2048, 8, 8),
        (2, 2048, 16, 16),
        (1, 2048, 7, 7),
    ]

    for shape in test_cases:
        latents = [torch.randn(*shape) for _ in range(3)]
        output = model(*latents)
        assert output.shape == shape, f"Failed for shape {shape}"

def test_different_spatial_dimensions():
    """Test with different spatial dimensions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(dim=2048, num_heads=32, depth=4).to(device)

    test_cases = [
        (2, 2048, 4, 4),
        (2, 2048, 8, 8),
        (2, 2048, 16, 16),
        (1, 2048, 7, 7),
    ]

    for shape in test_cases:
        x = torch.randn(*shape)
        output = model(x)
        assert output.shape == shape, f"Failed for shape {shape}"

def test_basic_forward_pass_three_views():
    """Test basic forward pass with 3 views"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTCrossViewFusion(dim=2048, num_views=3, num_heads=32, depth=4).to(device)

    view1 = torch.randn(2, 2048, 8, 8)
    view2 = torch.randn(2, 2048, 8, 8)
    view3 = torch.randn(2, 2048, 8, 8)

    output = model(view1, view2, view3)

    assert output.shape == view1.shape, f"Expected shape {view1.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

def test_cross_view_attention():
    """Test that views actually interact (not just processed separately)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTCrossViewFusion(dim=2048, num_views=2, num_heads=32, depth=4).to(device)
    model.eval()

    # Create two different views
    view1 = torch.randn(1, 2048, 4, 4)
    view2 = torch.randn(1, 2048, 4, 4)
    view2_different = torch.randn(1, 2048, 4, 4)

    with torch.no_grad():
        output1 = model(view1, view2)
        output2 = model(view1, view2_different)

    # Outputs should be different when second view changes
    assert not torch.allclose(output1, output2, atol=1e-5), \
        "Cross-view fusion doesn't seem to be working - outputs are identical"

if __name__ == "__main__":
    pytest.main([__file__])