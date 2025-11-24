"""
Test script to verify the new 6-layer discriminator works with 1024x1024 images.
"""

import torch
from models.gan.discriminator import PatchGANDiscriminator

def test_discriminator():
    print("="*80)
    print("Testing 6-layer PatchGAN Discriminator for 1024x1024 Images")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create discriminator with 6 layers
    disc = PatchGANDiscriminator(
        in_channels=8,  # 3 (albedo) + 1 (roughness) + 1 (metallic) + 3 (normal)
        n_filters=64,
        n_layers=6,
        use_sigmoid=False  # For hinge loss
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in disc.parameters())
    print(f"\nDiscriminator Parameters: {num_params:,}")
    
    # Test with 1024x1024 input
    batch_size = 2
    input_size = 1024
    
    print(f"\nTesting with batch_size={batch_size}, input_size={input_size}x{input_size}")
    
    # Create fake input (concatenated PBR maps)
    fake_input = torch.randn(batch_size, 8, input_size, input_size, device=device)
    
    print(f"Input shape: {fake_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = disc(fake_input)
    
    print(f"Output shape: {output.shape}")
    
    # Calculate receptive field info
    output_size = disc.get_output_size(input_size)
    print(f"Output spatial size: {output_size}x{output_size}")
    
    # Calculate receptive field (approximate)
    # Each layer with stride=2 doubles the receptive field
    # kernel=4, stride=2 layers: 6 layers
    receptive_field = 4  # First layer
    for i in range(1, 6):  # 5 more stride=2 layers
        receptive_field = receptive_field * 2 + 4 - 2
    receptive_field = receptive_field * 2 + 4 - 2  # Penultimate stride=1
    receptive_field = receptive_field + 4 - 2  # Output stride=1
    
    print(f"Approximate receptive field: {receptive_field}x{receptive_field} pixels")
    print(f"Receptive field as % of image: {(receptive_field/input_size)*100:.1f}%")
    
    print("\n" + "="*80)
    print("✓ Discriminator test passed!")
    print("="*80)
    
    # Compare with simple discriminator
    print("\nComparison with simple 4-layer discriminator:")
    from losses.losses import PatchGANDiscriminator as SimpleDisc
    simple_disc = SimpleDisc(in_channels=8, ndf=64).to(device)
    simple_params = sum(p.numel() for p in simple_disc.parameters())
    
    with torch.no_grad():
        simple_output = simple_disc(fake_input)
    
    print(f"  Simple disc parameters: {simple_params:,}")
    print(f"  Simple disc output shape: {simple_output.shape}")
    print(f"  Configurable disc parameters: {num_params:,}")
    print(f"  Configurable disc output shape: {output.shape}")
    print(f"  Parameter increase: {((num_params/simple_params)-1)*100:.1f}%")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    test_discriminator()
