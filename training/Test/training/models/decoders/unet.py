import pytest
import torch
from training.models.decoders.unet import UNetDecoder, UNetDecoderHeads

def test_decoder_shapes_unet_encoder():
    from training.models.encoders.unet import UNetEncoder

    encoder = UNetEncoder(in_channels=3)
    decoder = UNetDecoder(
        in_channel=2048,
        skip_channels=[2048, 1024, 512, 256, 128, 64],  # Reversed from encoder
        out_channel=8
    )

    x = torch.randn(1, 3, 1024, 1024)
    x, skips = encoder(x)
    output = decoder(x, skips)

    # Should output 2048×2048 with 8 channels
    assert output.shape == torch.Size([1, 8, 2048, 2048])


def test_decoder_shapes_resnet_stride1():
    from training.models.encoders.unet import UNetResNetEncoder

    encoder = UNetResNetEncoder(in_channels=3, stride=1)
    decoder = UNetDecoder(
        in_channel=2048,
        skip_channels=[1024, 512, 256, 64],
        out_channel=8
    )

    x = torch.randn(1, 3, 1024, 1024)
    x, skips = encoder(x)
    output = decoder(x, skips)

    assert output.shape == torch.Size([1, 8, 2048, 2048])


def test_decoder_shapes_resnet_stride2():
    from training.models.encoders.unet import UNetResNetEncoder

    encoder = UNetResNetEncoder(in_channels=3, stride=2)
    decoder = UNetDecoder(
        in_channel=2048,
        skip_channels=[1024, 512, 256, 64],
        out_channel=8,
        sr_scale=4
    )

    x = torch.randn(1, 3, 1024, 1024)
    x, skips = encoder(x)
    output = decoder(x, skips)

    assert output.shape == torch.Size([1, 8, 2048, 2048])


def test_decoder_heads_output():
    from training.models.encoders.unet import UNetEncoder

    encoder = UNetEncoder(in_channels=3)
    decoder = UNetDecoderHeads(
        in_channel=2048,
        skip_channels=[2048, 1024, 512, 256, 128, 64],
        out_channels=[3, 1, 1, 3]
    )

    x = torch.randn(1, 3, 1024, 1024)
    x, skips = encoder(x)
    outputs = decoder(x, skips)

    assert len(outputs) == 4
    assert outputs[0].shape == torch.Size([1, 3, 2048, 2048])
    assert outputs[1].shape == torch.Size([1, 1, 2048, 2048])
    assert outputs[2].shape == torch.Size([1, 1, 2048, 2048])
    assert outputs[3].shape == torch.Size([1, 3, 2048, 2048])

if __name__ == "__main__":
    pytest.main([__file__])