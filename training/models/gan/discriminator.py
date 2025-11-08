import torch
import torch.nn as nn

"""
Training:
--------------
- Learning rate: 0.0002 (standard)
- Beta1: 0.5 (for Adam optimizer)
- Loss: BCELoss
- Patch output represents real/fake probability per patch
"""

# Consider experimenting with the number of layers
# For 2048x2048 images, 3 layers may be too "local" (small receptive field)
# Adding layers increases the receptive field, capturing larger context
# Fewer layers focus more on fine local details
# try 4–5 layers; 6 layers might be the max we want to go too
class PatchGANDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_filters=64,
            n_layers=3,
            use_sigmoid=True  # Optional for different losses
    ):
        """
        Parameters:
        -----------
        input_channels : int
            Number of channels in input image (in_chan = Source(in_chan) + target (in_chan))
        n_filters : int
            Number of filters in the first convolutional layer
            Subsequent layers double this (up to 8x)
        n_layers : int
            Number of downsampling layers (typically 3-4)
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid

        self.model = self._build_model()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize network weights.
        Uses normal distribution with mean=0, std=0.02 for Conv layers.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def _build_model(self):
        sequence = []

        # First layer: Conv -> LeakyReLU (no normalization)
        sequence += [
            nn.Conv2d(self.in_channels, self.n_filters,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        # Intermediate layers: Conv -> Norm -> LeakyReLU
        multiplier = 1
        prev_multiplier = 1
        for n in range(1, self.n_layers):
            prev_multiplier = multiplier
            multiplier = min(2 ** n, 8)  # Cap at 16x base filters

            sequence += [
                nn.Conv2d(self.n_filters * prev_multiplier,
                          self.n_filters * multiplier,
                          kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(self.n_filters * multiplier),
                nn.LeakyReLU(0.2, True)
            ]

        # Penultimate layer: Conv -> Norm -> LeakyReLU (stride=1)
        prev_multiplier = multiplier
        multiplier = min(2 ** self.n_layers, 8)
        sequence += [
            nn.Conv2d(self.n_filters * prev_multiplier,
                      self.n_filters * multiplier,
                      kernel_size=4, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.n_filters * multiplier),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer: 1 channel prediction map
        sequence += [
            nn.Conv2d(self.n_filters * multiplier, 1,
                      kernel_size=4, stride=1, padding=1)
        ]

        # sigmoid activation
        if self.use_sigmoid:
            sequence += [nn.Sigmoid()]

        return nn.Sequential(*sequence)

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        return self.model(x)

    def get_output_size(self, input_size=2048):
        """Calculate output feature map size for given input size"""
        size = input_size
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                size = (size + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        return size