import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    """double conv"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # can switch norms to groupNorm or layerNorm if small epoch size ~1 or having stability issues

        # keeps output size the same, and doesn't recompute bias
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    """downscaling: conv then max pool"""
    def __init__(self, in_channel, out_channel, skip=False):
        super().__init__()
        self.conv = ConvBlock(in_channel, out_channel)
        self.pool = nn.MaxPool2d(2)
        self.skip = skip

    def forward(self, x):
        x = self.conv(x)

        if self.skip:
            skip = x
        else:
            skip = None

        x = self.pool(x)

        return x, skip

class StrideEncoderBlock(nn.Module):
    """downscaling: strided conv"""
    # may slightly improve quality
    def __init__(self, in_channel, out_channel, skip=False):
        super().__init__()
        # First conv: regular conv for feature extraction
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # Second conv: downsampling
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu= nn.ReLU(inplace=True)

        self.skip = skip

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.skip:
            skip = x
        else:
            skip = None

        x = self.conv2(x)
        x = self.bn2(x) # downsample
        x = self.relu(x)

        return x, skip

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, channel_list=None, skip=False):
        super().__init__()

        if channel_list is None:
            channel_list = [64, 128, 256, 512, 1024, 2048]

        self.conv = ConvBlock(in_channels, channel_list[0]) # 1024x1024

        self.encoders = nn.ModuleList()
        for in_ch, out_ch in zip(channel_list[:-1], channel_list[1:]):
            self.encoders.append(EncoderBlock(in_ch, out_ch, skip))

        # self.enc1 = EncoderBlock(64, 128, skip) # 512x512
        # self.enc2 = EncoderBlock(128, 256, skip) # 256x256
        # self.enc3 = EncoderBlock(256, 512, skip) # 128x128
        # self.enc4 = EncoderBlock(512, 1024, skip) # 64x64
        # self.enc5 = EncoderBlock(1024, 2048, skip) # 32x32

    def forward(self, x):
        x = self.conv(x)

        skips = []
        for enc in self.encoders:
            if enc.skip:
                x, skip = enc(x)
                skips.append(skip)

            else:
                x = enc(x)

        return x, skips

class UNetStrideEncoder(nn.Module):
    def __init__(self, in_channels, channel_list=None, skip=False):
        super().__init__()

        if channel_list is None:
            channel_list = [64, 128, 256, 512, 1024, 2048]

        self.conv = ConvBlock(in_channels, channel_list[0]) # 1024x1024

        self.encoders = nn.ModuleList()
        for in_ch, out_ch in zip(channel_list[:-1], channel_list[1:]):
            self.encoders.append(StrideEncoderBlock(in_ch, out_ch, skip))

    def forward(self, x):
        x = self.conv(x)

        skips = []
        for enc in self.encoders:
            if enc.skip:
                x, skip = enc(x)
                skips.append(skip)

            else:
                x = enc(x)

        return x, skips

# --> Unet encoder with resnet backbone
"""
backbone: 
'resnet18'
'resnet34'
'resnet50'
'resnet101'
'resnet152'
"""
class UNetResNetEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            backbone='resnet101',
            freeze_backbone=False,
            freeze_bn=False,
            skip=False
    ):
        super().__init__()

        backbone_fn = getattr(models, backbone)
        resnet= backbone_fn(pretrained=True)

        self.encoder_stack = list(resnet.children())[:-2]

        # Expand pretrained conv1 from 3 to in_channels
        if in_channels != 3:
            original_weight = self.encoder_stack[0].weight.data.clone()  # [64, 3, 7, 7]
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            new_weight = original_weight.repeat(1, in_channels // 3, 1, 1) # [64, in_channels, 7, 7]
            new_weight = new_weight / (in_channels / 3.0)  # Average (normalize weight for each image)
            new_conv.weight.data = new_weight

            self.encoder_stack[0] = new_conv

        self.encoder_stack = nn.Sequential(*self.encoder_stack)

        self.skip = skip

        if freeze_backbone:
            for name, module in self.encoder_stack[:6].named_modules():
                if isinstance(module, nn.Conv2d):
                    for param in module.parameters():
                        param.requires_grad = False

        if freeze_bn:
            for module in self.encoder_stack[:6].modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Freeze BN learnable parameters (gamma, beta)
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, x):
        skips = []

        x = self.encoder_stack[0:3](x)  # initial
        if self.skip:
            skips.append(x)

        x = self.encoder_stack[3:5](x)  # layer 1
        if self.skip:
            skips.append(x)

        x = self.encoder_stack[5:6](x)  # layer 2
        if self.skip:
            skips.append(x)

        x = self.encoder_stack[6:7](x)  # layer 3
        if self.skip:
            skips.append(x)

        x = self.encoder_stack[7:](x)  # layer 4

        return x, skips

    def get_backbone_params(self):
        return self.encoder_stack.parameters()


'''
testing?
encoder = Encoder()
# input image
x    = torch.randn(1, 3, 572, 572)
ftrs = encoder(x)
for ftr in ftrs: print(ftr.shape)

>> 
torch.Size([1, 64, 568, 568])
torch.Size([1, 128, 280, 280])
torch.Size([1, 256, 136, 136])
torch.Size([1, 512, 64, 64])
torch.Size([1, 1024, 28, 28])
'''