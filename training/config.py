from dataclasses import dataclass

@dataclass(frozen=True)
class PretrainedResnetTransform:
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]

@dataclass(frozen=True)
class DefaultTransform:
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5]

@dataclass(frozen=True)
class TransformConfig:
    Resnet: PretrainedResnetTransform = PretrainedResnetTransform()
    Default: DefaultTransform = DefaultTransform()

@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = None
    freeze_backbone: bool = None
    freeze_bn: bool = None

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = None
    batch_size: int = None
    epochs: int = None

@dataclass(frozen=True)
class Config:
    encoder: EncoderConfig
    training: TrainingConfig
    TransformConfig: TransformConfig

config = Config(
    encoder=EncoderConfig(),
    training=TrainingConfig(),
    TransformConfig=TransformConfig(),
)