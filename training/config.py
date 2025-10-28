from dataclasses import dataclass

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

config = Config(
    encoder=EncoderConfig(),
    training=TrainingConfig()
)