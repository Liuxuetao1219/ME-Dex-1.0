from dataclasses import dataclass


@dataclass(frozen=True)
class TactileEncoderConfig:
    max_regions: int = 32
    input_channels: int = 6
    latent_dim: int = 256
    tokens_per_region: int = 4
    anatomy_tokens: int = 12
    anatomy_layers: int = 2
    anatomy_residual_scale_init: float = 0.1
    attention_heads: int = 8
    ffn_dim: int = 1024
    hand_side_types: int = 3
    surface_type_types: int = 22
