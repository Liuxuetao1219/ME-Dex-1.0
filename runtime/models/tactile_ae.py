"""Frozen tactile encoder: physical force -> [B,T,12,256]."""
from pathlib import Path
import torch
from .tactile_autoencoder import TactileEncoder, TactileEncoderConfig
from .tactile_autoencoder.metadata import gripper_metadata
from .tactile_autoencoder.normalizer import ForceNormalizer
from .tactile_autoencoder.normalization_config import ForceNormalizationConfig


class TactileAE:
    def __init__(self, *, checkpoint_path, device, dtype):
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config = state["config"]
        self.model = TactileEncoder(TactileEncoderConfig())
        weights = {key.removeprefix("core.encoder."): value
                   for key, value in state["ema"]["shadow"].items()
                   if key.startswith("core.encoder.")}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        robot = next(d for d in config["domains"] if d["name"] == "robotwin_clean")
        spec = config["normalizations"][robot["normalization"]]
        self.normalizer = ForceNormalizer(
            ForceNormalizationConfig(**spec["parameters"])).to(device=device)
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode_condition(self, observed_source, *, observed_support_source, **_):
        force = observed_source.to(device=self.device, dtype=torch.float32)
        support = observed_support_source.to(device=self.device, dtype=torch.bool)
        batch = force.shape[0]
        # Preserve the structural mask when the observed force is zero.
        normalized, _, _ = self.normalizer.normalize(force, support)
        native, packed = gripper_metadata(support[:, :, 0].flatten(0, 1))
        owner = torch.arange(batch, device=self.device).repeat_interleave(4)
        dest = owner * 32 + native
        dense = normalized.new_zeros(batch * 32, 1, 3, 10, 14).to(self.dtype)
        dense = dense.index_copy(0, dest, normalized.flatten(0, 1).to(self.dtype))
        dense = dense.reshape(batch, 32, 1, 3, 10, 14).transpose(1, 2).flatten(0, 1)
        metadata = {}
        for key, value in packed.items():
            filled = value.new_zeros(batch * 32, *value.shape[1:]).index_copy(0, dest, value)
            metadata[key] = filled.reshape(batch, 32, *value.shape[1:])
        encoded = self.model(dense, **metadata)
        return (encoded["frame_tokens"].reshape(batch, 1, 12, 256),
                encoded["anatomy_token_valid"].reshape(batch, 1, 12))
