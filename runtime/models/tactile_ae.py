"""Frozen new-data V3 AE encoder: physical force -> [B,T,12,256]."""
from pathlib import Path
import torch
from .tactile_autoencoder import AnatomyEncoderV3, AnatomyTactileAEV3Config
from .tactile_autoencoder.metadata import packed_metadata
from .tactile_autoencoder.normalizer import ForceNormalizerV41
from .tactile_autoencoder.normalization_config import ForceNormalizationV41Config


class TactileAE:
    def __init__(self, *, checkpoint_path, device, dtype):
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config = state["config"]
        if state["architecture"] != "tactile_ae_v3_newdata":
            raise ValueError("ME-Dex-1.0 requires the new V3 AE, not VAE or legacy V3")
        if config["model"]["variant"] != "ae" or config["sequence"]["window_frames"] != 16:
            raise ValueError("Expected the chunk16 AE checkpoint")
        self.model = AnatomyEncoderV3(AnatomyTactileAEV3Config(
            max_regions=32, surface_type_types=22, extended_pad_layout=True))
        weights = {key.removeprefix("core.encoder."): value
                   for key, value in state["ema"]["shadow"].items()
                   if key.startswith("core.encoder.")}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        robot = next(d for d in config["domains"] if d["name"] == "robotwin_clean")
        spec = config["normalizations"][robot["normalization"]]
        if spec["type"] != "positive_normal_si":
            raise ValueError("Expected RoboTwin positive-normal physical forces")
        self.normalizer = ForceNormalizerV41(
            ForceNormalizationV41Config(**spec["parameters"])).to(device=device)
        self.slots = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7],
                                   [17, 18, 19, 20], [21, 22, 23, 24]], device=device)
        self.device, self.dtype = device, dtype

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Tactile AE must stay frozen and eval-only")

    @torch.no_grad()
    def encode_condition(self, observed_source, *, observed_support_source, **_):
        self.assert_frozen()
        force = observed_source.to(device=self.device, dtype=torch.float32)
        support = observed_support_source.to(device=self.device, dtype=torch.bool)
        batch = force.shape[0]
        if force.shape[1:] != (4, 1, 3, 10, 14) or support.shape != (batch, 4, 1, 1, 10, 14):
            raise ValueError("Expected current-frame force[B,4,1,3,10,14] and structural support")
        # Exact training preprocessing. Physical zeros do NOT mean zero latent.
        normalized, _, _ = self.normalizer.normalize(force, support)
        native, packed = packed_metadata(support[:, :, 0].flatten(0, 1), self.slots.repeat(batch, 1))
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
