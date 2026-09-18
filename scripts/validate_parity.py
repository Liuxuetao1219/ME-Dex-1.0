"""Run reference/release in separate processes, then compare saved tensors.

The optional reference paths are developer validation inputs, not runtime dependencies.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("WAN_DISABLE_FLASH_ATTN", "1")
import numpy as np
import torch


def main(a):
    if a.mode == "compare":
        ref = torch.load(a.reference_output, map_location="cpu", weights_only=False)
        out = torch.load(a.output, map_location="cpu", weights_only=False)
        errors = {}
        for key in ref:
            assert ref[key].shape == out[key].shape, key
            errors[key] = float((ref[key].float() - out[key].float()).abs().max())
            torch.testing.assert_close(ref[key], out[key], rtol=0, atol=0, msg=key)
        print(json.dumps({"exact_parity": True, "max_absolute_errors": errors}, indent=2))
        return
    # NumPy-2 training pickle / NumPy-1 evaluation compatibility.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        for child in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            sys.modules.setdefault("numpy._core." + child, __import__("numpy.core." + child, fromlist=[child]))
    if a.mode == "reference":
        sys.path[:0] = [str(a.source_root / "bak"), str(a.source_root)]
        from models.motus import Motus, MotusConfig
        spec = importlib.util.spec_from_file_location("reference_adapter", a.reference_adapter)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        saved = json.loads((a.checkpoint / "config.json").read_text())
        model = Motus(adapter.build_config(saved, str(a.wan_path), MotusConfig))
        report = model.load_checkpoint(str(a.checkpoint / "pytorch_model"), strict=True)
        assert report["global_steps"] == 50000
        support_mask = adapter.support_mask
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
        from policy import MEDexPolicy, support_mask
        from checkpoint import load_metadata
        policy = MEDexPolicy.__new__(MEDexPolicy)
        policy.device = torch.device("cuda")
        policy.wan_path = a.wan_path
        policy.metadata = load_metadata(a.checkpoint)
        policy.config = policy.metadata["config"]
        model = policy._build_model()
    model.eval()
    support = torch.from_numpy(support_mask()).to(model.device)[None, :, None, None]
    zero = torch.zeros((1,4,1,3,10,14), device=model.device)
    results = {}
    with torch.inference_mode():
        for name, force in (("zero", zero), ("contact", torch.arange(zero.numel(), device=model.device).reshape_as(zero).float() / 1000), ("deadband", torch.full_like(zero, 1e-5))):
            tokens, valid = model.tactile_codec.encode_condition(force, observed_support_source=support)
            results["codec_" + name] = tokens.cpu()
            results["valid_" + name] = valid.cpu()
        torch.manual_seed(618)
        frame = torch.rand((1,3,384,320), device=model.device)
        state = torch.linspace(-.2,.2,14,device=model.device)[None]
        language = [torch.randn((24,4096),device=model.device,dtype=model.dtype)]
        original = model._joint_video_action_tactile_velocity
        def traced(**kwargs):
            step = len([k for k in results if k.startswith("action_velocity_")])
            outputs = original(**kwargs)
            for name, value in zip(("video", "action", "tactile"), outputs):
                results[f"{name}_velocity_{step}"] = value.cpu()
            return outputs
        model._joint_video_action_tactile_velocity = traced
        torch.manual_seed(20260918)
        video, action = model.inference_step(
            first_frame=frame, state=state, language_embeddings=language,
            num_inference_steps=10, video_schedule_shift=1, action_schedule_shift=1,
            tactile_schedule_shift=1, tactile_observed_source=zero,
            tactile_observed_support_source=support,
            tactile_observed_frame_times=torch.tensor([[0.]],device=model.device,dtype=model.dtype),
            tactile_future_query_times=.06*torch.arange(3,49,3,device=model.device,dtype=model.dtype)[None])
        results["final_action"] = action.cpu()
        results["final_video"] = video.cpu()
        assert torch.isfinite(action).all() and action.shape == (1,16,14)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, a.output)
    print("VALIDATION_SAVED", a.mode, a.output, "tensors", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["reference", "release", "compare"])
    for name in ("checkpoint", "wan-path", "source-root", "reference-adapter", "reference-output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
