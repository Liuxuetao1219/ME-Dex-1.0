"""Export a trusted, completed training checkpoint without optimizer state."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from policy import _install_numpy_pickle_compat
import torch

PREFIX = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "


def export(source, ae_source, destination):
    saved = json.loads((source / "config.json").read_text())
    if saved["attention_topology"]["mode"] != "full_joint":
        raise ValueError("Only the evaluated Full Attention checkpoint is supported")
    tac = saved["model"]["tactile"]
    expert = dict(saved["model"]["tactile_expert"])
    if tac["codec_type"] != "v3_newdata" or expert.pop("token_packing") != "none":
        raise ValueError("Expected unpacked new V3 AE")
    if tuple(expert[k] for k in ("latent_slices", "queries_per_slice", "condition_slices", "latent_dim")) != (17, 12, 1, 256):
        raise ValueError("Unexpected tactile layout")
    state = torch.load(source / "pytorch_model/mp_rank_00_model_states.pt",
                       map_location="cpu", mmap=True, weights_only=False)
    ae = torch.load(ae_source, map_location="cpu", weights_only=False)
    if state["global_steps"] != 50000 or ae["step"] != 50000:
        raise ValueError("Both model and AE must be the selected 50k weights")
    if ae["architecture"] != "tactile_ae_v3_newdata":
        raise ValueError("Wrong AE checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    # Preserve names, dtypes and aliases; never cast or reshape trained weights.
    torch.save({"module": state["module"]}, destination / "model.pt")
    # Publish the complete AE EMA weights but only inference configuration.
    # Dataset paths and training-only configuration are not runtime assets.
    ae_config = ae["config"]
    robot = next(d for d in ae_config["domains"] if d["name"] == "robotwin_clean")
    ae_config = {"model": {"family": "v3", "variant": "ae"},
                 "sequence": {"window_frames": 16},
                 "domains": [{"name": "robotwin_clean", "normalization": robot["normalization"]}],
                 "normalizations": {robot["normalization"]: ae_config["normalizations"][robot["normalization"]]}}
    torch.save({"architecture": ae["architecture"], "config": ae_config,
                "ema": ae["ema"], "step": ae["step"]}, destination / "tactile_ae.pt")
    config = {
        "format_version": 3, "variant": "ME-Dex-1.0", "step": 50000,
        "architecture": "wam_va", "joint_attention_mode": "full_bidirectional",
        "common": saved["common"], "action_expert": saved["action_expert"],
        "model": {"wan": {"precision": saved["model"]["wan"]["precision"]},
                  "tactile": {"enabled": True, "codec_type": "v3_newdata", "checkpoint_path": "tactile_ae.pt"},
                  "tactile_expert": expert},
        "prompt": {"prefix": PREFIX},
        "inference": {"num_inference_timesteps": 10, "video_schedule_shift": 1.0,
                      "action_schedule_shift": 1.0, "tactile_schedule_shift": 1.0,
                      "input_color_order": "bgr", "predicted_actions": 16,
                      "executed_actions": 16, "tactile_input": "physical_zero_with_support"}}
    manifest = {"format_version": 3, "variant": "ME-Dex-1.0", "step": 50000,
                "release": "newv3-ae-clean-full-50k-20260918", "ae_step": 50000,
                "training_commit": "a463123894ba3d520bdd9179edb8f03666842a4d",
                "training_data": "RoboTwin Clean50 only", "state_key_count": len(state["module"]),
                "random_successes": 3406, "random_episodes": 5000,
                "files": ["model.pt", "tactile_ae.pt", "model_config.json"]}
    for name, data in (("model_config.json", config), ("manifest.json", manifest)):
        (destination / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--ae-source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.ae_source, a.destination)
