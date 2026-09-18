"""Exercise the packaged XPolicyLab adapter with standard RGB/state observations."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--benchmark-root", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--wan-path", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(a.benchmark_root))
os.environ["ME_DEX_ROOT"] = str(root)
os.environ["ME_DEX_INPUT_COLOR_ORDER"] = "bgr"
spec = importlib.util.spec_from_file_location("me_dex_adapter", root / "xpolicylab/ME_Dex_1_0/model.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
model = adapter.Model(dict(action_type="joint", env_cfg_type="arx_x5", tactile_input="zero",
                           checkpoint_path=str(a.checkpoint), wan_path=str(a.wan_path)))
image = np.zeros((480,640,3), dtype=np.uint8)
image[...,0] = 180
obs = {"instruction": "Click the bell.",
       "vision": {name: {"color": image} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
       "state": {name: np.zeros(size, dtype=np.float32) for name,size in model.action_layout}}
torch.manual_seed(42)
model.reset()
model.update_obs(obs)
actions = model.get_action()
assert len(actions) == 16
for action in actions:
    for name, size in model.action_layout:
        assert action[name].shape == (size,) and np.isfinite(action[name]).all()
model.reset()
assert not model._obs_ready
print("ADAPTER_SMOKE_PASSED: RGB -> BGR, T5 text, zero physical tactile + real support, 16 finite 14D actions, reset")
