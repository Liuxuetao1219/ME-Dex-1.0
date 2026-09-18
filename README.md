# ME-Dex-1.0

Eval-only Video–Action–Tactile policy for RoboTwin Clean50-to-Random evaluation.
This release uses the **new-data V3 AE, Full Attention, Clean-only 50k** model.
It is not the legacy V3 checkpoint, VAE, Uni Hand2, or H-bridge variant.

- Code: https://github.com/Liuxuetao1219/ME-Dex-1.0
- Weights: https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard
- External VAE / T5 / tokenizer: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

## Model and evaluation contract

All 30 layers use full bidirectional Video–Action–Tactile joint attention.
WAN and Action are initialized from Motus Stage2 during training; the released
weights are the final trained model, not the initialization checkpoint.
The frozen tactile encoder uses its 50k EMA weights and checkpoint-defined
smooth vector deadband + signed-asinh normalization, applied exactly once.

One observed tactile frame `t` and 16 future frames give `[B,17,12,256]` latents
(204 tokens). The tactile expert width is 512. All 12 anatomical slots are kept;
absent slots remain invalid/zero in the AE output. Structural support is distinct
from contact presence. The source surface order is preserved by explicit metadata.

The standard benchmark has no tactile measurements. We encode **physical zero
force with the real structural support mask**, not a manually zeroed latent.
The model still generates future tactile jointly with video and action.

| Setting | Released evaluation |
|---|---|
| Camera API | RGB, explicitly converted once to BGR for this checkpoint |
| Mosaic | Head 320×240, two wrists 160×120; 12-pixel top/bottom padding |
| State/action | Raw 14-D joint positions |
| Observation | Current tactile frame only, physical zero |
| Sampling | 10 flow-matching steps, V/A/T shift=1 |
| Horizon | Predict 16, execute 16 |
| Action offsets | t+3, t+6, …, t+48 |
| Training | Clean50 only, 50k; no state/route dropout |

The evaluated source implementation achieved **3406/5000 = 68.12% Random**
(50 tasks × 100 episodes, unseen instructions, seed 0). Packaging equivalence
is checked separately; this number is not a new rerun of the packaged release.
Do not compare different tactile-input, color, shift, or instruction protocols
as if they were controlled model-only ablations.

## Install and download

```bash
conda create -n me-dex-1 python=3.10 -y
conda activate me-dex-1
pip install -r runtime/requirements.txt
pip install huggingface_hub pytest
bash scripts/download_weights.sh /path/to/checkpoints/ME-Dex-1.0
```

The runtime is self-contained: no private training checkout or ManiFeel import
is required. The weights directory contains `model.pt`, `model_config.json`,
`manifest.json`, `tactile_ae.pt`, and downloaded `wan/` assets.
Only load trusted PyTorch checkpoint files.

## XPolicyLab adapter

Copy `xpolicylab/ME_Dex_1_0` to your benchmark's `XPolicyLab/policy/ME_Dex_1_0`.
Keep the benchmark's shared helpers and simulator unchanged. Install XPolicyLab
and RoboTwin using their official instructions, then:

```bash
export ME_DEX_ROOT=/path/to/ME-Dex-1.0
export ME_DEX_CHECKPOINT_PATH=/path/to/checkpoints/ME-Dex-1.0
export ME_DEX_WAN_PATH="$ME_DEX_CHECKPOINT_PATH/wan"
export ME_DEX_INPUT_COLOR_ORDER=bgr
export ROBOTWIN_TASK_CONFIG=demo_randomized
export ROBOTWIN_TEST_NUM=100
bash /path/to/RoboTwin/XPolicyLab/policy/ME_Dex_1_0/eval.sh \
  RoboTwin click_bell me_dex_1_0_newv3 arx_x5 joint 0 0 0 \
  /path/to/conda/envs/me-dex-1 /path/to/conda/envs/robotwin
```

The adapter uses `eval_batch=false`. BGR is an explicit checkpoint compatibility
setting, not a change to XPolicyLab's RGB API. The shell entry point enables SDPA
to match the evaluated implementation. Configure seen/unseen instructions using
the benchmark's standard Clean/Random settings.

## Validation and provenance

```bash
WAN_DISABLE_FLASH_ATTN=1 python -m pytest tests -q
```

`scripts/validate_parity.py` compares the original evaluated implementation with
this runtime using identical inputs and RNG seeds. It checks physical-zero,
nonzero, and deadband AE encodings; all 10 steps of V/A/T velocities; final action
and video tensors. `scripts/export_checkpoint.py` preserves trained tensor names,
dtypes and aliases while excluding optimizer state. No runtime checksum scan is
required.

The source training revision is `a463123894ba3d520bdd9179edb8f03666842a4d`
in `Liuxuetao1219/world-tactile-action-model`. This eval-only branch starts at
the previous public-runtime commit `894e53ab751e51985509716b9bcea0f7064ef200`.
Prior code and training support remain in Git history/the legacy branch.
See [CHANGELOG](CHANGELOG.md) for the migration scope.

## Third-party code and license

The minimal WAN runtime derives from [Wan2.2](https://github.com/Wan-Video/Wan2.2)
and retains its copyright headers. Initialization originates from
[Motus](https://github.com/motus-robotics/Motus). Apache-2.0.
