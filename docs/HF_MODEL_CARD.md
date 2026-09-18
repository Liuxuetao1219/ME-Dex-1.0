---
license: apache-2.0
tags:
- robotics
- robotwin
- video-action-tactile
---

# ME-Dex-1.0 — RoboTwin Clean2Random

Inference checkpoint for **new-data V3 AE + WTAM V2.5 Full Attention,
Clean-only, 50k steps**. This replaces the legacy two-observation-frame release.
Use the matching [ME-Dex-1.0 eval-only runtime](https://github.com/Liuxuetao1219/ME-Dex-1.0/tree/codex/me-dex-1.0-newv3-eval-only-20260918).
Old runtime versions are incompatible with these weights.

## Files

- `model.pt`: final trained Video/Action/Tactile model state, no optimizer state.
- `tactile_ae.pt`: new V3 AE 50k EMA weights and inference normalization config.
- `model_config.json`: architecture and released inference settings.
- `manifest.json`: checkpoint identity and evaluation provenance.

Obtain WAN VAE, T5 encoder, tokenizer and config separately from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
Installation, adapter integration and launch commands are in the code README.

## Contract and result

- Clean50-only training, Full Attention across all 30 MoT layers; no dropout.
- One current tactile observation plus 16 future frames; 17×12=204 tokens,
  latent width 256, tactile expert width 512.
- Frozen AE: smooth vector deadband + signed-asinh, exactly once.
- Standard no-tactile benchmark: physical current force is zero; real structural
  support is retained; future tactile is generated normally, not zeroed.
- RGB observations converted once to BGR; 10 denoising steps; all shifts=1;
  predict16/execute16; raw14-D qpos.
- Evaluated source: Random **3406/5000 = 68.12%**, 50 tasks ×100 episodes,
  unseen instructions, seed0. This is the source-model result, not a fresh
  packaged-runtime benchmark run.

Training code revision: `a463123894ba3d520bdd9179edb8f03666842a4d` in
`Liuxuetao1219/world-tactile-action-model`.
The packaged runtime is validated against that evaluated implementation with
fixed inputs and noise. See the code repository validation report for scope.

## Limitations

This checkpoint is designed for RoboTwin Aloha-AgileX joint control, not certified
for real-world deployment. Open-loop parity does not establish safety or success
on another robot. Use trusted PyTorch weights only. Results depend on the stated
evaluation protocol and should not be attributed solely to tactile sensing:
the reported standard evaluation uses no real tactile observation.
