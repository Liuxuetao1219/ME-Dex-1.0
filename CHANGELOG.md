# New V3 eval-only migration — 2026-09-18

## Changed

- `runtime/models/me_x.py` → `me_dex.py`, MEX classes → MEDex classes.
- Brand/repository references: ME-X / MachEmbodied → ME-Dex-1.0.
- Frozen codec: old V3 anatomy checkpoint → new-data V3 AE EMA checkpoint.
- Encoder metadata: 32-region extended layout, pad identities, valid-taxel UVs
  and true structural support, matching the evaluated training codec.
- Normalizer: old robust linear scaling → checkpoint-defined smooth deadband
  plus signed-asinh. Physical force is normalized exactly once.
- Observations: 2 → 1; tactile sequence: 18×12 → 17×12. Future horizon stays 16.
- Denoising updates: match the reference out-of-place FP32 promotion. No new
  architecture, attention mask, H-bridge, or dropout has been introduced.
- Camera preprocessing: reuse the evaluated resize/mosaic/padding implementation.
- Export config: use the evaluated shift=1 rather than the old release shift=5.
- Add portable `ME_Dex_1_0` XPolicyLab adapter; no private absolute paths.
- Add export, parity validation, input-contract tests, and local backup tooling.

## Unchanged

- Trained WAN/Action/Tactile parameter values and their state-dict names.
- Full 30-layer joint attention; Action width 1024, Tactile width 512.
- Raw qpos actions, BGR checkpoint compatibility, prompt, 10 denoising steps,
  16-step predicted/executed chunk, and physical-zero tactile evaluation protocol.
- Shared XPolicyLab/RoboTwin code, datasets, training results and original weights.

The new release is not backward compatible with old 18-slice model weights.
Legacy code is retained separately; do not silently load old weights with this
runtime. Hub replacement is performed only after validation and local backup.
