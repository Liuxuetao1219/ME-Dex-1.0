# Release validation — 2026-09-18

Target: new V3 **AE**, WTAM Full Attention, Clean-only, both checkpoints at 50k.
Reference: the actual `WTAM_NewV3_Full_Clean50` evaluated implementation using
training revision `a463123894ba3d520bdd9179edb8f03666842a4d`.

## Completed checks

| Check | Result |
|---|---|
| Python compilation / Git whitespace checks | Passed |
| Input-contract unit tests | 3 passed |
| Exported model strict load | Passed, 3,347 state items |
| PPU reference vs release, 38 tensor checks | Exact equality; max absolute error 0 |
| NVIDIA 5880 reference vs release, 38 tensor checks | Exact equality; max absolute error 0 |
| XPolicyLab adapter direct smoke on 5880 | Passed: T5, cameras, force support, finite action chunk, reset |

The 38 comparisons comprise three AE inputs (zero, nonzero, below-deadband),
their three validity masks, V/A/T velocities at each of 10 denoising steps, and
the final action and video outputs. Each platform compares its own reference
against its own release output; this is not a claim of cross-platform bitwise
identity. The sequence uses a fixed synthetic camera/state/text-embedding input
and fixed RNG seeds, not all possible observations or a full closed-loop rerun.

The adapter smoke additionally uses actual T5 encoding for a text instruction,
standard RGB/state observation dictionaries, physical-zero observed force with
the support mask, and checks all 16 returned 14-dimensional action vectors.
It does not exercise simulator dynamics or establish a fresh success rate.

## Numerical migration detail

The old minimal runtime used in-place updates of BF16 initial latents. The
evaluated new implementation uses out-of-place updates that retain FP32 velocity
precision. The new release follows the evaluated implementation exactly; retaining
the old in-place updates would not preserve the evaluated numerical trajectory.

## Release safety

- No original training weights or datasets are deleted or modified.
- Old exported weights are preserved locally, with sizes checked against the
  prior Hub manifest and both torch archives successfully opened. This check
  is not a byte-for-byte comparison against a newly downloaded Hub payload.
- New exports contain model weights, complete AE EMA weights, and inference
  configuration only; no optimizer state or private dataset paths are needed.
- Hub upload replaces the current release in a single commit; old commit history
  is not purged. The training-code version is retained on a legacy Git branch.
- GitHub/HF visibility is unchanged; repository rename does not make a private
  repository public.
