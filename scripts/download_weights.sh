#!/bin/bash
set -euo pipefail
destination="${1:?Usage: bash scripts/download_weights.sh /path/to/checkpoints}"
hf download liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard \
  --local-dir "${destination}"
hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json \
  --local-dir "${destination}/wan"
