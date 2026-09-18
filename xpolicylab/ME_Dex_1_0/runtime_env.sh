#!/bin/bash
# Set these paths explicitly; no private shared-drive dependency.
: "${ME_DEX_ROOT:?Set ME_DEX_ROOT to your ME-Dex-1.0 checkout}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WAN_DISABLE_FLASH_ATTN=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ME_DEX_INPUT_COLOR_ORDER="${ME_DEX_INPUT_COLOR_ORDER:-bgr}"
