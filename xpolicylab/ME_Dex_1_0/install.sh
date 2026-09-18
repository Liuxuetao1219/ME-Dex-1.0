#!/bin/bash
set -euo pipefail
: "${ME_DEX_ROOT:?Set ME_DEX_ROOT to your checkout}"
python -m pip install -r "${ME_DEX_ROOT}/runtime/requirements.txt"
# Install XPolicyLab's own transport/client dependencies using its instructions.
