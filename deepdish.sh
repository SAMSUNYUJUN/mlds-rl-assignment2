#!/usr/bin/env bash
# Run the full experiment on one CUDA device from the repository root.
set -euo pipefail
cd "$(dirname "$0")"

python_bin=${A2_PYTHON:-.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${A2_GPU:-0}
if [[ ! $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]; then
    echo 'A2_GPU must be a single GPU index' >&2
    exit 2
fi
"$python_bin" -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'

"$python_bin" -u run.py train cartpole --out runs/cartpole --device cuda \
    --seed 137 --episodes 500 --n-step 3 --epsilon-steps 12000 --normalize-cartpole \
    --validation-episodes 80 --validation-every 20 --threads 1
"$python_bin" -u run.py evaluate --run runs/cartpole --device cuda --seed 240000
"$python_bin" -u validate.py --run runs/cartpole --device cuda

"$python_bin" -u run.py train mspacman --out runs/mspacman --device cuda
"$python_bin" -u run.py evaluate --run runs/mspacman --device cuda
"$python_bin" -u validate.py --run runs/mspacman --device cuda
