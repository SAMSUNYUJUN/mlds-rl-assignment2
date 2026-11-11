# Assignment 2 · Deep Q-Network

Deep Q-Network experiments for `CartPole-v1` and `ALE/MsPacman-v5`. Both agents use uniform replay, a target network, a dueling action-value network, and three-step Double DQN targets in the reported runs.

## Run on one DeepDish GPU

```bash
git clone https://github.com/SAMSUNYUJUN/mlds-rl-assignment2.git
cd mlds-rl-assignment2
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
A2_GPU=0 bash deepdish.sh
```

Set `A2_GPU` to the assigned GPU index. The script exposes that device alone, checks that PyTorch sees exactly one CUDA device, then trains and validates both agents. The reported run used the DeepDish installation of PyTorch 2.6.0+cu124. The `ale-py` package includes the MsPacman ROM. A full run creates `runs/cartpole/` and `runs/mspacman/`. Move an existing run to a different path before rerunning the script.

To run individual stages, use the corresponding command below. `--device auto` picks CUDA, Metal, or CPU based on availability. Replace it with `cuda` on DeepDish and set `CUDA_VISIBLE_DEVICES` to one index before invoking Python.

```bash
.venv/bin/python run.py train cartpole --out runs/cartpole --device auto \
  --seed 137 --episodes 500 --n-step 3 --epsilon-steps 12000 --normalize-cartpole \
  --validation-episodes 80 --validation-every 20 --threads 1
.venv/bin/python run.py evaluate --run runs/cartpole --device auto --seed 240000
.venv/bin/python validate.py --run runs/cartpole --device auto

.venv/bin/python run.py train mspacman --out runs/mspacman --device auto
.venv/bin/python run.py evaluate --run runs/mspacman --device auto
.venv/bin/python validate.py --run runs/mspacman --device auto
```

Each evaluation defaults to 500 greedy episodes, with fixed quotas of 20 episodes across each of 25 seeded environments. Training selects `best.pt` by greedy mean reward on seeds starting at 100000. Final evaluation starts at 240000 for CartPole and 200000 for MsPacman. Policy validation uses a distinct set starting at 300000. Training also saves `initial.pt`, `early.pt`, and `last.pt` for validation.

## Validation experiments

`validate.py` runs the following interventions with the saved checkpoints. The JSON summary and CSV files record the quantitative evidence for each one.

| Experiment | Procedure and interpretation |
| --- | --- |
| Paired greedy rollouts | Run random, untrained greedy, early, final and selected policies on the same 48 seeds. Compute paired mean differences and 95% percentile intervals from 10,000 bootstrap resamples. This tests policy improvement on unseen starts at fixed exploration. |
| CartPole first-action intervention | Center and stop the cart, sweep 25 pole angles and 25 pole angular velocities, and force left or right for the first step. Thereafter, both rollouts use the selected greedy controller. Compare their discounted returns to the sign of the network's action preference and record the regret of following it. |
| CartPole time cap and failure | Replay the same first three actions in a normal environment and a three-step cap. Confirm the shared successor and show how the target changes if the cap is treated as a physical failure. Force a threshold crossing from a near-failure state and verify the target uses the immediate reward. |
| MsPacman frame history | On the same paired seeds, repeat the newest frame across all four input channels for the selected model. This removes temporal variation and changes the input distribution. The score contrast measures the effect of that intervention on this trained policy. |
| MsPacman life and time boundaries | Trace one full seeded greedy game. At each boundary, independently compute the final discounted reward sums and compare the replay successors and multipliers. Replay the prefix under a 160-frame time limit to verify that the cap retains the three bootstrap multipliers. |

CartPole ends on physical failure or the 500-step time limit. On a time limit, replay bootstraps from the last observation. If a physical failure and time cap occur together, the target treats the transition as terminal. MsPacman uses a learning terminal at a lost life while continuing to accumulate the full game score. Each trajectory owns its own return queue, and reset-only steps are excluded from replay and reward counts.

## Report and evidence

Generate the PDF from the checked-in experiment evidence without retraining.

```bash
.venv/bin/python make_report.py --cartpole results/cartpole --mspacman results/mspacman
```

For a newly trained report, pass `runs/cartpole` and `runs/mspacman` instead. The PDF contains the maximum Q and reward plots, 500-episode histograms, the CartPole first-action maps, and the MsPacman game trace. It is written to `output/pdf/assignment2_report.pdf` by default.

- `training.csv` logs raw reward, length, exploration and the largest pre-action Q over all visited states and actions in each episode.
- `selection.csv` logs the held-out greedy means used to choose the checkpoint.
- `evaluation.csv` lists the 500 evaluation returns. `evaluation.json` gives the mean, population standard deviation and checkpoint episode.
- `validation_policies.csv` holds the matched seed comparisons. `validation_grid.csv` holds the CartPole interventions and `validation_trace.csv` holds the MsPacman trajectory. `validation.json` records their summaries and the boundary calculations.
- `metadata.json` records the hyperparameters, device, software versions and training duration.
