# CueRator

Agentic Search for Symbolic Rules to Adapt Frozen Multimodal Encoders [[Paper](https://arxiv.org/abs/2610.07868)]

This repository contains the code for the OV-AVEL experiments on OV-AVEBench with ImageBind.

## Setup

```bash
cp env.sh.example env.sh          # point OVAVEL_ROOT at your OV-AVEBench copy

conda create -n cuerator python=3.10 -y
conda activate cuerator
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Codex CLI — drives the agents
npm install -g @openai/codex
codex login
codex features enable use_legacy_landlock   # Ubuntu 24.04+
```

The Codex CLI and `bwrap` (bubblewrap) are needed only for the search loop, which calls a
paid Codex account and runs each agent in an unprivileged user namespace — training and
evaluation need neither. The backend model is unset in `experiments/cuerator/config.yaml`,
so the Codex CLI's configured default is used; set `backend.model` to pin one.
The paper's results use `gpt-5.4` with `reasoning_effort: high` via Codex CLI v0.125.0;
`gpt-5.4` has since been retired from Codex with ChatGPT sign-in.
One CUDA GPU per session.

## Data

OV-AVEBench: https://github.com/jasongief/OV-AVEL — set `OVAVEL_ROOT` to that checkout.

```bash
source env.sh
python -m step1_prepare.extract    --encoder imagebind --output data/embeddings/imagebind
python -m step1_prepare.precompute --embeddings data/embeddings/imagebind \
                                   --output     data/similarities/imagebind
```

Writes `data/embeddings/<encoder>/{train,val,test}.npz` and
`data/similarities/<encoder>/{train,val,test}_sim.npz`. If you already have these, drop them
in and skip this step — nothing else reads the raw dataset. The oracle search reads the
similarities only; policy training and evaluation also need the embeddings, which the policy
is conditioned on.
The ImageBind weights download on first use. Install the three PyTorch packages as one
pinned command, as above: installing `torch` alone lets a later dependency pull `torchvision`
from PyPI and replace the CUDA build, and the reported numbers reproduce exactly on torch
2.0.1 but not on newer releases.

## Run

`source env.sh` first — every command below reads `$OVAVEL_ROOT` and resolves `data/`
relative to the repository root.

### Search loop

Runs the full agent loop: propose a formulation, verify it, measure its oracle ceiling,
train a policy on it, and feed the gap back into the next iteration.

```bash
bash experiments/cuerator/orchestrate.sh --run-name my_session
```

Results land in `runs/cuerator/my_session/`:

```
agent_context/formulations/fNNN.py   proposed formulations
agent_context/results/iter_NNN.json  oracle result per iteration
policy_runs/iter_NNN/                policy training output
history.jsonl                        one line per iteration
final_test_summary.json              best-val iteration and its test metrics
orchestrate.log, agent_reasoning.log
```

Re-running with the same `--run-name` resumes and skips completed iterations;
`--start-iter N` forces a starting point. Hyperparameters — iteration count, oracle grid
size, verifier limits, policy training — are in `experiments/cuerator/config.yaml`.

### Train a policy on one formulation

Skips the search loop and trains the sample-conditioned policy directly on a given
formulation. `assets/discovered_formulation.py` is the one reported in the paper:

```bash
python -m step3_bandit.train \
    --formulation assets/discovered_formulation.py \
    --encoder imagebind \
    --epochs 20 --batch_size 32 --num_samples 4 \
    --fixed_action_std 0.3 --train_reward_metric frame_acc \
    --device cuda:0 --seed 42 \
    --per_epoch_test \
    --save_dir runs/policy_only --run_name paper_formulation
```

Writes `runs/policy_only/paper_formulation/`:

```
best.ckpt                  best-validation policy weights
config.json                resolved hyperparameters
epoch_summary.csv          per-epoch train/val/test curve
best_val_summary.json      metrics at the selected epoch
best_val_predictions.jsonl per-video params + metrics on val
test_summary.json          test metrics at the selected epoch
test_predictions.jsonl     per-video params + metrics on test
```

`--run_name` must not already exist; omit it for a timestamped directory instead. Drop
`--per_epoch_test` to keep test untouched during training. `--no_last_ckpt` skips
`last.ckpt`; `--no_train_steps_log` skips the per-step log.

### Evaluate a checkpoint

`assets/policy_best.ckpt` is the policy trained on `assets/discovered_formulation.py`:

```bash
python -m step3_bandit.eval \
    --checkpoint  assets/policy_best.ckpt \
    --formulation assets/discovered_formulation.py \
    --split test \
    --device cuda:0
```

Point `--checkpoint` at any `best.ckpt` from the section above to evaluate your own run.
The encoder and data directories are read back from the checkpoint; `--encoder`,
`--embeddings_dir`, and `--similarities_dir` override them. `assets/policy_best.ckpt`
already points to `assets/discovered_formulation.py`; for your own runs, pass `--formulation`
explicitly, since a checkpoint records the absolute path it was trained from.

Prints per-group (`all` / `close` / `open`) metrics and writes `eval_<split>_summary.json`
plus `eval_<split>_predictions.jsonl` next to the checkpoint, or into `--output_dir`.

## Notes

Third-party code notices: see `NOTICE`. The vendored ImageBind code is CC-BY-NC 4.0.

## Citation

```bibtex
@article{park2026cuerator,
  title   = {CueRator: Agentic Search for Symbolic Rules to Adapt Frozen Multimodal Encoders},
  author  = {Park, Sunchan and Cho, Beomkwon and Kong, Kyeongbo},
  journal = {arXiv preprint arXiv:2610.07868},
  year    = {2026}
}
```