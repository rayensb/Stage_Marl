# Commands

All commands assume the working directory is the repo root (`/home/rayen/stage` locally).
`training/train.py` and other scripts under `training/` insert the repo root onto
`sys.path` themselves (`train.py:1-2`), so they can be run directly with `python` without
needing `PYTHONPATH` set manually or `-m` module syntax.

**No `requirements.txt`, `setup.py`, or `pyproject.toml` exists in this repo** (confirmed by
directory listing). Dependencies are whatever's installed in the active Python environment —
see `ENVIRONMENT.md` for the versions verified locally. If setting up a fresh environment,
install at minimum: `torch numpy gymnasium pettingzoo pandas matplotlib`.

## Smoke test the environment (fast, local, no training)

```bash
python test_env.py
```

Runs 200 random-action steps against `FormationEnv3D(num_agents=4, k_neighbors=2)`. No
assertions beyond not crashing — this is a liveness check, not a correctness test suite (no
formal test suite exists in this repo as of this writing).

## Train (full run — normally run on Kaggle, not locally; see `ENVIRONMENT.md`)

Default (`NUM_AGENTS=4`, random seed, CPU, unsuffixed output files):

```bash
python training/train.py
```

Reproducible single-seed run with suffixed outputs:

```bash
SEED=1 python training/train.py
```

Curriculum stage at a different agent count (e.g. `NUM_AGENTS=3`), reproducible seed —
output files get an `n3_`-prefixed `RUN_ID` automatically since `NUM_AGENTS != 4`:

```bash
NUM_AGENTS=3 SEED=1 python training/train.py
```

Explicit device override (only meaningful where CUDA actually works — not this local
sandbox):

```bash
DEVICE=cuda SEED=1 python training/train.py
```

Longer training run (`TOTAL_STEPS` env var, added 2026-08-17 — default `600_000`; this
session validated up to `3_000_000` for the vision-tracking system, where it made a real
difference — see `EXPERIMENT_LOG.md`):

```bash
TOTAL_STEPS=3000000 NUM_AGENTS=3 SEED=1 python training/train.py
```

Parallel multi-seed runs (the pattern used on Kaggle — run from separate processes/sessions,
each with its own `SEED`, pinning single-threaded math libs to avoid CPU contention):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=1 python training/train.py &
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=2 python training/train.py &
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=3 python training/train.py &
wait
```

Interrupting a run (`Ctrl+C` / `SIGINT`, or `SIGTERM`) triggers a graceful stop: the current
rollout finishes, a checkpoint is saved, then the process exits (`train.py:91-99`) — safe to
interrupt rather than killing the process outright.

Outputs (paths relative to repo root, all git-ignored — see `.gitignore`):
- `logs/training_log[_<run_id>].csv` — one row per rollout, see `training/logger.py:5-10`
  for the full field list.
- `checkpoints/` — regular training checkpoints (via `training/checkpoint.py`).
- `models/actor_best[_<run_id>].pt` — best checkpoint by the lexicographic criterion (see
  `DECISIONS.md`).
- `models/actor[_<run_id>].pt` — final model, written only on normal (non-interrupted)
  completion.

## Plot training diagnostics from a log

```bash
python training/plot_training.py
```

Resolves `run_id` the same way `train.py` does (CLI arg > `SEED` env var, `NUM_AGENTS`-aware
prefix) — pass the run explicitly if not relying on env vars:

```bash
python training/plot_training.py n3_1
```

Produces an **8x2** diagnostic subplot grid (grew from 7x2 on 2026-08-17): collision rate +
target_lost rate overlay (was collision-only), entropy, pairwise distance, swarm diameter,
reward components (now 8, includes `r_contact`), critic loss, approx KL, clip fraction,
throughput, entropy coefficient, entropy-recovery activity, `log_std_mean`, `mean_action_abs`,
and the closing-speed brake's mean speed removed. Saves to
`logs/training_curves[_<run_id>].png` (confirmed — `plt.savefig`, not just interactive
display).

## Kaggle runs, driven from this machine via the API

See `ENVIRONMENT.md` for the `~/.kaggle-venv` setup. The pattern used throughout the
2026-08-17 session: write `verify.py` (clones the repo, optionally a specific branch via
`git clone --branch <name>`, installs `pettingzoo`/`gymnasium`, runs train → evaluate (final)
→ evaluate `--best` → plot) plus a matching `kernel-metadata.json`
(`kernel_type: "script"`, `enable_internet: true`, `enable_gpu: false`), then:

```bash
~/.kaggle-venv/bin/kaggle kernels push -p <dir-containing-verify.py-and-metadata>
~/.kaggle-venv/bin/kaggle kernels status <ref-from-push-output>
~/.kaggle-venv/bin/kaggle kernels output <ref> -p <download-dir>
```

Multiple seeds/configs can run as separate kernels in parallel (each is an independent Kaggle
session — no `OMP_NUM_THREADS`-style contention between separate kernels the way there is
between `subprocess.Popen`-launched processes *within* one kernel).

## Evaluate a trained model (deterministic rollout)

Regular final model, default settings:

```bash
python training/evaluate.py --run-id 1
```

Best checkpoint instead of the final model:

```bash
python training/evaluate.py --run-id 1 --best
```

Full option set (`training/evaluate.py`, argparse-defined): `--episodes, --model-dir,
--run-id, --seed (default 0), --device, --save-trajectory, --best`. Run
`python training/evaluate.py --help` to confirm current exact flag behavior/defaults rather
than trusting this list if it's been a while — this file wasn't re-diffed against the
argparse block in this session beyond the initial read.

Output: a per-episode metrics CSV (path logic in `evaluate.py` depends on the
`--best`/`--run-id` combination — check the script if the exact output filename matters).

## Git

Standard workflow — no custom scripts or CI pipeline exist in this repo as of this writing
(no `.github/workflows`, no `Makefile` found). Check `git status` / `git log` directly
rather than assuming any hook or pipeline runs automatically on commit/push.
