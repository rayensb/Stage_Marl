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

Runs random-action steps against `FormationEnv3D`. No assertions beyond not crashing — this
is a liveness check, not a correctness test suite.

```bash
python test_geometry.py
```

**Added 2026-08-20.** Verifies `_IDEAL_NEIGHBOR_ANGLE` (the local, drone-viewpoint spread
angle) against an independent analytical construction (regular simplex via standard basis
vectors), and explicitly asserts it's a different quantity from `_PACKING_RATIO`'s global,
target-viewpoint angle. This is a real correctness check with assertions (unlike
`test_env.py`'s liveness-only check) — added specifically because the two angles were once
conflated in shipped code; see `DECISIONS.md`/`KNOWN_ISSUES.md` item 5.

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

Longer training run (`TOTAL_STEPS` env var — default `600_000`; runs up to `5_000_000` have
been used for full validation sweeps — see `EXPERIMENT_LOG.md`):

```bash
TOTAL_STEPS=3000000 NUM_AGENTS=3 SEED=1 python training/train.py
```

Longer episodes and a non-default dead-reckoning grace period (`MAX_STEPS`/`LOST_TIMEOUT_SEC`
env vars, both added 2026-08-20 for Phase 3 — see `ARCHITECTURE.md`/`KNOWN_ISSUES.md` item 12;
`MAX_STEPS=1800`/`LOST_TIMEOUT_SEC=2.0` are now the defaults, shown explicitly here since
they're the values currently being swept):

```bash
MAX_STEPS=1800 LOST_TIMEOUT_SEC=10 NUM_AGENTS=4 SEED=1 TOTAL_STEPS=3000000 python training/train.py
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

Produces a multi-panel diagnostic subplot grid — collision/target_lost/ground_strike rate
overlays, entropy, pairwise distance, swarm diameter, all 11 reward components (see
`ARCHITECTURE.md`), critic loss, approx KL, clip fraction, throughput, entropy coefficient,
entropy-recovery activity, `log_std_mean`, `mean_action_abs`, and the brake's mean speed
removed plus its Phase-1-era instrumentation (`mean_brake_passes` etc. — see
`training/logger.py`'s `FIELDS` for the exhaustive current list, it has grown incrementally
each phase and the exact panel count wasn't re-verified for this doc pass). Saves to
`logs/training_curves[_<run_id>].png` (`plt.savefig`, not just interactive display).

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

### Resuming a training run from an existing checkpoint (added 2026-08-22)

`train.py` already resumes automatically (`load_checkpoint` at the start of `main()`) whenever
`checkpoints/latest[_<run_id>].pt` exists in the working directory and `TOTAL_STEPS` is higher
than what's saved in it — no code change needed, just get the checkpoint file into place before
`train.py` runs. Since each Kaggle kernel does a fresh `git clone`, the checkpoint has to be
supplied separately as a Kaggle Dataset:

```bash
# one-time: package the checkpoint(s) and upload as a private dataset
mkdir -p ckpt_upload && cp checkpoints/latest_1.pt ckpt_upload/
cat > ckpt_upload/dataset-metadata.json << 'EOF'
{"title": "My Resume Checkpoints", "id": "rayensboui/my-resume-ckpts", "licenses": [{"name": "CC0-1.0"}]}
EOF
~/.kaggle-venv/bin/kaggle datasets create -p ckpt_upload
~/.kaggle-venv/bin/kaggle datasets status rayensboui/my-resume-ckpts  # wait for "ready"
```

Then reference it from `kernel-metadata.json`'s `dataset_sources: ["rayensboui/my-resume-ckpts"]`,
and in `verify.py`, **before** running `train.py`, copy the file from the dataset's mount path
into `checkpoints/` — note the path (see `ENVIRONMENT.md` for why this isn't the classically-
documented shorter form):

```python
import os, subprocess
os.makedirs("checkpoints", exist_ok=True)
subprocess.run(["cp", "/kaggle/input/datasets/rayensboui/my-resume-ckpts/latest_1.pt",
                "checkpoints/latest_1.pt"], check=True)
```

Set `TOTAL_STEPS` to the new (higher) target in `verify.py`'s env — everything else (`SEED`,
`NUM_AGENTS`, any other override) should match the original run exactly, or the resumed run is a
different experiment, not a continuation of the same one. **Caveat**: `train.py`'s LR and
entropy-coefficient schedules are driven by `total_steps / TOTAL_STEPS` — if the original run's
schedule had already annealed to its floor (i.e. it ran to completion), resuming with a higher
`TOTAL_STEPS` makes both jump back up rather than continuing one smooth anneal. Real, not a bug,
but means the resumed portion isn't quite equivalent to an uninterrupted single run at the new
step count — see `EXPERIMENT_LOG.md`'s search-t10 5M-resume entry for a worked example.

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

## Check sustained-flight behavior past the training horizon

**Added 2026-08-20.** Loads a frozen checkpoint and runs it well past `MAX_STEPS`, without
terminating on the first failure, so degradation past the training horizon is actually visible
instead of just pass/fail. This is what surfaced the tracking-degradation evidence that
motivated Phase 3 — see `EXPERIMENT_LOG.md`.

```bash
python training/diagnose_horizon.py
```

Check the script directly for its current CLI flags (model/run-id selection, duration) — not
re-documented here in detail since it's a newer, less-used tool than `evaluate.py`.

## Git

Standard workflow — no custom scripts or CI pipeline exist in this repo as of this writing
(no `.github/workflows`, no `Makefile` found). Check `git status` / `git log` directly
rather than assuming any hook or pipeline runs automatically on commit/push.
