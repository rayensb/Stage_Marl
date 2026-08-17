# Environment

Only verified information — checked directly in this repo/session, not assumed. If
something here looks wrong, re-verify with the commands shown rather than trusting this
file blindly; environments (especially Kaggle's) can change between sessions.

## Local development machine (this sandbox)

Verified via direct inspection this session:

- OS: Linux, Ubuntu 24.04-based, kernel `7.0.0-28-generic`, x86_64.
- Python: `3.12.3`.
- Key package versions (as installed in this environment):
  - `torch==2.13.0`
  - `numpy==1.26.4`
  - `gymnasium==1.3.0`
  - `pettingzoo==1.26.1`
  - `pandas==3.0.5`
  - `matplotlib==3.6.3`
- **CUDA is not usable on this machine** — the installed NVIDIA driver is too old for the
  installed PyTorch/CUDA build. `training/train.py`'s `DEVICE` env var will silently fall
  back to behaving like `cpu` regardless of what's requested here. This is a machine-level
  limitation, not a project bug (and CPU is the project's actual measured-optimal default
  anyway — see `DECISIONS.md`).
- This machine is used for: code edits, local smoke tests (`test_env.py`), reviewing
  logs/CSVs, and running `plot_training.py`/`evaluate.py` against already-produced data. It
  is **not** used for full training runs (too slow / no GPU benefit locally, and Kaggle is
  the established training environment — see below).

## Training environment: Kaggle

- Full training runs (`training/train.py`) are executed on Kaggle notebooks, not locally.
- Measured on Kaggle (recorded in `training/train.py:67-74`): CPU sustains ~330-440
  steps/sec; CUDA only ~190 steps/sec for this network size — hence `DEVICE` defaults to
  `cpu`.
- Multi-seed runs are launched in parallel within a single Kaggle session via
  `subprocess.Popen`, with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
  set to prevent the parallel processes from contending over CPU threads.
- **Kaggle's `FileLink`/base64 download mechanism was confirmed unreliable** (verified via
  web search in a prior session) for retrieving trained model files — use Kaggle's native
  Output panel to download artifacts instead.
- Exact Kaggle-side package versions are UNKNOWN/UNVERIFIED from this session — only the
  local versions above were directly checked. If Kaggle's versions diverge meaningfully
  from local (e.g. a different PyTorch major version), that's worth confirming directly on
  Kaggle rather than assuming parity, especially for any RNG- or numerics-sensitive change.

## Git remote

- Origin: `https://github.com/rayensb/Stage_Marl.git`.
- Default branch: `main`.
- This session's push access was already established in prior sessions (multiple successful
  pushes) — treat as configured/working unless a push actually fails.
- **Feature-branch workflow used this session for a large, exploratory change** (the
  vision-tracking redesign): pushed to a separate branch (`vision-tracking`), validated there
  with multiple Kaggle runs, then merged to `main` via a fast-forward push
  (`git push origin vision-tracking:main`) once validated — not a `git merge`/local checkout
  of `main`, since the working session was in an isolated git worktree at the time. Worth
  reusing this pattern for the next similarly-large, unproven change rather than committing
  straight to `main`.

## Kaggle API access from this machine (set up 2026-08-17)

- A dedicated venv at `~/.kaggle-venv` has the `kaggle` CLI installed (`pip install kaggle`
  failed directly into the system Python — externally-managed-environment — hence the venv).
- Credentials (`~/.kaggle/kaggle.json`) were placed directly on this machine by the user,
  outside the chat — the assistant never saw the actual key content, only verified the file
  existed with correct permissions (`600`) and that `kaggle kernels list` authenticated
  successfully.
- Workflow used throughout this session: write a small script-type kernel (`kernel-
  metadata.json` + a `.py` file that `git clone`s the repo — optionally a specific branch via
  `--branch` — installs deps, runs `training/train.py`/`evaluate.py`/`plot_training.py`),
  `kaggle kernels push -p <dir>` to start it, poll `kaggle kernels status <ref>` until
  `COMPLETE`, then `kaggle kernels output <ref> -p <dir>` to download logs/CSVs/models. This
  fully replaced the manual "copy script into a Kaggle notebook cell, run, download via the
  Output panel" workflow described below for prior sessions — both still work, but the API
  path is faster and lets an AI session drive it directly.
- **Gotcha confirmed this session**: transient local network errors when polling status can
  look like remote kernel failures — e.g. `NameResolutionError` contains the substring
  "Error", so a naive case-insensitive error-detection grep in a polling loop can mistake a
  local DNS hiccup for the kernel itself failing. Always re-check the specific kernel's status
  directly (`kaggle kernels status <ref>`) rather than trusting a polling script's classification
  when something looks off.
- Kaggle kernel titles that don't closely match their `id` slug get auto-resolved to a
  derived slug (a warning, not an error) — the actual resulting URL/ref can differ slightly
  from what was requested; always confirm the real slug from the push output before polling.

## Environment variables the project reads (all in `config.py` / `training/train.py`)

| Variable | Default | Purpose |
|---|---|---|
| `NUM_AGENTS` | `4` | Number of drones; drives `OBS_DIM`, `TARGET_DIST`, `SENSOR_RANGE`, etc. Supported values: `2, 3, 4` (others raise `ValueError` in `config.py`). |
| `N_REACT` | `10` | Reaction-time budget (steps) used to derive `REACTION_DIST`/safety margins. |
| `VELOCITY_WEIGHT` | `-0.15` | Reward weight for the velocity component. |
| `SEED` | unset (random) | Sets a reproducible seed and enables `RUN_ID` filename suffixing for parallel multi-seed runs. |
| `DEVICE` | `cpu` | Training device (`cpu` or `cuda`); CPU is the measured-optimal default on Kaggle. |
| `TOTAL_STEPS` | `600_000` | **New 2026-08-17.** Total training steps; made env-var overridable so longer-training tests (this session used up to 3,000,000) don't need a hardcode-then-revert commit cycle. |

No other env vars are read by the training/eval code as of this writing.

## What is NOT part of this environment (as of this writing)

No ROS2, Gazebo, PX4, MAVSDK, or MAVROS anywhere — no such packages, workspaces, or config
files exist in this repo. Confirmed by directory listing; do not assume a robotics
middleware stack is present unless it's actually been added.
