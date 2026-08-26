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

- Origin: `https://github.com/rayensb/Stage_Marl.git` (note: this is the *repo's* org/name —
  the Kaggle account username is different, see below, don't conflate the two).
- Default branch: `main`. **`origin/main` was fast-forwarded to `phase2-combined` (`8b724bb`) on
  2026-08-23** — no longer behind, correcting a claim ("still at `b59c139`, substantially
  behind") that persisted across several doc passes after the push actually happened. Note that
  a local `main` ref in a given worktree/clone can still show an older commit until someone
  checks it out and pulls — check `origin/main` directly (`git log --oneline -1 origin/main`,
  after a `git fetch`) rather than trusting a local `main` ref or an older doc claim.
  `phase3-resilience` (current work) branches off `phase2-combined`, i.e. off this same point.
- **Feature-branch workflow, now used repeatedly, not just once**: `vision-tracking` (merged),
  `n-aware-margin` and `xyz-spread` (hypothesis branches, merged into `phase2-combined`),
  `xyz-spread-fixed` (isolated angle-bug retest, not merged — its finding was "no measurable
  difference," so nothing to merge beyond the fix already in `phase2-combined`'s successor),
  `phase2-combined` (validated reference point, not yet fast-forwarded to `main`),
  `phase3-resilience` (current work, off `phase2-combined`), `px4-deployment-wip` (the
  concurrent deployment thread's branch — separate workstream, see `ARCHITECTURE.md`). Worth
  continuing this pattern for any further large, unproven change rather than committing
  straight to `main`.
- **This worktree is shared with a concurrent conversation** doing PX4/Gazebo/ROS2 deployment
  work (see `ARCHITECTURE.md`'s `deployment/` section). Coordination happens via
  `PHASE2_CHECKPOINT.md` at the repo root — check it before touching `envs/formation_env.py`,
  `config.py`, or `training/*.py` if unsure whether another session has in-flight work there.

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
- **The Kaggle account username is `rayensboui`** (confirmed via `kaggle kernels list --mine`)
  — every kernel ref is `rayensboui/<kernel-slug>`. This is easy to get wrong by analogy with
  the GitHub org (`rayensb`, one letter different) — a wrong-username guess fails with
  `Permission 'kernels.get' was denied`, which reads like a slug typo, not an owner typo; if
  that error shows up, check the owner segment first, not just the slug.
- **`kaggle kernels list --mine`'s default sort order is not reliable for "what's running right
  now."** Its `lastRunTime` column does not sort strictly newest-first across the full account
  — kernels pushed hours later can appear far down a default-page listing, behind kernels
  pushed earlier the same day (confirmed directly: a kernel pushed at 22:17 didn't appear in
  the top 15 of a `--page-size 100` listing that still led with a kernel pushed at 13:34 the
  same day). Don't use this listing's ordering to infer recency or currently-running state —
  check specific kernels' status individually via `kaggle kernels status <ref>` instead.
- **Confirm `git status` shows up to date with `origin` immediately before every Kaggle
  launch, every time.** Standing rule, adopted after ~8 kernels across two feature branches
  silently ran stale code because the local commits they were meant to test hadn't been pushed
  yet — the kernels' fresh `git clone` picked up whatever was on `origin` at push time, not what
  was on disk locally. Caught via a missing-columns anomaly in the downloaded eval CSVs, not
  assumed. See `KNOWN_ISSUES.md` item 17.
- **Kaggle datasets (`dataset_sources` in `kernel-metadata.json`) mount at
  `/kaggle/input/datasets/<owner>/<dataset-slug>/`, not the classic `/kaggle/input/
  <dataset-slug>/`** the current `kaggle-cli` docs still describe (confirmed against the docs
  directly, not just assumed stale). Confirmed 2026-08-22 via a throwaway diagnostic kernel
  (`os.walk("/kaggle/input")`) after two real kernels failed with `cp: cannot stat` on the
  documented path — `kaggle datasets status` reporting `ready` does not mean the dataset is
  correctly referenceable at the path the docs describe. If using a dataset to resume training
  from a checkpoint (see `COMMANDS.md`), use the `datasets/<owner>/` form.
- **Kaggle's "Maximum batch CPU session count of 5 reached" can fire even when fewer than 5
  kernels show `RUNNING` via individual `kernels status` checks.** Observed directly
  (2026-08-20): 4 kernels confirmed `RUNNING` by name, every other kernel on the account
  confirmed `COMPLETE` by name, yet a 5th push still hit the cap. Leading hypothesis, not
  confirmed: `kernels status` reports only a kernel's *latest version's* session — if an
  earlier push of the same kernel slug was interrupted mid-flight and had already started
  running server-side before a newer version was pushed on top of it, the old version's
  session may continue occupying a slot invisibly. The `kaggle` CLI has no command to list
  active sessions independent of a specific kernel ref, and no command to cancel one — only
  Kaggle's own web UI (`kaggle.com` → "Your Work" → Notebooks) shows this directly. If this
  recurs: check the web UI for a stray running session before assuming the count itself is
  wrong, and avoid re-pushing the *suspected* orphaned kernel's slug again while a known-good
  version of it is still legitimately training (that would risk restarting real progress).

## Environment variables the project reads (all in `config.py` / `training/train.py`)

| Variable | Default | Purpose |
|---|---|---|
| `NUM_AGENTS` | `4` | Number of drones; drives `OBS_DIM`, `TARGET_DIST`, `SENSOR_RANGE`, etc. Supported values: `2, 3, 4` (others raise `ValueError` in `config.py`). |
| `N_REACT` | `10` | Reaction-time budget (steps) used to derive `REACTION_DIST`/safety margins. |
| `VELOCITY_WEIGHT` | `-0.15` | Reward weight for the velocity component. |
| `SEED` | unset (random) | Sets a reproducible seed and enables `RUN_ID` filename suffixing for parallel multi-seed runs. |
| `DEVICE` | `cpu` | Training device (`cpu` or `cuda`); CPU is the measured-optimal default on Kaggle. |
| `TOTAL_STEPS` | `600_000` | Total training steps; env-var overridable (this session has used up to 5,000,000). |
| `MAX_STEPS` | `1800` (was `200`) | **New 2026-08-20 (Phase 3).** Episode length in steps (`1800` = 90s at `DT=0.05`, raised from the original 10s after `diagnose_horizon.py` showed tracking degrading past the old horizon). `ROLLOUT_LEN = MAX_STEPS * ROLLOUT_EPISODES` derives from this automatically — see `ARCHITECTURE.md`. |
| `LOST_TIMEOUT_SEC` | `6.0` (was `2.0`) | **New 2026-08-20.** Vision-tracking dead-reckoning grace period before `target_lost` termination. Was hardcoded; made overridable after the flat 2.0s value broke badly once `MAX_STEPS` grew 9x (see `KNOWN_ISSUES.md` item 12). A 6/10/18s sweep found no clean dose-response; default resolved to `6.0` — active search (not this constant) turned out to be the real lever. |
| `DISABLE_TARGET_LOST_TERMINATION` | `0` (off) | **New 2026-08-21.** Ablation flag — when `1`, `target_lost` still computes/penalizes identically but no longer ends the episode. Tested as a diagnostic (improved `contact_fraction` but made collision safety seed-dependent and unstable) and **not adopted** — leave at the default. See `DECISIONS.md`. |
| `ACTOR_HIDDEN` | `128` | **New 2026-08-24.** Actor MLP hidden width. Validated by a 5-way network-capacity sweep as a genuine sweet spot for `NUM_AGENTS=4` — see `EXPERIMENT_LOG.md`. |
| `CRITIC_HIDDEN` | `256` | **New 2026-08-24.** `CentralCritic` trunk hidden width, same sweep as above. |
| `CRITIC_LR` | equals `LR` (`3e-4`) | **New 2026-08-25.** Separate critic optimizer learning rate. Tested at `1e-5` as a fix for critic-saturation (see `KNOWN_ISSUES.md` item 19) and **rejected** — mixed/inconsistent behavioral effects at full scale. Leave at the default. See `DECISIONS.md`. |

No other env vars are read by the training/eval code as of this writing.

## PX4/Gazebo/ROS2 environment — a separate workstream, not part of this doc suite's scope

**This is now out of date if it still says "no ROS2/Gazebo/PX4 in this repo" — that claim was
true through 2026-08-17 and is not true anymore.** `deployment/` contains a real PX4 SITL +
Gazebo + ROS2 inference pipeline, built by a concurrent conversation sharing this worktree
(see `ARCHITECTURE.md`'s `deployment/` section for how the two workstreams connect). Its own
environment details (PX4-Autopilot SITL, Gazebo, ROS2 Jazzy, px4_msgs/px4_ros_com, Micro XRCE-
DDS agent, and whatever machine/setup specifics apply) belong in `deployment/docs/`, not here —
this file documents the MARL training/simulation environment specifically. If you need the
deployment side's environment details, read `deployment/docs/` directly rather than assuming
this file covers it.
