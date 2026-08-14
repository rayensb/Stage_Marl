# Session Handoff

**Read this file every new session — it's the most likely to be stale, and the most
important for not repeating work or missing what just happened.** Update it whenever a
session ends mid-task or reaches a natural checkpoint.

## Last updated

2026-08-14, end of the session that analyzed the `NUM_AGENTS=3`, 3-seed validation run and
applied the `RECOVERY_MAX_TRIGGERS` fix.

## What was happening

Continuing from the previous session's `docs/ai_context/` creation, this session: (1)
reconstructed and verified project understanding against the actual repo (confirmed accurate
except two discrepancies — see below), (2) received and fully analyzed a complete
`NUM_AGENTS=3`, 3-seed dataset the user had locally, (3) diagnosed the collision-rate-collapse
root cause, (4) applied a fix, (5) the user then shared the Kaggle launcher script intended
for the `NUM_AGENTS=4`, 3-seed validation run and asked whether to run it — the fix was
applied first, per the user's explicit choice, before that run should be launched.

## What changed this session

- **Verified `docs/ai_context/` against the actual repo** — accurate except: (a) it
  undercounted the `NUM_AGENTS=3` data present locally (only recorded 2 of 6 eval CSVs), and
  (b) `envs/formation_env.py`'s own module docstring is stale — it claims `TARGET_DIST` is
  "not a general-N formula" / "a 4-agent study," but `config.py`'s `_PACKING_RATIO` has
  already been generalized to `NUM_AGENTS ∈ {2,3,4}` since commit `2576f92`. **This docstring
  fix was not applied** — flagged but out of scope for this session's actual task; do it if
  picking up loose ends.
- **Analyzed the complete `NUM_AGENTS=3`, 3-seed dataset** (`training_log_n3_{1,2,3}.csv`,
  `eval_n3_{1,2,3}.csv`, `eval_best_n3_{1,2,3}.csv`, plus the training-curve plots). Full
  results and root-cause diagnosis are in `EXPERIMENT_LOG.md` (new top entry) and
  `KNOWN_ISSUES.md` item 1 (rewritten from "unverified" to confirmed). Headline: final-model
  collision_rate 26-46% across all 3 seeds by `TOTAL_STEPS=600k`, caused by
  `RECOVERY_MAX_TRIGGERS=5` being exhausted by rollout ~110-125, leaving entropy unprotected
  for the remaining ~60% of training.
- **Applied a fix**: `RECOVERY_MAX_TRIGGERS` raised 5 → 20 in `training/train.py` (with a
  detailed inline comment per this project's convention). Documented in `DECISIONS.md` (new
  follow-up note under "Plateau-triggered entropy recovery") and `KNOWN_ISSUES.md` item 1.
  **This fix is applied locally but not yet committed/pushed, and is itself unverified
  against a completed run.**
- **The user's prepared `NUM_AGENTS=4`, 3-seed Kaggle launcher script was reviewed and judged
  correct in structure**, but deliberately **not run yet** — running it against the pre-fix
  code would very likely just reproduce the same relapse at a harder curriculum stage. The
  user chose to fix first, then run. See `EXPERIMENT_LOG.md`'s "deliberately deferred" entry.

## Unresolved / pending as of this handoff

1. **Commit and push the `RECOVERY_MAX_TRIGGERS` fix** (`training/train.py`) — not yet done
   as of this handoff; check `git status`/`git log` to confirm before assuming it's live on
   `origin/main`. The Kaggle launcher script `git clone`s from GitHub, so the fix must be
   pushed before that script will pick it up.
2. **Verify the fix with a single-seed `NUM_AGENTS=3` rerun** before spending a 3-seed
   `NUM_AGENTS=4` Kaggle batch on it — confirm `collision_rate` no longer relapses in the
   back half of training (i.e. stays low past rollout ~125 through `TOTAL_STEPS`).
3. **If the single-seed check looks good**, the user's prepared `NUM_AGENTS=4`, 3-seed
   Kaggle launcher script (already reviewed, structurally correct — clones repo, no
   `NUM_AGENTS` override needed since `config.py` defaults to 4, seeds 1/2/3 in parallel with
   thread pinning, evaluates + plots each on completion) is ready to launch as-is.
4. `envs/formation_env.py`'s stale module docstring (see above) — not fixed, not urgent,
   flagged for whoever has time.
5. `readme.txt` is still stale (unchanged from prior sessions — describes the removed
   per-drone-file save scheme).

## Exact recommended next step for a new session

Check `git status`/`git log` first — has the `RECOVERY_MAX_TRIGGERS` fix been committed and
pushed? If not, do that (or ask the user to). Then: has a single-seed `NUM_AGENTS=3`
verification run happened since the fix landed? If yes, check whether `collision_rate` stayed
low past rollout ~125 — if so, the `NUM_AGENTS=4` 3-seed Kaggle run is cleared to launch using
the script already reviewed this session. If no verification run has happened yet, that's the
next thing to do before touching N=4.
