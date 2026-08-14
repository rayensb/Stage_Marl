# TODO / Roadmap

Prioritized, not sequenced by date. "Critical" blocks understanding current results;
"High" is the next real work; "Medium" is valuable but not blocking; "Research/Future" is
speculative scope, not committed work.

## Critical

- **Commit and push the `RECOVERY_MAX_TRIGGERS` fix** (`training/train.py`, 5 → 20) — applied
  locally 2026-08-14, not yet confirmed committed/pushed. See `SESSION_HANDOFF.md`.
- **Verify the fix with a single-seed `NUM_AGENTS=3` rerun** before launching the prepared
  `NUM_AGENTS=4`, 3-seed Kaggle run — confirm `collision_rate` doesn't relapse past rollout
  ~125 (where the old 5-trigger budget used to run out). See `KNOWN_ISSUES.md` item 1 and
  `EXPERIMENT_LOG.md` for the full diagnosis this fix is based on.
- **Once verified, launch the `NUM_AGENTS=4`, 3-seed Kaggle run** (launcher script already
  reviewed and structurally confirmed correct this session) — deliberately deferred, not
  abandoned. This validates the full fix chain at the actual target agent count.
- **Confirm `test_env.py` still passes after the `a40db9b`..`63274b1` commit chain.** Simple,
  fast, and hasn't been confirmed re-run since those commits per this session's information.

## High

- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-
  sum switch, see `KNOWN_ISSUES.md` item 3) before doing any cross-run comparison of
  `r_track`/`r_safety`/etc. magnitudes between pre- and post-`63274b1` logs.
- **Update or replace `readme.txt`** (see `KNOWN_ISSUES.md` item 2) — it currently describes
  a removed per-drone-file architecture and will mislead anyone who reads it instead of
  `docs/ai_context/`.
- **Fix `envs/formation_env.py`'s stale module docstring** — claims `TARGET_DIST` is "not a
  general-N formula" / "a 4-agent study," but `config.py` has generalized it to
  `NUM_AGENTS ∈ {2,3,4}` since commit `2576f92`. Found 2026-08-14, not yet fixed.

~~Properly analyze the `NUM_AGENTS=3` evaluation CSVs~~ — **done 2026-08-14**, full 3-seed
analysis in `EXPERIMENT_LOG.md`, root cause identified and fixed (see Critical, above).

## Medium

- Once `NUM_AGENTS=4` results are in, decide whether the curriculum is "done" (2→3→4 all
  validated) or whether another iteration of reward tuning is needed at N=4 specifically —
  the previous conflicts (relock loophole, cohesion/safety collapse) were all discovered at
  higher agent counts, so N=4 is the most likely place for a new one to surface.
- Consider whether `r_spread`'s horizontal-only limitation (`KNOWN_ISSUES.md` item 5) is
  actually causing any exploitable behavior, now that there's more training data across
  agent counts to check against.

## Research / Future (not committed, exploratory)

- A genuinely decentralized neighbor-discovery mechanism, if the project ever needs to move
  beyond the current centralized mutual-k-NN simulator shortcut (`DECISIONS.md`) — e.g. for
  real-hardware or ROS2/Gazebo/PX4 integration. No such integration exists in this repo
  today; this would be new scope, not a bug fix.
- An actor architecture that's agent-count-invariant (e.g. attention over a variable number
  of neighbors instead of a fixed `EFFECTIVE_K`-sized observation), which would enable actual
  weight transfer across the `NUM_AGENTS` curriculum stages instead of the current
  from-scratch comparative validation at each stage (`DECISIONS.md`).
