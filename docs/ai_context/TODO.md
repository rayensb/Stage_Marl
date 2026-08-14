# TODO / Roadmap

Prioritized, not sequenced by date. "Critical" blocks understanding current results;
"High" is the next real work; "Medium" is valuable but not blocking; "Research/Future" is
speculative scope, not committed work.

## Critical

- **Get and analyze the pending `NUM_AGENTS=4`, 3-seed Kaggle results.** This validates the
  entire current fix chain (see `EXPERIMENT_LOG.md` — "NUM_AGENTS=4, 3-seed validation run").
  Until this lands, it's unknown whether the `63274b1` cohesion/safety fix actually resolved
  the collision-rate collapse it targeted. This is the explicit next step the user flagged
  before this documentation task was created — check `SESSION_HANDOFF.md` first, results may
  have already arrived by the time this is read.
- **Confirm `test_env.py` still passes after the `a40db9b`..`63274b1` commit chain.** Simple,
  fast, and hasn't been confirmed re-run since those commits per this session's information.

## High

- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-
  sum switch, see `KNOWN_ISSUES.md` item 3) before doing any cross-run comparison of
  `r_track`/`r_safety`/etc. magnitudes between pre- and post-`63274b1` logs.
- **Properly analyze the `NUM_AGENTS=3` evaluation CSVs** (`eval_best_n3_2.csv`,
  `eval_best_n3_3.csv` in `~/Downloads/`, see `EXPERIMENT_LOG.md`) rather than relying on the
  informal skim already done — locate the matching training logs too, not just the
  evaluation output.
- **Update or replace `readme.txt`** (see `KNOWN_ISSUES.md` item 2) — it currently describes
  a removed per-drone-file architecture and will mislead anyone who reads it instead of
  `docs/ai_context/`.

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
