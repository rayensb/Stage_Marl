# TODO / Roadmap

Prioritized, not sequenced by date. "Critical" blocks understanding current results;
"High" is the next real work; "Medium" is valuable but not blocking; "Research/Future" is
speculative scope, not committed work.

## Critical

- **Run `NUM_AGENTS=4`, 3-seed validation on the current `main` (`b59c139` or later).** This
  is the actual target agent count and the whole reason the collision and tracking problems
  needed solving. Both problems are now resolved and validated at `NUM_AGENTS=3` — this is
  the first time in the project's history that an N=4 run would be testing something
  genuinely ready, rather than a known-broken baseline. Use the same launcher pattern
  (parallel per-seed Kaggle kernels) established throughout this session — see `COMMANDS.md`.
- **Specifically check the closing-speed brake's rare multi-agent collision edge case at
  N=4.** ~1% collision rate showed up at `NUM_AGENTS=3` over long/replicated runs, consistent
  with a known gap in the brake's proof (verified for two agents at a time, not simultaneous
  multi-threat braking). More agents means more chances for this — worth explicit attention,
  not just a passive check of the aggregate `collision_rate` number.

## High

- **Update or replace `readme.txt`** — more stale than ever after this session (see
  `KNOWN_ISSUES.md` item 2).
- **Fix `envs/formation_env.py`'s remaining stale docstring paragraph** (the old "4-agent
  study, not general-N" SCOPE note — contradicted by `config.py`'s actual generalized
  `_PACKING_RATIO` since before this session). A new, accurate docstring paragraph was added
  for vision-tracking without touching this pre-existing one.
- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-
  sum switch) before comparing `r_track`/`r_safety`/etc. magnitudes between logs from before
  and after that commit — still relevant for anyone comparing very old logs to current ones.

## Medium

- **UWB-realistic neighbor sensing**, replacing the still-exact-ground-truth neighbor
  observations (`rel_p`/`rel_v`/`d`). Deliberately deferred when vision-tracking was built
  (see `DECISIONS.md`) — real UWB gives clean range, not clean bearing/velocity, so this
  deserves its own careful design pass, not a quick patch. Natural next realism step now that
  target-sensing is done.
- **Make target motion non-constant-velocity** (occasional direction/speed changes
  mid-episode). Currently the 2-second dead-reckoning grace period in the vision-tracking
  system is mathematically exact, not a real approximation, precisely because the target
  never maneuvers — this change would make that part of the system actually get tested under
  real uncertainty. See `KNOWN_ISSUES.md` item 9.
- **Sensing noise/occlusion on the target reading itself**, and a directional FOV cone
  (needs heading/orientation added as a new state dimension first — a bigger prerequisite
  change than the noise/occlusion pieces). See `KNOWN_ISSUES.md` item 11.
- Consider whether `r_spread`'s horizontal-only limitation is actually causing any
  exploitable behavior, now that there's much more training data to check against.

## Research / Future (not committed, exploratory)

- A genuinely decentralized neighbor-discovery mechanism for the *graph maintenance* itself
  (separate from UWB-realistic sensing above) — `_repair_connectivity` still uses centralized
  ground-truth position knowledge to maintain the lock graph, which is a simulator-level
  shortcut, not just an observation-realism question.
- An actor architecture that's agent-count-invariant (e.g. attention over a variable number
  of neighbors instead of a fixed `EFFECTIVE_K`-sized observation), which would enable actual
  weight transfer across the `NUM_AGENTS` curriculum stages instead of the current
  from-scratch comparative validation at each stage (`DECISIONS.md`).
- Real drone dynamics (inertia/acceleration limits, actuator delay) instead of the current
  direct-velocity-command kinematic model — flagged as a "not okay for claiming realistic
  drone collision avoidance" simplification by an earlier supervisor review, still true.
- A learned nominal controller + safety-filter architecture more broadly (the closing-speed
  brake is a first, narrow instance of this pattern for one specific failure mode — a
  supervisor review's broader recommendation for "control barrier functions"/"action
  projection" as a general design philosophy is only partially adopted so far).
