# Session Handoff

**Read this file every new session — it's the most likely to be stale, and the most
important for not repeating work or missing what just happened.** Update it whenever a
session ends mid-task or reaches a natural checkpoint.

## Last updated

2026-08-19. Prior entry (2026-08-17) covered the session that solved collision and tracking at
`NUM_AGENTS=3` and merged to `main` (`b59c139`). Since then: `NUM_AGENTS=4` was run (3 seeds,
3M steps, against commit `45b42a2`) with a mixed result — tracking generalized cleanly,
collision did not — followed by a design discussion (brake determinism, formation geometry,
proposed fixes) and this doc update. **No code has changed in this pass** — `main` is still at
`45b42a2`; everything new below is analysis and proposals, not yet implemented.

## What was happening (the short version)

This session started by reconstructing context from a prior session's handoff (which was
waiting on `NUM_AGENTS=4` results that had never actually arrived), found the actual local
data was `NUM_AGENTS=3`, and used it to diagnose a long-standing collision-rate collapse. Six
reward-shape/schedule fixes in a row failed (recovery-trigger budget, recovery timing,
`r_safety` zone reshape, a diameter floor) before direct instrumentation revealed the real
cause: the policy was learning to commit to larger-magnitude actions over training, not
losing exploration noise — every prior fix had targeted the wrong variable. The actual fix
was a **deterministic action-space safety layer** (the closing-speed brake), not a reward
change, and it worked immediately and completely (`collision_rate=0.000`).

With collision solved, focus shifted to tracking quality, which led to a deliberate,
carefully-scoped redesign: **vision-based cooperative target tracking**, replacing the
ground-truth telemetry every drone had always had access to. This surfaced a new failure mode
(`target_lost`) that took two more iterations to resolve (disabling a diameter floor that had
become counterproductive, then training substantially longer) before a 3-seed, 3M-step
validation confirmed it: 0-3% target-lost, 0-1% collision, best tracking accuracy of the whole
session. Merged to `main`, then `docs/ai_context/` was brought up to date on explicit request
("document all") — commit `45b42a2`.

**Then (2026-08-19)**: the planned `NUM_AGENTS=4` validation ran (3 seeds, 3M steps) — see
`EXPERIMENT_LOG.md`. Tracking transferred cleanly with no changes. Collision did not: eval-time
numbers looked fine (0-2%) but training-time rolling-window data showed the real picture —
scattered, non-converging collision events throughout the full run in all 3 seeds, confirming
what `KNOWN_ISSUES.md` item 8 had only flagged as theoretical before. This led to a design
discussion covering: whether the closing-speed brake is "still ML" (it isn't — a deterministic
layer on top of a fully-learned policy, same pattern as ABS on a car), a geometric check of
whether `N=4`'s target formation numbers are even mutually satisfiable (they are — verified
exactly, see `EXPERIMENT_LOG.md`'s geometric feasibility entry), and four concrete, **not yet
implemented** proposed fixes now queued in `TODO.md` (multi-pass brake convergence, a direct
brake-engagement reward penalty, an N-aware safety margin, and a 3D/XYZ `r_spread`). Separately,
the user shared a detailed PX4+Gazebo+ROS2 SITL deployment-architecture writeup (their own setup
work on a separate machine) requesting feedback — feedback was given inline in conversation, but
per the user's own explicit request that workflow's documentation is deferred, not saved to
`docs/ai_context/` yet.

## What changed this session (chronological, commits on `main`)

- `494c23b` → `e4af171` → `67ccfd7`: three consecutive attempts at fixing the collision
  collapse via reward/schedule changes. **All three tested and falsified or found
  insufficient** — see `EXPERIMENT_LOG.md` for the full data. Do not re-propose these without
  reading why they failed first.
- `ff104b4`: added `log_std_mean`/`mean_action_abs` instrumentation. This is the turning point
  of the whole session — it's what actually revealed the real cause instead of another guess.
- `0b6278d`: the closing-speed brake. **Solved collision** — `collision_rate=0.000` from here
  on, replicated across every subsequent config.
- `09e3038`: fixed a real bug (found via an external review, verified before acting on it) —
  `r_collision` had been silently logging `0.0` in every rollout of every run ever produced,
  due to a stale `env.agents` check reading state after it had already been mutated.
  Training itself was unaffected; only that one logged/plotted column was blind.
- `dc1de73` (on a `vision-tracking` branch, later merged): the full vision-based cooperative
  tracking redesign — see `ARCHITECTURE.md`/`DECISIONS.md` for the mechanism.
- `b59c139`: disabled the diameter floor (`DIAMETER_FLOOR_WEIGHT` → 0.0), which had become
  counterproductive once vision-tracking added a sensor-range constraint the floor was
  fighting. This is where `main` is now, after merging the `vision-tracking` branch in.
- Between `dc1de73` and merge: single-seed verification at 600k (floor active, then
  disabled), 1.2M (no floor), then a **3-seed, 3M-step validation** (no floor) that confirmed
  the system works well and reliably. All of this happened on the `vision-tracking` branch,
  which is now merged — `main` and `vision-tracking` point at the same commit (`b59c139`).

## Discoveries worth knowing

- **An external review of this repo made several claims; some held up, some didn't — verify,
  don't trust.** Real: the `r_collision` logging bug (confirmed and fixed, see above). Fair:
  a critique that some earlier commits (`e4af171`, `67ccfd7`) bundled multiple changes into
  one tested commit, weakening attribution. False: a claim that no runs had been tested
  against two specific commits — they had been, the review's evidence was just an incomplete
  local file listing (this session's own oversight — not all Kaggle run results were being
  copied into `stage/logs/` consistently; fixed by archiving everything found).
- **The Kaggle API works from this machine now.** `~/.kaggle-venv` has a working `kaggle` CLI
  (kaggle.json placed by the user outside this chat, never seen by the assistant). Kernels can
  be pushed, polled, and their output downloaded directly — used for every Kaggle run this
  session instead of manual notebook copy/paste. `kaggle kernels status`/`logs`/`output` all
  work correctly *once a kernel has actually been pushed at least once*; don't be alarmed by
  transient local DNS/network errors when polling — they look like remote failures in the
  error text (e.g. `NameResolutionError` contains the substring "Error") but aren't; re-check
  the specific kernel's status directly rather than trusting a poll loop's broad error match.
- **Track provenance by commit tag, not just filename pattern.** `stage/logs/`/`stage/models/`
  now contain many timestamped runs across many commits (pre-brake, brake-only, vision-
  tracking-with-floor, vision-tracking-without-floor, at 600k/1.2M/3M). The commit tag in each
  filename is load-bearing — don't compare across configs without checking it.
- **The closing-speed brake is not learned — confirmed and worth restating precisely.** It's a
  deterministic clamp applied to the network's raw output (`step()`, after `actions` come from
  the policy), zero learned parameters, can only subtract from commanded velocity, never
  redirect. Everything else (formation, tracking, coordination) is fully learned. Same pattern
  as ABS on a car layered under a human driver — a recognized "safety shield over a learned
  policy" technique in safe RL, not a workaround.
- **`N=4`'s target formation numbers are geometrically consistent, not contradictory — verified,
  don't re-litigate.** `TARGET_DIST` (4.78) and `EDGE_TARGET` (7.80) correspond exactly to a
  regular tetrahedron's circumradius and edge (`_PACKING_RATIO[4] = (8/3)**0.5` is exactly that
  ratio). What actually differs from `N=3` is that the tetrahedron is inherently non-planar
  while `r_spread` is horizontal-only — see `KNOWN_ISSUES.md` item 5.
- **The shared-actor architecture simplifies any future deployment work.** There is one `Actor`
  (one weights file), not one per drone — reused for every agent's observation, not four
  separate networks to keep track of. Relevant if/when a PX4/ROS2/Gazebo inference bridge gets
  built.

## Unresolved / pending as of this handoff

1. **`N=4` collision persistence — the real open problem now.** Confirmed, not theoretical (see
   above). Two fixes proposed (multi-pass brake convergence, direct brake-engagement reward
   penalty) — see `TODO.md` Phase 1 — **neither implemented yet**, pending confirmation before
   touching code.
2. **Two more proposed changes queued behind that (`TODO.md` Phase 2)**: an N-aware safety
   margin (`EDGE_TARGET` scaled by simultaneous-neighbor count) and a 3D/XYZ `r_spread`. Test
   these separately, after Phase 1, to preserve attribution — bundling untested changes has
   burned this project before (see `AI_CONTEXT.md`'s closing note).
3. **PX4/Gazebo/ROS2 SITL deployment — scoping in progress, not started.** The user has already
   built real infrastructure on their own machine (PX4-Autopilot SITL, Gazebo, ROS2 Jazzy,
   px4_msgs/px4_ros_com, Micro XRCE-DDS agent) and shared a deployment-architecture plan.
   Feedback was given (not yet saved to docs, per explicit request — will be documented once the
   user says so). Key technical notes for whoever picks this up: (a) the observation adapter
   needs to replicate `_get_obs()`'s vision-tracking state machine, not just raw
   position/velocity — treat Gazebo/PX4 as a physics layer under the *same* sensor-range
   abstraction the policy trained on, not a new real-perception problem; (b) the closing-speed
   brake is not part of the network — it must be reimplemented in the inference/adapter node
   explicitly, or the deployed swarm loses the one mechanism that solved collision at `N=3`;
   (c) action space is direct velocity (no inertia in training), maps to PX4 offboard velocity
   setpoints, but expect real degradation from the zero-inertia mismatch; (d) `K_NEIGHBORS=2` in
   the user's own notes is stale — it's `NUM_AGENTS-1=3` now, always re-derive from `config.py`;
   (e) start any demo with an `N=3` checkpoint, not `N=4`, given `N=4`'s open collision issue.
   Waiting on the user to share "what's on the PC" before scoping further.
4. **`readme.txt`** is more stale than ever — still not fixed, still out of scope unless
   explicitly requested.
5. **`envs/formation_env.py`'s module docstring** has one remaining stale paragraph (the
   pre-existing "4-agent study" SCOPE note) that a new docstring paragraph was added next to
   without fixing the old one. Low priority, flagged not fixed.

## Exact recommended next step for a new session

Check `git log`/`git status` to confirm `main` is still at `45b42a2` (or later, if more work has
happened since). If the user wants to proceed on the `N=4` collision problem, implement
`TODO.md` Phase 1 (multi-pass brake convergence + direct brake-engagement reward penalty)
together, test at `N=4` (that's where the problem is confirmed), and compare the same
training-time rolling-window data — not just eval-time numbers, which understated the problem
last time — against this run's baseline (`EXPERIMENT_LOG.md`'s `N=4` entry) before declaring it
fixed. If the user has instead moved on to the PX4/Gazebo/ROS2 deployment work, read their
system-state info first and scope against it rather than assuming what's installed — see
"Unresolved / pending" item 3 above for the technical notes already worked out.
