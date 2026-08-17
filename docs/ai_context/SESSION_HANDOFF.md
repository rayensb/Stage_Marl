# Session Handoff

**Read this file every new session — it's the most likely to be stale, and the most
important for not repeating work or missing what just happened.** Update it whenever a
session ends mid-task or reaches a natural checkpoint.

## Last updated

2026-08-17, end of a long session that solved both the collision problem and (separately) the
target-tracking problem, replaced ground-truth target sensing with a vision-based cooperative
system, validated the result at `NUM_AGENTS=3` across 3 seeds at 3M steps, and merged
everything to `main` (commit `b59c139`).

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
session. Merged to `main`, then this file (and the rest of `docs/ai_context/`) was brought up
to date on explicit request ("document all").

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

## Unresolved / pending as of this handoff

1. **`NUM_AGENTS=4` has never been run against the current code.** This is the clear next
   step — the whole reason the collision and tracking problems needed solving in the first
   place was to eventually validate at the target agent count. A launcher script pattern
   (parallel per-seed Kaggle kernels, `OMP_NUM_THREADS=1` etc.) has been used all session and
   is ready to reuse with `NUM_AGENTS=4` (or left unset, since 4 is `config.py`'s default).
2. **The brake's rare multi-agent collision edge case** (~1% observed at `N=3` over long
   runs) — worth explicit attention at `N=4`, where it may be more frequent. Not necessarily
   blocking, but shouldn't be assumed away either.
3. **`readme.txt`** is more stale than ever — still not fixed, still out of scope unless
   explicitly requested.
4. **`envs/formation_env.py`'s module docstring** has one remaining stale paragraph (the
   pre-existing "4-agent study" SCOPE note) that a new docstring paragraph was added next to
   without fixing the old one. Low priority, flagged not fixed.

## Exact recommended next step for a new session

Check `git log`/`git status` to confirm `main` is still at `b59c139` (or later, if more work
has happened since). If `NUM_AGENTS=4` hasn't been run yet, that's the next real work item —
launch it the same way every other Kaggle run this session was launched (see `COMMANDS.md`),
using the current `main` branch directly (no feature branch needed this time — everything's
already merged). Analyze collision_rate, target_lost_rate, and tracking_rmse the same way
every prior run this session was analyzed, and specifically check whether the brake's rare
collision edge case (item 2 above) shows up more often at this agent count.
