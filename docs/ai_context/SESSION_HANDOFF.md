# Session Handoff

**Read this file every new session — it's the most likely to be stale, and the most
important for not repeating work or missing what just happened.** Update it whenever a
session ends mid-task or reaches a natural checkpoint.

## Last updated

2026-08-27. Prior entry (2026-08-22) covered resolving the `N=4` collision problem, validating it
alongside two formation-quality fixes (`phase2-combined`), bundling five sustained-flight changes
(Phase 3) whose one serious failure (`target_lost_rate` near 100%) was diagnosed and improved by
active search, and reformulating the closing-speed brake with relative velocity in response to a
newly-discovered collision-safety cost under deterministic execution. **Since then, in brief**:
`main` was fast-forwarded to `phase2-combined` (no longer behind); the relative-velocity brake
was tested at full scale across 7 seeds and found to break training convergence for an
unidentified reason — **reverted**; a `NUM_AGENTS=3` deployment-training attempt failed under
both brake formulations; a network-capacity sweep confirmed Phase 3's hidden-width default is a
genuine sweet spot for `N=4` but found `N=3` doesn't learn *at all* regardless of size — a new,
unexplained deployment blocker; a deep supervisor-guided investigation traced `target_lost`
failures through a real-but-secondary critic-saturation pathology (tested and rejected as a fix)
to a decisive finding: **the root cause is the learned actor's own search execution, not the
environment** — proven by branching non-learned controllers from PPO's own real failure states
and watching them succeed 100% of the time where PPO fails 100% of the time. **The 2026-08-26
doc pass ended there; since then (this entry)**: the project moved from diagnosis to
intervention. `AUX_DIR_COEF=0.01` (a small actor-loss term nudging search direction toward
`unit(_last_known_vel)`) was implemented, smoke-tested, and validated in a matched 3-seed
comparison — `target_lost_rate` improved in **3/3 seeds at both final and best checkpoints**
(e.g. 0.83/0.86/0.98 → 0.26/0.14/0.38 final), the first intervention in the whole investigation
to move the primary metric rather than just localize it — and **adopted as the new default**. A
no-retraining audit of the new baseline found 26.0% `target_lost` remains, concentrated in late,
first-loss-event-fatal failures (67.9% of failures die on the episode's very first loss event).
Two follow-on interventions built directly from that audit were tested and **both rejected**:
`AUX_DIVERSIFY` (mixed: 1 up, 1 down, 1 flat) and `AUX_DIR_RAMP` (a clean, decisive 3/3-seed
regression). This doc-update pass ("document all," alongside a full project report) is the most
recent action.

## What happened this session (2026-08-26 through 2026-08-27, intervention phase)

**`AUX_DIR_COEF` implemented and validated.** Built from a chain of small, targeted diagnostics
run just before this: a separability check (logistic regression on the raw pre-correction
observation predicting sustained-positive-progress, AUC=0.834 vs. a 0.786 majority-class
baseline) confirmed the agent-observable state already contains exploitable information the
policy wasn't using, ruling out "not enough information" and justifying an actor-side
intervention. `AUX_DIR_COEF` adds a small auxiliary actor-loss term — active only during search,
gradient-connected to the actor's current mean action (required extending `Actor.evaluate()` and
`RolloutBuffer`) — nudging toward `unit(_last_known_vel)`. Smoke-tested at 0 (no-op), 1.0
(exaggerated, confirmed working end-to-end), and 0.01 (the sized treatment value) before any
Kaggle spend. Matched 3-seed comparison passed decisively at both final and best checkpoints —
see `EXPERIMENT_LOG.md`. Adopted; the user's explicit instruction after adoption was **not** to
launch another long training run to see if more steps would close the gap, but to audit the
existing data first.

**New-baseline failure-mode audit (zero retraining cost).** Pooled the 300 already-collected
`AUX_DIR_COEF=0.01` eval episodes: 72.3% success / 26.0% target_lost / 1.7% collision / 0.0%
ground_strike. Failed episodes are late (`episode_len` mean 1176.5/1800) and high-contact
(`contact_fraction` 0.865 even at failure) — genuine recovery failures, not early collapse — and
67.9% fail on the swarm's very first loss event of the episode. This reframed the remaining
problem as "ran out of time on the one search that mattered," which directly motivated the next
two interventions.

**`AUX_DIVERSIFY` and `AUX_DIR_RAMP`, both specified from this audit's evidence, both tested via
matched 3-seed comparisons, both rejected.** `AUX_DIVERSIFY` blended the shared directional cue
with each agent's own search heading (targeting the diversity hypothesis) — mixed result (1 up,
1 down +0.18, 1 flat), and `productive_agent_fraction` didn't move the way the hypothesis
predicted either. `AUX_DIR_RAMP` scaled the coefficient up as the timeout approached (targeting
the first-event-fatal finding) — a clean 3/3-seed regression, worse than the mixed result.
Leading hypothesis for the ramp's failure, not further tested: the pull target
(`unit(_last_known_vel)`) gets staler the longer contact stays lost, especially with Phase 3's
dynamic target — escalating commitment to it as urgency rises is backwards if its reliability is
actually decaying over the same window. Neither adopted; `AUX_DIR_COEF=0.01` (flat) remains the
baseline. See `EXPERIMENT_LOG.md`/`DECISIONS.md` for full detail on all three interventions.

**Process notes, consistent with established practice**: every launch this session re-confirmed
`git status` was up to date with `origin` immediately before pushing Kaggle kernels (per the
standing rule from the earlier active-search incident). The local `run_id=3` checkpoint-collision
hazard (reusing a `SEED` that still holds a previous unrelated local run's artifacts) recurred
twice more this session and was caught both times by checking `ps`/logs before trusting a launch
had started training, not assumed — the stale files were moved aside (preserved, not deleted)
each time before relaunching. Both negative results (`AUX_DIVERSIFY`, `AUX_DIR_RAMP`) were
reported plainly as failing the pre-agreed acceptance criterion rather than reframed positively,
consistent with this project's established anti-overclaiming discipline.

## What was happening (the condensed history, through 2026-08-22)

Two fixes (multi-pass POCS-style brake convergence; a thresholded brake-engagement reward
penalty) resolved the original `N=4` collision problem. Two formation-quality hypotheses
(N-aware safety margin; true-3D `r_spread`, after a geometry-conflation bug was caught by review
and fixed) were validated alongside it as `phase2-combined` — 3/3 `N=4` seeds clean at 0%
collision, the best tracking numbers of the project to that point. A sustained-flight diagnostic
then showed the best checkpoint's tracking error nearly quadrupling once flown 60s instead of its
10s training horizon, in a window matching where the real PX4/Gazebo deployment (a concurrent
conversation's work, sharing this worktree) actually crashed — real evidence, not a hunch. The
user made an explicit, informed call to bundle five sustained-flight changes at once (longer
episodes, a larger network, ground awareness, per-axis Z/XY speed limits, dynamic target motion,
`phase3-resilience`) rather than isolate each first. Ground awareness and per-axis dynamics
worked cleanly; `target_lost_rate` came back catastrophic (89-100%, flat from rollout 1) because
`LOST_TIMEOUT_SEC` (a flat 2s) was never rescaled for 9x-longer episodes. A 6/10/18s sweep found
no clean dose-response; **active search** (each agent fans out on its own heading when the swarm
loses contact, instead of coasting toward a stale shared point) was built instead and validated
as the real fix — `contact_fraction` ~16-21%→85-90.7% at 6s, up to 99%+ at 8-10s for seeds that
converge well. That same tight convergence then exposed a **new** collision-safety cost (8-14%)
invisible in training-time numbers but real at deterministic eval — traced to a policy's mean
action converging to a knife-edge equilibrium that training-time noise had been perturbing away
from. The closing-speed brake was reformulated with true relative velocity in direct response,
adversarially verified, and queued for Kaggle-scale validation — this is where the last handoff
ended. (Two process lessons from this era, still standing: confirm `git status` is up to date
with `origin` before every Kaggle launch, and a one-time autonomy grant covers exactly the round
it was given for, not follow-on rounds — both in `KNOWN_ISSUES.md`/persistent memory.)

## What happened this session (full detail — 2026-08-22 through 2026-08-26)

**`main` fast-forwarded.** `git push origin phase2-combined:main` finally ran clean (no
conflicts) — `origin/main` is now at `8b724bb`, no longer far behind. Several docs across this
suite had continued to claim "`main` still at `b59c139`" for three days after this actually
happened; corrected in this pass. Check `origin/main` directly rather than trusting a doc's
claim about it if this matters for what you're about to do.

**Relative-velocity brake: inconclusive at 6s, then reverted after full testing.** The 6s-timeout
validation batch came back inconclusive — all 3 seeds landed in the same bad-tracking regime the
plain-6s sweep already showed, never reaching the tight convergence where the original 8-14%
collision problem was ever observed, so nothing stressed the brake either way. Retested at 8-10s
(5 more seeds, 3 with direct old-brake baselines): **all 3 baseline-matched seeds flipped from a
clean 3/3 convergence record under the old brake to a clean 0/3 record under the new one** — not
the project's usual bimodal seed-variance, a genuine reversal on repeat. 6 of 7 total seeds
failed to converge, including both `NUM_AGENTS=3` deployment-target attempts (99%/96%
`target_lost`). The leading hypothesis (the new formula reads ~2x the closing speed and
penalizes/engages harder) was tested directly against a matched pair and **refuted** — the failed
run's brake barely engaged at all, because the swarm never converged close enough to need it. The
actual mechanism was never found. Given deployment-readiness was the more urgent priority (no
`N=3` checkpoint existed either way) and the new brake had no confirmed safety benefit to weigh
against its convergence cost, it was **reverted to the original one-sided formula** (commit
`55e7232`) rather than continuing to chase an unexplained effect. `test_brake_relative_velocity.py`
stays as a general regression test since both its scenarios are symmetric and pass under either
formula.

**Network-capacity sweep, bundled with the revert.** `ACTOR_HIDDEN`/`CRITIC_HIDDEN` were made
env-overridable in the same commit as the revert, and a 5-kernel sweep launched immediately:
3x `N=4` (64/128, 128/256, 256/512) and, for the first time on a *working* brake, 2x `N=3`
(128/256, 256/512). Result: `N=4`'s Phase 3 default (128/256) is a genuine sweet spot — smaller
partially hurts (45% `target_lost`), larger collapses learning almost entirely (97%
`target_lost`, flat from rollout 1, the same shape as a badly broken run, not a slow-converging
one). **`N=3` failed completely at both sizes tested** — flat ~100% `target_lost_rate` from the
very first rollout. This is a new, still-unexplained deployment blocker: `deployment/
inference_node.py` hard-requires `NUM_AGENTS=3`, and there is currently no Phase-3-era checkpoint
at that agent count that learns anything at all, under either brake formulation or either network
size tried.

**The critic-collapse investigation.** A supervisor-guided review noticed something suspicious:
active search's `target_lost` failures (79-100%) persisted even though a scripted, non-learned
controller solves the identical forced-loss task 96-100% of the time using the same search
mechanism. A chain of local diagnostics followed: forced-loss masking (confirmed the search
mechanism itself is usable — rules out "the mechanism is broken"), PPO-vs-scripted action/reward
instrumentation on a real checkpoint (PPO's actions diverge substantially from scripted, but
one-step rewards are nearly identical — points at credit assignment, not reward shaping), then a
value/critic diagnostic that found the production critic's value predictions are **essentially
constant regardless of state** across 90 real loss-states. Traced to the critic's final Tanh
layer saturating 100% within the first rollout of training (confirmed via from-scratch
reproduction), not specific to any one layer position, not caused by reward heavy-tails
(loss-function-invariant), but controlled by critic learning rate: `CRITIC_LR=1e-5` (30x smaller)
fully prevented saturation over a short reproduction. A corrected, properly stratified diagnostic
(the first attempt used a temporally-contiguous reference batch, a sampling artifact caught
before being trusted) confirmed the earlier-measured critic/return anti-correlation was
substantially a sampling artifact, not a severe real effect.

**The real CRITIC_LR experiment — rejected, not simply "failed."** `CRITIC_LR` was added as a
separate, env-overridable optimizer LR (commit `71327be`, defaults to the actor's `LR`, so
default behavior is unchanged) along with per-rollout critic-health diagnostics in
`training_log.csv`. **A real bug was caught before any Kaggle spend**: the pre-existing LR-anneal
block unconditionally overwrote `CRITIC_LR` back to an `LR`-derived value every rollout — found
via a smoke test producing bit-identical output under two different `CRITIC_LR` settings, fixed,
reverified. A matched 3-seed comparison (`CRITIC_LR` unset vs. `1e-5`, following the user's
explicit "whenever it's better to use Kaggle do it, keep it local if simple and quick" rule — 5
seeds on Kaggle, treatment-seed3 run locally since local throughput measured faster) found
saturation **delayed, not prevented** (all 3 treatment seeds reached 100% saturation again by the
end of the run), and behavior was a mixed bag: one seed traded a real tracking improvement for a
new collision cost, two showed no improvement or regression, and `mean_reacquisition_steps` was
worse under the treatment in all three pairs — the single most consistent effect in the dataset,
pointing against it. Per the user's own framing: **"rejected as a general solution;
seed-dependent behavioral redistribution observed" — not simply "failed."** `CRITIC_LR`'s default
stays unchanged; the diagnostic columns stay as useful instrumentation regardless.

**Actor search-action dynamics, in increasing precision.** With the critic demoted to a secondary
pathology, investigation moved to the actor's own behavior during search, in four escalating
passes, all against real natural loss events from `search_v2_seed1`/`actor_1.pt` (no
environment/training changes, read-only instrumentation throughout):
1. Per-(event, step, agent) action dynamics: scripted's own action persistence is *higher* than
   PPO's (0.999 vs 0.906-0.930) — the "PPO freezes" hypothesis doesn't hold. Per-step progress
   toward the sensor boundary is negative on average in both outcomes but ~4x worse in failures.
   One fully-detailed failure case: one agent closing correctly and steadily, ~30 steps from
   success, while three teammates drifted the wrong way the entire event — not oscillation, not
   frozen, confidently wrong headings that never get corrected.
2. Best-agent/productive-agent reanalysis (correcting a pooled-4-agent-mean caveat from pass 1):
   reacquired events average **2.25** productive agents (0% have zero); target_lost events
   average **0.41** (**70.6% have zero**) — the cleanest single signature in the whole
   investigation. Confidence-response: the actor's mean action direction changes sharply
   entering search mode (confidence high→mid, cosine=0.198) but barely adapts further as things
   get worse (mid→low, cosine=0.922) — it reacts once, then stops adjusting.
3. A scripted-controller A/B/C search-strategy comparison (fixed heading / last-known-velocity /
   adaptive reassignment) plus a 6/8/10/12s timeout-margin sweep, both forced-loss from a
   well-formed warmup: **every condition hit a 100% reacquisition ceiling.** Rules out heading
   strategy and timeout length as standalone bottlenecks when execution is already good, but is
   structurally blind to PPO's own worse-formed states — motivates the final experiment.
4. **The decisive experiment**: branched PPO / scripted / adaptive controllers from the *exact*
   states (via `copy.deepcopy(env)`, full RNG-included snapshot) PPO's own policy produced at 28
   real loss-onset moments — not an artificial mask. From the 16 states where PPO's real
   trajectory actually failed, **PPO replayed from that state fails again 100% of the time, but
   both scripted and adaptive reacquire from every single one (100%)**, typically within 1-2
   steps, at distances right at the sensor boundary (not hard states). This holds environment,
   geometry, timeout, and starting state exactly fixed — only the controller differs — and
   resolves the entire investigation's central question: **the problem is the learned actor's own
   search execution.**

All four passes and the network-capacity sweep are written up in full in `EXPERIMENT_LOG.md`;
this doc-update pass is what added them there.

## What changed this session (chronological, key commits since the last handoff's `3eae55b`)

- `2e7a826`: the last full "document all" pass (already covered by the prior handoff entry).
- `eedda7e`: recorded the 6s brake-relvel batch as inconclusive.
- `6a03bdd`: recorded the `main` fast-forward, the 8s/10s brake-relvel relaunch, and a
  `deployment/` diagnostic-trio check.
- `5941364`: recorded the `NUM_AGENTS=3` deployment-training launch and first brake-relvel
  t8/t10 partial results (the seed-reuse-isn't-a-clean-comparison finding).
- `97ffc23`: recorded the full brake-relvel result — 6/7 seeds failed, `r_brake` hypothesis
  refuted, no deployment-ready checkpoint exists.
- `55e7232`: **reverted** the relative-velocity brake; added `ACTOR_HIDDEN`/`CRITIC_HIDDEN` env
  overrides for the network-capacity sweep.
- `6b31baf`: recorded the revert and the network-capacity sweep launch.
- `71327be`: added `CRITIC_LR` override + per-rollout critic-health diagnostics (the critic-LR
  ablation experiment itself, and the entire actor-search-dynamics investigation chain, were run
  from local/Kaggle scratch scripts and analysis, not further commits to production code).
- This doc-update pass itself (uncommitted as of this writing) — updates all 11
  `docs/ai_context/` files to reflect everything above.

## Discoveries worth knowing

- **A real, measured, reproducible mechanism (critic saturation) does not automatically mean
  fixing it is the dominant behavioral lever.** The critic-LR ablation is the concrete case that
  taught this distinction — every diagnostic confirmed the mechanism was real, and fixing it
  still didn't reliably improve behavior. Don't skip the full-scale behavioral test just because
  a mechanism diagnostic came back clean.
- **A scripted-controller ceiling effect can hide the exact thing you're trying to measure.** The
  search-strategy/timeout sweep's 100%-across-the-board result initially looks like "nothing
  matters," but it actually means "not from an easy starting state" — branching from the
  learned policy's own real (harder, worse-formed) states was what actually distinguished the
  hypotheses. When a diagnostic saturates at a ceiling, check whether it's testing the state
  distribution that actually matters before concluding the variable doesn't matter.
- **Pooled-mean metrics across cooperating agents can hide the real signature.** The original
  action-dynamics pass's 4-agent pooled progress numbers were far less informative than the
  best-agent/productive-agent-count reframing that came right after — worth defaulting to a
  "does at least one agent succeed" framing for any swarm task where one success is enough,
  rather than an average that dilutes the one agent that mattered.
- **A doc claiming "`main` is far behind" was stale for three days after the push actually
  happened.** `git status`/`git log` on the actual ref beats trusting a doc's claim about
  cross-branch state, even a recently-updated one — worth spot-checking this kind of claim
  directly when it matters for what you're about to do, not just when something looks obviously
  wrong.
- **Network capacity sweeps can produce a "never learns" failure mode that looks identical
  whether the cause is "too big" or "wrong agent count."** `n4-large` (256/512) and both `n3`
  attempts (128/256, 256/512) all show the identical flat-from-rollout-1 shape — worth explicitly
  checking the training curve's *shape* (flat from step 1 vs. a mid-training regression), not
  just its final value, before assuming two failures share a cause.
- Carried forward, still true: unpushed commits can silently invalidate a Kaggle batch (confirm
  `git status` before every launch); deterministic eval, not training-time rolling numbers, is
  the trusted safety signal; a confidently-stated causal explanation can still be wrong even when
  nothing about it "sounds off" — re-verify against the specific numbers cited, including your
  own prior statements in the same session.

## Unresolved / pending as of this handoff

1. **The actor-search-execution problem — still the most important open question in the
   project, now with a partial fix and two rejected follow-ons.** `AUX_DIR_COEF=0.01` (search-
   direction auxiliary loss) closed part of the gap (3/3 seeds improved, e.g. target_lost_rate
   0.83/0.86/0.98 → 0.26/0.14/0.38 final) and is adopted. 26% target_lost remains, concentrated in
   late, first-loss-event-fatal failures. Two follow-ons built from that evidence —
   `AUX_DIVERSIFY` (heading diversity) and `AUX_DIR_RAMP` (urgency-scaled coefficient) — were both
   tested and both rejected (mixed, then a clean 3/3 regression). **No further fix direction has
   been chosen** for the remaining 26% — this is an open question for the user, not a decided next
   step. Whoever picks this up should read `EXPERIMENT_LOG.md`'s three AUX-intervention entries
   before proposing a fourth idea, since the two most directly evidence-motivated ones from the
   failure-mode audit have already been tried.
2. **`NUM_AGENTS=3` cannot learn at all under Phase 3's mechanisms, at any tested network size.**
   A real deployment blocker, not yet under active investigation. Untried: a smaller network at
   `N=3` specifically (only 128/256 and 256/512 were tested), more training steps, or isolating
   which single Phase 3 mechanism (if any one) is responsible by testing `N=3` against the
   pre-Phase-3 mechanism set (which is known to have worked).
3. **The collision-safety cost under deterministic execution (8-14%) is still real and still
   unaddressed.** The fix built for it was reverted for unrelated reasons (it broke convergence);
   the underlying safety question itself was never actually resolved either way.
4. **Vertical-jiggling fix still unverified** — implemented since Phase 3, still not re-measured
   against any trained checkpoint across now several doc passes.
5. **`readme.txt`** and **`envs/formation_env.py`'s stale "4-agent study" docstring paragraph** —
   still not fixed, still low priority, flagged across several doc passes now.
6. **`PHASE2_CHECKPOINT.md`** (repo root, shared coordination doc with the deployment
   conversation) is up to date as of this pass, including the decisive actor-localization
   finding (which was not yet written into `EXPERIMENT_LOG.md` when that checkpoint entry was
   made — it is now, as of this doc pass).

## Exact recommended next step for a new session

Still a decision point for the user, not a mechanical next step — now with more evidence about
what *doesn't* work. `AUX_DIR_COEF=0.01` is adopted and real; the two most directly evidence-
motivated follow-ons (diversify the pull direction, ramp its strength with urgency) have both been
tried and rejected. If picking this up cold: first read `EXPERIMENT_LOG.md`'s three AUX-
intervention entries (in order) to see what's already been ruled out, then either (a) propose a
genuinely different mechanism for the remaining 26% (not a variant of the shared-direction pull
already tested twice) and discuss before implementing, or (b) if the user wants to prioritize
deployment-readiness instead, pivot to item 2 above (`N=3` won't learn at all) since that's the
more concrete blocker for actually flying a Phase-3-era checkpoint. Check `PHASE2_CHECKPOINT.md`
first if the deployment thread has moved in the meantime — this worktree is still shared.
