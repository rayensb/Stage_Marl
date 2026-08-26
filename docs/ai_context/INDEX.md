# AI Context Index

This directory is a durable knowledge base for AI assistants (and humans) picking up this
project across sessions. **The actual code in the repo root is the source of truth.** These
documents explain, contextualize, and preserve reasoning that isn't visible from the code
alone (why a value is what it is, what was tried and rejected, what's still open).

Written 2026-08-14, reflecting commit `b14fe48` on `main`; substantially updated 2026-08-17
to reflect commit `b59c139` — a session that solved the collision problem (a deterministic
action-space safety layer, not reward shaping) and separately replaced ground-truth target
sensing with a vision-based cooperative tracking system, both validated at `NUM_AGENTS=3`.
If the repo has moved on significantly since then, treat anything here that contradicts the
current code as stale — the code wins, and the discrepancy is worth noting back into
`KNOWN_ISSUES.md` or `CURRENT_STATE.md`.

Updated 2026-08-19 (still commit `45b42a2`, no code changed): `NUM_AGENTS=4` was validated and
found to need more work — tracking generalizes, collision doesn't yet.

**Updated again 2026-08-20, now against commit `62a685d` on `phase3-resilience` (`main` is
still at `b59c139` — see `TODO.md`): the `N=4` collision problem from the previous update is
now resolved.** Since then: two more formation-quality fixes were validated alongside it
(`phase2-combined`), a sustained-flight diagnostic motivated a 5-change bundle for longer
missions (`phase3-resilience`), and that bundle's one serious failure (near-total target
tracking loss over longer episodes) is diagnosed with a fix currently being swept on Kaggle.
**Also new since the last update**: `deployment/` now contains a real PX4/Gazebo/ROS2
inference pipeline, built by a separate, concurrent conversation sharing this worktree — its
own documentation lives in `deployment/docs/` and is deliberately **not** part of this index
or this suite's scope; see `ARCHITECTURE.md`'s `deployment/` section for how the two connect.

**Updated again 2026-08-22, now against commit `94216fd`/`3eae55b` on `phase3-resilience`**:
the plain `LOST_TIMEOUT_SEC` sweep found no clean dose-response; **active search** was built and
validated as a real, major fix for `target_lost_rate` (though not a complete one); testing it at
a longer timeout surfaced a **new** collision-safety cost under deterministic execution, which
was investigated, explained (a policy's mean action can converge to a knife-edge equilibrium
training-time noise never sat at — deterministic eval, not training-time numbers, is now the
trusted safety signal), and addressed by reformulating the closing-speed brake with true
relative velocity — adversarially verified, Kaggle-scale validation in progress. Two stuck
seeds were resumed to 5M steps to test "does more training help" (partially yes). See
`SESSION_HANDOFF.md` first for the current picture; `KNOWN_ISSUES.md` items 12/13/16/17 and
`TODO.md` Critical for the specifics of what's still open.

**Updated again 2026-08-26, now against commit `71327be` on `phase3-resilience`**: `main` was
fast-forwarded (no longer behind, correcting a stale claim that had persisted three days past
the actual push). The relative-velocity brake from the previous update was tested at full scale
and found to **break training convergence** for a reason never identified — **reverted**. A
network-capacity sweep confirmed `N=4`'s hidden-width default is a genuine sweet spot but found
**`NUM_AGENTS=3` — the real deployment target — cannot learn tracking at all**, at any tested
size, a new and still-unexplained blocker. A deep investigation (critic-collapse diagnostics,
finding a real but non-dominant critic-saturation pathology; then actor search-action dynamics,
finding a clean productive-agent-count signature and a confidence-response saturation) culminated
in a **decisive experiment**: branching non-learned controllers from PPO's own real failure
states reacquires the target 100% of the time from states where PPO's own trajectory failed 100%
of the time — **the root cause of `target_lost` is the learned actor's own search execution, not
the environment.** No fix has been designed yet; this is now the project's single most important
open problem. See `SESSION_HANDOFF.md` first; `KNOWN_ISSUES.md` items 12/13/18/19 and `TODO.md`
Critical for specifics.

## Read order for a new session

1. **`AI_CONTEXT.md`** — start here. What this is, current status, what to read next.
2. **`SESSION_HANDOFF.md`** — what was happening right before this session ended, and the
   exact next step. Read this second, always — it's the most likely to be stale but also
   the most important for not repeating work.
3. **`CURRENT_STATE.md`** — what works, what's unverified, what's broken, right now.
4. **`ARCHITECTURE.md`** — how the system is put together, if you need to change code.
5. **`DECISIONS.md`** — before proposing a redesign of anything, check here — it's very
   likely already been tried, and the reasoning for the current approach is recorded.
6. **`KNOWN_ISSUES.md`** — open problems, with what's already been ruled out.
7. **`EXPERIMENT_LOG.md`** — the actual numbers behind claims made in `DECISIONS.md`. Read
   when you need evidence, not just conclusions.
8. **`TODO.md`** — prioritized roadmap.
9. **`ENVIRONMENT.md`** / **`COMMANDS.md`** — reference, when you need to actually run something.

## What NOT to do

- Don't re-propose fixes already covered in `DECISIONS.md`/`KNOWN_ISSUES.md` without reading
  why they were rejected or deferred first — several plausible-looking ideas have already
  been tried and measured to make things worse (see `EXPERIMENT_LOG.md`).
- Don't trust an external review/critique of this code at face value — this project has
  received several, and multiple times a claimed bug turned out to be based on a stale
  snapshot or a misreading (verified by grep against the actual file each time). Verify
  claims against the real file before acting on them, and say so either way.
- Don't copy large code blocks into these docs going forward — reference `file:line` instead
  so this doesn't rot out of sync with the code.
