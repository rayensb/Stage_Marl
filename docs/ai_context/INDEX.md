# AI Context Index

This directory is a durable knowledge base for AI assistants (and humans) picking up this
project across sessions. **The actual code in the repo root is the source of truth.** These
documents explain, contextualize, and preserve reasoning that isn't visible from the code
alone (why a value is what it is, what was tried and rejected, what's still open).

Written 2026-08-14, reflecting commit `b14fe48` on `main`. If the repo has moved on
significantly since then, treat anything here that contradicts the current code as stale —
the code wins, and the discrepancy is worth noting back into `KNOWN_ISSUES.md` or `CURRENT_STATE.md`.

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
