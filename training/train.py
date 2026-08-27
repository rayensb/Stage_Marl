import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from scipy.stats import spearmanr

from envs.formation_env import FormationEnv3D
from training.networks import Actor, CentralCritic
from training.buffer import RolloutBuffer
from training.checkpoint import save_checkpoint, load_checkpoint, save_best_actor
from training.logger import init_logger, log_row
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, MAX_STEPS, ACTOR_HIDDEN, CRITIC_HIDDEN, LOST_TIMEOUT_STEPS

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]
# Phase 3 (2026-08-20): derived from MAX_STEPS instead of hardcoded
# independently -- was a flat 2048, which gave ~10 episodes/rollout at the
# old MAX_STEPS=200, but only ~1 at the new MAX_STEPS=1800, which would
# have hurt PPO's per-update sample diversity (each rollout mostly one
# long correlated trajectory instead of several independent resets).
# ROLLOUT_EPISODES keeps that "episodes per rollout" ratio explicit and
# self-maintaining across future MAX_STEPS changes.
ROLLOUT_EPISODES = 8
ROLLOUT_LEN = MAX_STEPS * ROLLOUT_EPISODES
EPOCHS = 10
BATCH_SIZE = 256
LR = 3e-4
# Critic-LR recovery experiment (2026-08-25): the critic-collapse
# investigation (see PHASE2_CHECKPOINT.md / EXPERIMENT_LOG.md) found the
# production critic's second Tanh layer saturates 100% within the first
# rollout of training at CRITIC_LR=LR, and that a per-update trajectory +
# LR sweep on a short (2250-update) reproduction showed this is
# LR-dependent -- 1e-5 (30x smaller) fully prevented saturation over that
# same budget, where 3e-4/1e-4 still reached 100%. A corrected, properly
# stratified (not temporally-contiguous) diagnostic then found the
# critic's anti-correlation with real returns shrinks toward zero as LR
# drops (-0.263 at 3e-4 -> -0.042 at 1e-5, statistically indistinguishable
# from noise at n=896), though still far from confirmed-positive at that
# short budget. This is deliberately kept SEPARATE from the actor's LR --
# only the critic's optimizer step size is in question, not policy
# learning rate -- so this is its own env-var-overridable constant,
# defaulting to the unchanged, original behavior (same value as LR)
# unless explicitly set. Override with CRITIC_LR=1e-5 for the treatment
# arm of the matched-seed comparison this was built for; do not change
# this default without the comparison's own result in hand.
CRITIC_LR = float(os.environ.get("CRITIC_LR", LR))

# Search-direction auxiliary objective (2026-08-26): follow-on to the
# actor-search-dynamics investigation (see PHASE2_CHECKPOINT.md /
# EXPERIMENT_LOG.md). A correction-quality diagnostic on natural loss
# events found PPO's directional corrections beat a same-state random
# rotation 61-62% of the time (real, non-random judgment) but only
# converted to sustained positive progress 3.3-4.5% of the time for
# agents that stayed unproductive, vs 61.6-69.8% for agents that
# eventually became productive -- weak but genuine directional judgment,
# not "no objective at all." A quick separability check (logistic
# regression on the raw pre-correction observation, AUC=0.834 vs a 0.5
# chance/0.786 majority-class baseline) confirmed the agent-observable
# state already contains substantial information the policy isn't fully
# exploiting -- ruling out "the observation doesn't contain enough
# information" as the explanation. This adds a small auxiliary actor loss
# term, active only during search (steps_since_contact > 0), encouraging
# the actor's current mean action direction toward unit(_last_known_vel)
# -- a cue already shown to correlate with PPO's own successful corrections
# (0.292 vs 0.175 pooled, 0.410 vs 0.256 on matched failure states) and
# available to the agent without touching ground truth. Deliberately NOT
# a reward-shaping term (the credit-assignment diagnostic found the
# analogous ground-truth-free reward proxy has ~44.8% sign mismatch with
# true progress -- too unreliable to reward) and NOT imitation of the
# scripted controller (no hard-coded action target, only a directional
# nudge the policy can override).
#
# RESULT (2026-08-27): matched 3-seed Kaggle comparison (control=unset vs
# treatment=0.01) -- target_lost_rate, contact_fraction, success_rate, and
# productive_agent_fraction all improved in 3/3 seeds at BOTH final and
# best checkpoints (e.g. target_lost_rate 0.83/0.86/0.98 -> 0.26/0.14/0.38
# final; 0.79/0.95/0.98 -> 0.26/0.14/0.56 best). collision_rate rose
# slightly in 2/3 seeds (by 1-3 points) -- real but small next to the
# 8-14% costs seen elsewhere in this project, accepted rather than
# chased further. This is the first change in the whole active-search/
# target_lost investigation that measurably improved the primary failure
# metric, not just localized it -- see EXPERIMENT_LOG.md's "Search-
# direction auxiliary objective" entry and DECISIONS.md for the full
# per-seed/per-checkpoint tables and the adoption reasoning.
# ADOPTED as the default (0.0 -> 0.01). Per explicit decision, no
# coefficient sweep or further diagnosis is planned -- override to 0.0 to
# recover the old (pre-2026-08-27) behavior for comparison/debugging.
AUX_DIR_COEF = float(os.environ.get("AUX_DIR_COEF", 0.01))

# AUX_DIVERSIFY (2026-08-27): targeted follow-on now that AUX_DIR_COEF is
# the baseline. A fresh baseline evaluation (3 seeds, current committed
# code) still shows target_lost_rate=26.0% -- 15x more common than the
# next failure mode (collision, 1.7%) -- with 67.9% of those failures
# fatal on the swarm's very first loss event of the episode, despite
# contact_fraction ~0.87 even in the failed episodes (see PHASE2_CHECKPOINT.md).
# Specified from existing evidence, no new diagnostic run: (1) as built,
# AUX_DIR_COEF pulls EVERY agent toward the identical unit(_last_known_vel)
# direction every search step (see aux_dir_dict below); (2) active search's
# own design (_assign_search_directions, DECISIONS.md) deliberately spreads
# agents across DIFFERENT headings so the swarm covers more area, not one;
# (3) the single most robust signal in this whole investigation, checked
# both before and after AUX_DIR_COEF, is that productive-agent count/
# diversity is what separates success from failure (pre-AUX: reacquired
# events averaged 2.25 productive agents vs 0.41 for target_lost; post-AUX,
# productive_agent_fraction implies ~1.0-1.6 of 4, better but still well
# short of 2.25). Putting (1)-(3) together: pulling every agent toward the
# same direction may be trading away exactly the coverage diversity that
# (3) says still matters, capping how many independent chances the swarm
# gets per event. When enabled, blends unit(_last_known_vel) with each
# agent's OWN already-diverse _search_dirs[agent] (plain vector sum, then
# normalize -- no new tunable weight) instead of using the swarm-uniform
# direction. AUX_DIR_COEF's magnitude, the search-mode gate, and everything
# else stay exactly as validated -- this isolates one variable (uniform vs.
# per-agent-diverse target direction) against the CURRENT baseline, not the
# pre-AUX one. Defaults to 0 (current, validated behavior unchanged).
#
# RESULT (2026-08-27): matched 3-seed comparison -- 1 seed improved, 1 seed
# regressed clearly (target_lost_rate +0.18, the largest single movement in
# the table), 1 was flat. productive_agent_fraction did not improve
# consistently either (flat-to-down in 2/3 seeds) -- the specific
# hypothesis (uniform direction trades away useful coverage diversity)
# isn't supported by this implementation. REJECTED as a general fix; does
# not meet the pre-agreed >=2/3-seeds bar. Does not invalidate AUX_DIR_COEF
# itself (that was a clean 3/3 win) -- only this specific diversification
# formulation. Left at its default (off); no further diagnosis planned on
# this specific hypothesis. See EXPERIMENT_LOG.md/PHASE2_CHECKPOINT.md.
AUX_DIVERSIFY = bool(int(os.environ.get("AUX_DIVERSIFY", 0)))

# AUX_DIR_RAMP (2026-08-27): different kind of follow-on to AUX_DIR_COEF --
# not a different target direction (AUX_DIVERSIFY tried that, rejected
# above), but a different STRENGTH schedule for the same, already-validated
# direction (unit(_last_known_vel)). Specified from existing evidence, no
# new diagnostic run: (1) r_contact's own urgency ramp
# (CONTACT_URGENT_COEF * min(1, steps_since_contact/LOST_TIMEOUT_STEPS),
# see envs/formation_env.py) is an already-validated pattern in this exact
# codebase for "increase pressure as the timeout approaches" -- this reuses
# that shape, not a new idea; (2) the directional-entrenchment diagnostic
# found correction PROBABILITY does not rise on its own as a bad streak
# lengthens (flat ~5.5-6.8% for streaks of 6-10/11-20/21+ steps, after an
# initial 1-5-step spike) -- the actor doesn't self-generate escalating
# urgency, so an explicit signal that does could compensate; (3) the fresh
# post-AUX_DIR_COEF baseline audit found 67.9% of remaining target_lost
# failures are fatal on the swarm's very first loss event of the episode,
# with contact_fraction still ~0.865 even in failed episodes -- these are
# "ran out of time on the one search that mattered" failures, exactly where
# a pull that gets more assertive as time runs low has the most leverage.
# When set > 0, the per-transition auxiliary coefficient is
# AUX_DIR_COEF * (1 + AUX_DIR_RAMP * min(1, steps_since_contact/
# LOST_TIMEOUT_STEPS)) instead of a flat AUX_DIR_COEF for the whole search
# window -- see the minibatch loop below. Defaults to 0.0 (current,
# validated flat-coefficient behavior exactly reproduced -- see the
# aux_ramp_frac/coef_vals computation below for why this is an exact
# no-op at 0).
AUX_DIR_RAMP = float(os.environ.get("AUX_DIR_RAMP", 0.0))
CLIP = 0.2
GAMMA = 0.99
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001    # back to the original 10x decay -- the 0.004/3M-step
                          # experiment was a clear, decisive failure (see
                          # ENTROPY_RECOVERY below for what replaced this
                          # approach): collision_rate plateaued at 0.78 for
                          # the last 2M steps, worse than any prior run, and
                          # entropy still collapsed to -3.92 anyway. Raising
                          # the floor coefficient didn't stop the collapse,
                          # it just gave the policy more time to commit
                          # harder to trading safety for precise tracking --
                          # the reward genuinely permits that trade, so a
                          # fixed-calendar entropy schedule (of any shape)
                          # can't fix it by itself.
TARGET_KL = 0.02   # standard PPO trust-region safety net (SB3/CleanRL default
                     # range ~0.01-0.03) -- stop the epoch loop early if an
                     # update has already moved the policy this far, instead
                     # of blindly running all EPOCHS regardless of update size.
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 600_000))
# Was a flat 600_000 -- made env-var overridable (2026-08-14) so a longer-
# training test (e.g. TOTAL_STEPS=1200000) can be run from the same commit
# instead of a hardcode-then-revert cycle, matching how SEED/DEVICE/NUM_AGENTS
# already work. Note this reruns the earlier-rejected "more steps" experiment
# under a materially different setup than the one that failed (see the
# ENT_COEF_END comment above): that 3M-step run also raised the entropy floor
# at the same time, so it was never a clean isolated test of duration alone.
# This time only TOTAL_STEPS changes. Caveat: RECOVERY_PATIENCE/MAX_TRIGGERS
# below are calibrated for a 600k-step run's rollout count -- at 1.2M steps
# recovery coverage will still only span roughly the first half, so a bad
# result on that variant doesn't cleanly isolate "duration alone doesn't
# help" from "recovery still ran out." Flagged, not silently fixed, since
# rescaling those too would be a bigger decision than what was asked for.
# Default 600k -- 3M was only ever a test of the entropy floor above, not a
# new baseline.

# Plateau-triggered entropy recovery (same idea as ReduceLROnPlateau, but
# boosting exploration instead of cutting LR): the entropy<0 trigger tried
# before was too strict -- the NUM_AGENTS=3/shared-actor run relapsed
# (collision_rate 0.02 -> 0.25+) without entropy ever going negative, so
# that check never fired even though the run needed it. This instead
# watches the same collision_rate rolling window (last <=50 episodes)
# everything else uses, and tracks whether it's actually improving:
#   - new best (with a real margin, not noise) -> save it, reset patience
#   - no new best for RECOVERY_PATIENCE rollouts -> boost exploration for
#     RECOVERY_COOLDOWN rollouts, then resume watching
# The cooldown is what prevents "triggered multiple times and explode":
# once boosted, we don't re-judge for a full cooldown window, so the boost
# gets a real chance to work before being re-evaluated, and can't fire
# again immediately if the very next rollout still looks bad.
# RECOVERY_MAX_TRIGGERS caps total interventions so a run that's simply
# stuck can't loop forever -- it lets the schedule finish normally after
# that many honest attempts instead.
#
# History: raised 5 -> 20 (2026-08-14), tested, falsified -- 3-seed re-run
# gave 27%/25%/66% collision_rate at TOTAL_STEPS, no better than pre-fix
# 30%/26%/46%, even though recovery visibly fired across the whole run.
# Reverted to 5, then tested the r_safety-reshape hypothesis instead (also
# didn't fix it -- see envs/formation_env.py's r_safety comment and
# KNOWN_ISSUES.md). Current leading picture: r_track has no saturation, so it
# keeps rewarding more precision (= lower entropy) for the entire run
# regardless of what the entropy-coefficient side does -- see MIN_DIAMETER in
# config.py for the fix aimed at that directly, on a different axis.
#
# RECOVERY_PATIENCE retimed 15 -> 29 (2026-08-14) for a different reason: once
# best_collision_rate locks at its floor (observed happening by rollout ~40 in
# every run so far, since collision_rate can't improve past 0), since_best
# stops ever resetting except at a trigger, so the mechanism free-runs
# periodically with period == RECOVERY_PATIENCE (confirmed exactly: the last
# run's 5 triggers landed at rollouts 24/39/54/69/84 -- each exactly 15 apart,
# matching the old RECOVERY_PATIENCE=15 precisely; RECOVERY_COOLDOWN doesn't
# extend the period because since_best keeps incrementing during cooldown
# too). A 600k-step run is ~293 rollouts (600_000/ROLLOUT_LEN); to get ~10
# roughly uniform triggers spanning the whole run instead of 5 clustered in
# the first third, period should be ~293/10 ~= 29 rollouts, so
# RECOVERY_PATIENCE=29 with RECOVERY_MAX_TRIGGERS=10. Unverified against a
# completed run as of this comment.
BEST_MIN_DELTA = 0.01        # min collision_rate improvement to count as real progress, not noise
RECOVERY_PATIENCE = 29        # rollouts -- retimed for ~10 uniform triggers over a 600k-step run, see comment above
RECOVERY_COOLDOWN = 10        # rollouts (~20k steps) to hold the boost before re-judging
RECOVERY_MAX_TRIGGERS = 10    # was 5 -- see comment above
ENTROPY_RECOVERY_ENT_COEF = ENT_COEF_START

# Rollout collection is ~2048 sequential steps -- measured on Kaggle: CPU
# sustains ~330 steps/sec, CUDA only ~190 steps/sec (~40% slower) -- the
# network is tiny enough that per-call GPU overhead dominates over any
# compute benefit. Defaults to CPU because of that measurement, not a guess.
# DEVICE env var overrides (e.g. DEVICE=cuda to opt back in, if e.g. the
# network gets meaningfully bigger later) -- steps_per_sec stays logged
# either way so any future default change is also a measured decision.
DEVICE = os.environ.get("DEVICE", "cpu")

# SEED env var enables reproducible, non-colliding parallel runs (e.g. one
# Kaggle session per seed): set SEED=1/2/3 per session and each gets its own
# seeded stream plus suffixed log/checkpoint/model filenames. Leaving SEED
# unset keeps the original unsuffixed single-run behavior and paths.
_seed_env = os.environ.get("SEED")
if _seed_env is not None:
    SEED = int(_seed_env)
    # Prefix with agent count whenever it's non-default, so NUM_AGENTS=2/3
    # curriculum runs are self-documenting in their filenames instead of
    # relying on the user picking a distinguishing SEED by convention.
    RUN_ID = f"n{NUM_AGENTS}_{SEED}" if NUM_AGENTS != 4 else str(SEED)
else:
    SEED = int.from_bytes(os.urandom(4), "little") % (2**31)
    RUN_ID = ""

_stop_requested = False

def _handle_interrupt(signum, frame):
    global _stop_requested
    print("\n[!] Interrupt received — will save checkpoint and exit after current rollout.")
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)

COMPONENT_KEYS = ["track", "spread", "safety", "cohesion", "collision", "velocity", "joint", "contact", "brake", "ground", "altitude"]

def joint(obs_dict):
    return np.concatenate([obs_dict[a] for a in AGENTS]).astype(np.float32)


def main():
    global _stop_requested
    print(f"[run] device={DEVICE} seed={SEED} critic_lr={CRITIC_LR} aux_dir_coef={AUX_DIR_COEF} aux_diversify={AUX_DIVERSIFY} aux_dir_ramp={AUX_DIR_RAMP}" + (f" run_id={RUN_ID}" if RUN_ID else " (SEED env var unset -> outputs unsuffixed)"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    init_logger(RUN_ID)
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)

    # Shared actor: the drones are homogeneous (identical dynamics, same
    # local/neighbor-relative observation shape), so one policy learning
    # from all NUM_AGENTS agents' experience at once replaces NUM_AGENTS
    # separate actors each learning from only 1/NUM_AGENTS of the data.
    # The critic already shares a trunk across a per-agent head, so it's
    # untouched -- this only pools the actor's training.
    actor = Actor(OBS_DIM, ACT_DIM, hidden=ACTOR_HIDDEN).to(DEVICE)
    critic = CentralCritic(OBS_DIM * len(AGENTS), num_agents=len(AGENTS), hidden=CRITIC_HIDDEN).to(DEVICE)
    agent_idx = {a: i for i, a in enumerate(AGENTS)}
    opt_actor = optim.Adam(actor.parameters(), lr=LR)
    opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    total_steps, ep_count = load_checkpoint(actor, critic, opt_actor, opt_critic, RUN_ID, DEVICE)

    buf = RolloutBuffer(AGENTS, OBS_DIM, ACT_DIM, ROLLOUT_LEN, gamma=GAMMA)
    obs, _ = env.reset(seed=SEED)
    ep_rewards = {a: 0.0 for a in AGENTS}
    ep_components = {k: 0.0 for k in COMPONENT_KEYS}
    ep_len = 0

    recent_rewards = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)
    recent_target_lost = deque(maxlen=50)
    recent_ground_strike = deque(maxlen=50)  # Phase 3, 2026-08-20
    recent_min_dist = deque(maxlen=50)
    recent_ep_len = deque(maxlen=50)
    recent_mean_pw = deque(maxlen=50)
    recent_std_pw = deque(maxlen=50)
    recent_diameter = deque(maxlen=50)
    recent_components = {k: deque(maxlen=50) for k in COMPONENT_KEYS}

    last_entropy = 0.0   # carried across rollouts for logging
    best_collision_rate = float("inf")   # drives entropy-recovery patience only
    since_best = 0
    cooldown_remaining = 0
    recovery_trigger_count = 0
    # Drives which checkpoint gets saved as "best" -- separate from the
    # collision-rate-only tracking above. Pure collision_rate can prefer a
    # policy that's safe only by spreading out excessively (confirmed: one
    # run's best-by-collision-rate checkpoint had meaningfully worse
    # tracking_rmse and swarm_diameter than its own final-step model).
    # Rounding collision_rate to 2dp groups meaningfully-equal safety levels
    # together so tracking quality (comp_avgs['track'], less negative is
    # better) breaks ties within that group -- a genuinely safer policy
    # still always wins outright on the primary criterion.
    best_score = (float("inf"), float("inf"), float("inf"), float("inf"))  # Phase 3: +ground_strike_rate

    while total_steps < TOTAL_STEPS and not _stop_requested:
        rollout_start = time.time()
        buf.reset()
        action_abs_sum = 0.0
        action_count = 0
        brake_sum = 0.0
        # Brake instrumentation (2026-08-20, supervisor review) -- see
        # envs/formation_env.py's _apply_brake docstring for what each of
        # these actually measures. brake_passes/max_brake_violation are
        # env-wide (one value per env.step() call, not per agent);
        # brake_solo_sum/brake_multi_sum split brake_sum by whether the
        # agent had 1 vs 2+ simultaneously-active neighbors, to test
        # whether N-aware margin is actually reducing multi-neighbor brake
        # competition specifically, not just inferring it from the
        # aggregate collision rate.
        brake_passes_sum = 0.0
        env_step_count = 0
        max_brake_violation = 0.0
        brake_solo_sum = 0.0
        brake_multi_sum = 0.0
        for t in range(ROLLOUT_LEN):
            with torch.no_grad():
                agents_now = env.agents
                obs_batch = torch.as_tensor(np.stack([obs[a] for a in agents_now])).to(DEVICE)
                act_batch, logp_batch = actor.get_action(obs_batch)
                # Mean |action| this rollout -- how close to tanh's +-1
                # saturation the policy's actual commanded actions sit, as
                # distinct from log_std (exploration noise around the mean).
                # See Actor.get_log_std()'s docstring for why both are logged.
                action_abs_sum += act_batch.abs().sum().item()
                action_count += act_batch.numel()
                act_dict = {a: act_batch[i].cpu().numpy() for i, a in enumerate(agents_now)}
                logp_dict = {a: logp_batch[i].item() for i, a in enumerate(agents_now)}

                jobs = torch.as_tensor(joint(obs)).unsqueeze(0).to(DEVICE)
                values = critic(jobs).squeeze(0)
                value_dict = {a: values[agent_idx[a]].item() for a in AGENTS}

                # Search-direction auxiliary objective (2026-08-26): captured
                # from the state the action above was actually chosen from,
                # before stepping -- steps_since_contact/_last_known_vel are
                # swarm-wide (identical for every agent this step), see
                # AUX_DIR_COEF's comment above.
                in_search_now = float(env._steps_since_contact > 0)
                lkv_norm = float(np.linalg.norm(env._last_known_vel))
                lkv_dir = (env._last_known_vel / lkv_norm).astype(np.float32) if lkv_norm > 1e-6 else np.zeros(3, np.float32)
                if AUX_DIVERSIFY:
                    # Per-agent blend (see AUX_DIVERSIFY's comment above):
                    # unit(last_known_vel + own_search_dir), falling back to
                    # the agent's own search direction alone if the two
                    # nearly cancel (rare, but a real edge case -- a
                    # near-zero blended vector would otherwise normalize to
                    # an unstable/meaningless direction).
                    aux_dir_dict = {}
                    for a in AGENTS:
                        search_dir_a = env._search_dirs.get(a)
                        if search_dir_a is None:
                            aux_dir_dict[a] = np.zeros(3, np.float32)
                            continue
                        blended = lkv_dir + search_dir_a
                        b_norm = float(np.linalg.norm(blended))
                        aux_dir_dict[a] = (blended / b_norm).astype(np.float32) if b_norm > 1e-6 else search_dir_a.astype(np.float32)
                else:
                    aux_dir_dict = {a: lkv_dir for a in AGENTS}
                in_search_dict = {a: in_search_now for a in AGENTS}
                # AUX_DIR_RAMP (2026-08-27): 0 at loss onset, 1 at
                # LOST_TIMEOUT_STEPS -- see AUX_DIR_RAMP's comment above.
                # Swarm-wide like in_search_now/lkv_dir, for the same reason.
                aux_ramp_frac_now = min(1.0, env._steps_since_contact / LOST_TIMEOUT_STEPS)
                aux_ramp_frac_dict = {a: aux_ramp_frac_now for a in AGENTS}

            next_obs, rewards, terms, truncs, infos = env.step(act_dict)
            done = any(terms.values()) or any(truncs.values())
            # any(terms.values()) is true for EITHER a real collision or a
            # target_lost failure (both are real terminal states, see
            # formation_env.py) -- correct for the bootstrap-skip decision
            # below (neither should be bootstrapped, same as before), but
            # NOT the same thing as "collided" for metrics purposes. Read
            # the two apart explicitly via infos so collision_rate doesn't
            # silently start counting target_lost episodes as collisions.
            real_terminal = any(terms.values())
            any_info = next(iter(infos.values()), {})
            collided = bool(any_info.get("collision", False))
            target_lost_now = bool(any_info.get("target_lost", False))
            ground_strike_now = bool(any_info.get("ground_strike", False))  # Phase 3, 2026-08-20

            if done and not real_terminal:
                # Time-limit cutoff, not a true terminal state -- fold the
                # critic's estimate of the actual next observation into this
                # transition's reward so GAE doesn't treat "ran out of steps"
                # the same as "crashed" (matches CleanRL's truncation handling;
                # leaves buffer/GAE code untouched since the bootstrap is
                # already baked into the reward by the time it's stored).
                with torch.no_grad():
                    jobs_next = torch.as_tensor(joint(next_obs)).unsqueeze(0).to(DEVICE)
                    boot_values = critic(jobs_next).squeeze(0)
                for a in AGENTS:
                    rewards[a] = rewards[a] + GAMMA * boot_values[agent_idx[a]].item()

            buf.store(obs, joint(obs), act_dict, logp_dict, rewards, float(done), value_dict,
                      aux_dir_dict, in_search_dict, aux_ramp_frac_dict)
            # Sum across whichever agents were alive this step -- same
            # agent-step denominator as action_count, so mean_brake_reduction
            # (below) is directly comparable to mean_action_abs: how much of
            # the commanded speed the closing-speed brake actually removed,
            # averaged per agent per step. Zero for an entire rollout would
            # mean the brake never engaged -- worth checking before reading
            # anything into whether it changed collision_rate.
            brake_sum += sum(infos.get(a, {}).get("brake_reduction", 0.0) for a in agents_now)
            brake_passes_sum += any_info.get("brake_passes", 0)
            env_step_count += 1
            max_brake_violation = max(max_brake_violation, any_info.get("brake_violation", 0.0))
            for a in agents_now:
                red = infos.get(a, {}).get("brake_reduction", 0.0)
                if infos.get(a, {}).get("k_active", 0) >= 2:
                    brake_multi_sum += red
                else:
                    brake_solo_sum += red
            for a in AGENTS:
                ep_rewards[a] += rewards.get(a, 0.0)
            # Use infos (populated by env.step() before it internally clears
            # self.agents on a terminal step), not env.agents (checked here
            # AFTER step() already returned, so it's empty on exactly the
            # step that just ended). Bug found 2026-08-14: the old `if
            # env.agents:` gate meant every episode's terminal step -- the
            # ONLY step r_collision (-300) is ever nonzero -- never got
            # accumulated, so the logged r_collision column was silently 0.0
            # in every rollout of every run so far, 100% of the time.
            # Training itself was unaffected (buf.store above already has the
            # real reward), and collision_rate is computed independently from
            # terms -- this only blinded that one logged component.
            if infos:
                any_agent = next(iter(infos))
                comps = infos[any_agent].get("reward_components", {})
                for k in COMPONENT_KEYS:
                    ep_components[k] += comps.get(k, 0.0)
            ep_len += 1
            total_steps += 1

            if done:
                ep_count += 1
                min_d = min((infos[a]["min_dist"] for a in infos), default=0.0)
                swarm = env.get_swarm_stats()

                recent_rewards.append(np.mean(list(ep_rewards.values())) / max(ep_len, 1))
                recent_collisions.append(1.0 if collided else 0.0)
                recent_target_lost.append(1.0 if target_lost_now else 0.0)
                recent_ground_strike.append(1.0 if ground_strike_now else 0.0)
                recent_min_dist.append(min_d)
                recent_ep_len.append(ep_len)
                recent_mean_pw.append(swarm["mean_pairwise"])
                recent_std_pw.append(swarm["std_pairwise"])
                recent_diameter.append(swarm["swarm_diameter"])
                for k in COMPONENT_KEYS:
                    recent_components[k].append(ep_components[k] / max(ep_len, 1))

                obs, _ = env.reset()
                ep_rewards = {a: 0.0 for a in AGENTS}
                ep_components = {k: 0.0 for k in COMPONENT_KEYS}
                ep_len = 0
            else:
                obs = next_obs

            if _stop_requested:
                break

        rollout_elapsed = time.time() - rollout_start   # collection phase only
        train_start = time.time()

        with torch.no_grad():
            jobs = torch.as_tensor(joint(obs)).unsqueeze(0).to(DEVICE)
            last_values = critic(jobs).squeeze(0)
            last_value_dict = {a: last_values[agent_idx[a]].item() for a in AGENTS}

        collision_rate = float(np.mean(recent_collisions)) if recent_collisions else 0.0
        target_lost_rate = float(np.mean(recent_target_lost)) if recent_target_lost else 0.0
        ground_strike_rate = float(np.mean(recent_ground_strike)) if recent_ground_strike else 0.0  # Phase 3, 2026-08-20

        # Best-checkpoint tracking and entropy recovery share this signal:
        # only judge once the rolling window has real episodes in it, and
        # only count an improvement if it clears BEST_MIN_DELTA (avoids
        # noise flip-flopping the patience counter).
        entropy_recovery = False
        if recent_collisions:
            if collision_rate < best_collision_rate - BEST_MIN_DELTA:
                best_collision_rate = collision_rate
                since_best = 0
            else:
                since_best += 1

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                entropy_recovery = True
            elif since_best >= RECOVERY_PATIENCE and recovery_trigger_count < RECOVERY_MAX_TRIGGERS:
                entropy_recovery = True
                cooldown_remaining = RECOVERY_COOLDOWN
                recovery_trigger_count += 1
                since_best = 0   # fresh patience window once the boost ends

        last_actor_loss, last_critic_loss = 0.0, 0.0
        last_approx_kl, last_clip_frac = 0.0, 0.0
        last_aux_loss, last_aux_cos, last_aux_frac = 0.0, float("nan"), 0.0
        last_aux_coef_mean = AUX_DIR_COEF
        if buf.ptr == ROLLOUT_LEN:
            # Linear LR anneal to 0 over training -- constant with a
            # collapsing action std means the same gradient step moves the
            # policy further and further in probability space as training
            # progresses, which is what let approx_kl/clip_frac drift upward
            # in the back half of every prior run instead of settling.
            frac = min(1.0, total_steps / TOTAL_STEPS)
            lr_now = LR * (1.0 - frac)
            # Critic-LR recovery experiment (2026-08-25): this anneal used
            # to set BOTH optimizers to the same lr_now (derived only from
            # the actor's LR) every rollout -- silently overwriting
            # CRITIC_LR back to a LR-derived value on the very first
            # rollout, regardless of what it was constructed with. Caught
            # by a local smoke test producing bit-identical critic-health
            # output under CRITIC_LR=3e-4 vs CRITIC_LR=1e-5 before any
            # Kaggle spend. Each optimizer now anneals from its OWN base
            # LR by the same (1-frac) ratio, preserving the original
            # shared-schedule behavior exactly when CRITIC_LR==LR (the
            # unset/default case) while actually respecting an override.
            critic_lr_now = CRITIC_LR * (1.0 - frac)
            for g in opt_actor.param_groups:
                g["lr"] = lr_now
            for g in opt_critic.param_groups:
                g["lr"] = critic_lr_now

            scheduled_ent_coef = ENT_COEF_START + frac * (ENT_COEF_END - ENT_COEF_START)
            ent_coef = ENTROPY_RECOVERY_ENT_COEF if entropy_recovery else scheduled_ent_coef

            # Pool all agents' rollout data into one batch for the shared
            # actor -- GAE is still computed per-agent (each has its own
            # reward stream and critic head), only the actor's training
            # batch is combined.
            o_list, jo_list, act_list, old_logp_list, adv_list, ret_list, head_list = [], [], [], [], [], [], []
            aux_dir_list, in_search_list, aux_ramp_frac_list = [], [], []
            for a in AGENTS:
                o, jo, act, old_logp, adv, ret, aux_dir_a, in_search_a, aux_ramp_frac_a = buf.get_tensors(a, last_value_dict[a])
                o_list.append(o); jo_list.append(jo); act_list.append(act)
                old_logp_list.append(old_logp); adv_list.append(adv); ret_list.append(ret)
                head_list.append(torch.full((o.shape[0],), agent_idx[a], dtype=torch.long))
                aux_dir_list.append(aux_dir_a); in_search_list.append(in_search_a)
                aux_ramp_frac_list.append(aux_ramp_frac_a)

            o = torch.cat(o_list).to(DEVICE)
            jo = torch.cat(jo_list).to(DEVICE)
            act = torch.cat(act_list).to(DEVICE)
            old_logp = torch.cat(old_logp_list).to(DEVICE)
            adv = torch.cat(adv_list).to(DEVICE)
            ret = torch.cat(ret_list).to(DEVICE)
            head_idx = torch.cat(head_list).to(DEVICE)
            aux_dir = torch.cat(aux_dir_list).to(DEVICE)
            in_search = torch.cat(in_search_list).to(DEVICE) > 0.5
            aux_ramp_frac = torch.cat(aux_ramp_frac_list).to(DEVICE)

            n = o.shape[0]
            stop_early = False
            for _ in range(EPOCHS):
                idx = torch.randperm(n)
                epoch_kls = []
                for start in range(0, n, BATCH_SIZE):
                    b = idx[start:start + BATCH_SIZE]
                    logp, entropy, mean_dir = actor.evaluate(o[b], act[b])
                    ratio = torch.exp(logp - old_logp[b])
                    s1 = ratio * adv[b]
                    s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]

                    # Search-direction auxiliary objective (2026-08-26) --
                    # see AUX_DIR_COEF's comment above. mean_dir is
                    # gradient-connected to the actor's CURRENT parameters
                    # (unlike the buffer's stored, already-sampled action),
                    # which is what an objective on "what does the policy
                    # currently intend" must use. Restricted to in-search
                    # rows only (in_search[b]) -- inert (mask never true)
                    # outside active search, by construction.
                    #
                    # AUX_DIR_RAMP (2026-08-27): per-row coefficient instead
                    # of the flat AUX_DIR_COEF -- see AUX_DIR_RAMP's comment
                    # above. AUX_DIR_RAMP=0 makes coef_vals a constant
                    # AUX_DIR_COEF tensor, reproducing the old flat-coefficient
                    # loss exactly (verified: (1+0*x)==1 for all x).
                    search_mask = in_search[b]
                    if search_mask.any():
                        cos_vals = torch.nn.functional.cosine_similarity(
                            mean_dir[search_mask], aux_dir[b][search_mask], dim=-1)
                        ramp_frac_vals = aux_ramp_frac[b][search_mask]
                        coef_vals = AUX_DIR_COEF * (1.0 + AUX_DIR_RAMP * ramp_frac_vals)
                        per_row_loss = 1.0 - cos_vals
                        aux_term = (coef_vals * per_row_loss).mean()
                        last_aux_cos = cos_vals.mean().item()
                        last_aux_loss = per_row_loss.mean().item()
                        last_aux_coef_mean = coef_vals.mean().item()
                    else:
                        aux_term = torch.zeros((), device=o.device)
                        last_aux_cos = float("nan")
                        last_aux_loss = 0.0
                        last_aux_coef_mean = AUX_DIR_COEF
                    last_aux_frac = search_mask.float().mean().item()

                    actor_loss = -torch.min(s1, s2).mean() - ent_coef * entropy.mean() + aux_term

                    value_pred = critic(jo[b])[torch.arange(len(b)), head_idx[b]]
                    critic_loss = ((value_pred - ret[b]) ** 2).mean()

                    opt_actor.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Standard PPO safeguard (SB3 default max_grad_norm=0.5)
                    # -- matters more here than usual given the -300
                    # collision spike sitting next to much smaller ordinary
                    # rewards, which can produce occasional large advantage
                    # excursions and correspondingly large gradients.
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    opt_actor.step()

                    opt_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_critic.step()

                    with torch.no_grad():
                        last_approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                        last_clip_frac = (torch.abs(ratio - 1) > CLIP).float().mean().item()
                    epoch_kls.append(last_approx_kl)
                    last_actor_loss = actor_loss.item()
                    last_critic_loss = critic_loss.item()
                    last_entropy = entropy.mean().item()

                    # Per-minibatch check, not just per-epoch: the previous
                    # version only stopped between epochs, so a single bad
                    # minibatch could still be followed by the rest of that
                    # epoch's updates before anything reacted.
                    if last_approx_kl > TARGET_KL:
                        stop_early = True
                        break

                if stop_early or np.mean(epoch_kls) > TARGET_KL:
                    break

            train_elapsed = time.time() - train_start
            total_elapsed = rollout_elapsed + train_elapsed
            steps_per_sec = ROLLOUT_LEN / total_elapsed if total_elapsed > 0 else 0.0
            collect_sps = ROLLOUT_LEN / rollout_elapsed if rollout_elapsed > 0 else 0.0

            # Critic-health diagnostic (critic-LR recovery experiment,
            # 2026-08-25) -- see CRITIC_LR's comment above for the full
            # investigation this is checking. Computed on a random
            # subsample of this rollout's own already-pooled (jo, ret)
            # batch -- spans this whole rollout's real, mixed episodes
            # (contact + loss at every depth), not a narrow temporal
            # slice, avoiding the sampling artifact the
            # diagnostic work found in an earlier, contiguous-batch
            # version of this same check. Does not affect training in any
            # way -- read-only forward passes on already-computed tensors,
            # logged for analysis only.
            with torch.no_grad():
                diag_idx = torch.randperm(n)[:min(2000, n)]
                v_diag = critic(jo[diag_idx])[torch.arange(len(diag_idx)), head_idx[diag_idx]]
                ret_diag = ret[diag_idx]
                pre_h1 = critic.trunk[0](jo[diag_idx])
                h1 = critic.trunk[1](pre_h1)
                pre_h2 = critic.trunk[2](h1)
                h2 = critic.trunk[3](pre_h2)
            v_diag_np, ret_diag_np = v_diag.numpy(), ret_diag.numpy()
            if v_diag_np.std() > 1e-8:
                critic_pearson = float(np.corrcoef(v_diag_np, ret_diag_np)[0, 1])
                critic_spearman = float(spearmanr(v_diag_np, ret_diag_np).correlation)
            else:
                critic_pearson = critic_spearman = 0.0
            critic_v_mean, critic_v_std = float(v_diag.mean()), float(v_diag.std())
            critic_v_min, critic_v_max = float(v_diag.min()), float(v_diag.max())
            critic_target_mean, critic_target_std = float(ret_diag.mean()), float(ret_diag.std())
            critic_final_sat_frac = float((h2.abs() > 0.999).float().mean())
            critic_final_pre_std = float(pre_h2.std())

            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            avg_min_dist = float(np.mean(recent_min_dist)) if recent_min_dist else 0.0
            avg_ep_len = float(np.mean(recent_ep_len)) if recent_ep_len else 0.0
            mean_pw = float(np.mean(recent_mean_pw)) if recent_mean_pw else 0.0
            std_pw = float(np.mean(recent_std_pw)) if recent_std_pw else 0.0
            diameter = float(np.mean(recent_diameter)) if recent_diameter else 0.0
            comp_avgs = {k: float(np.mean(recent_components[k])) if recent_components[k] else 0.0
                          for k in COMPONENT_KEYS}

            if recent_collisions:
                # Extended (2026-08-14) with target_lost_rate as a second
                # safety-tier criterion, between collision_rate and the
                # track tie-breaker -- a checkpoint that's collision-free but
                # frequently loses the target isn't actually the best one,
                # same lexicographic-safety-first reasoning as collision_rate
                # itself (see the original comment below/DECISIONS.md).
                # Extended again (Phase 3, 2026-08-20) with ground_strike_rate,
                # ranked with collision_rate (both are hardware-destroying
                # failures, unlike target_lost's softer "mission failed but
                # nothing's damaged") rather than after target_lost.
                current_score = (round(collision_rate, 2), round(ground_strike_rate, 2),
                                  round(target_lost_rate, 2), -comp_avgs['track'])
                if current_score < best_score:
                    best_score = current_score
                    save_best_actor(actor, RUN_ID)

            log_std_mean = actor.get_log_std().mean().item()
            mean_action_abs = action_abs_sum / action_count if action_count > 0 else 0.0
            mean_brake_reduction = brake_sum / action_count if action_count > 0 else 0.0
            mean_brake_passes = brake_passes_sum / env_step_count if env_step_count > 0 else 0.0
            mean_brake_solo = brake_solo_sum / action_count if action_count > 0 else 0.0
            mean_brake_multi = brake_multi_sum / action_count if action_count > 0 else 0.0

            print(f"steps={total_steps:>8} ep={ep_count:>5} "
                  f"avg_rew={avg_reward:>7.1f} coll_rate={collision_rate:.2f} best={best_collision_rate:.2f} "
                  f"lost_rate={target_lost_rate:.2f} ground_rate={ground_strike_rate:.2f} "
                  f"min_dist={avg_min_dist:.2f} ep_len={avg_ep_len:.0f} "
                  f"entropy={last_entropy:.2f} log_std={log_std_mean:.3f} act_abs={mean_action_abs:.3f} "
                  f"brake={mean_brake_reduction:.4f} "
                  f"kl={last_approx_kl:.4f} clip_frac={last_clip_frac:.2f} "
                  f"lr={lr_now:.2e} critic_lr={critic_lr_now:.2e} ent_coef={ent_coef:.4f}{' [RECOVERY]' if entropy_recovery else ''} "
                  f"sps={steps_per_sec:.0f} (collect={collect_sps:.0f}) "
                  f"[track={comp_avgs['track']:.1f} spread={comp_avgs['spread']:.1f} "
                  f"safety={comp_avgs['safety']:.1f} cohesion={comp_avgs['cohesion']:.1f} "
                  f"coll_pen={comp_avgs['collision']:.1f} vel={comp_avgs['velocity']:.1f} "
                  f"joint={comp_avgs['joint']:.1f} contact={comp_avgs['contact']:.1f} "
                  f"brake_pen={comp_avgs['brake']:.1f}] "
                  f"critic[v={critic_v_mean:.1f}±{critic_v_std:.2f} tgt={critic_target_mean:.1f}±{critic_target_std:.1f} "
                  f"pearson={critic_pearson:.3f} spearman={critic_spearman:.3f} "
                  f"final_sat={critic_final_sat_frac*100:.0f}% pre_std={critic_final_pre_std:.2f}] "
                  f"aux[coef={AUX_DIR_COEF:.3f} ramp={AUX_DIR_RAMP:.2f} coef_mean={last_aux_coef_mean:.4f} "
                  f"loss={last_aux_loss:.4f} cos={last_aux_cos:.3f} frac_insearch={last_aux_frac:.3f}]")

            log_row(total_steps=total_steps, episode=ep_count, avg_reward=avg_reward,
                     collision_rate=collision_rate, target_lost_rate=target_lost_rate,
                     ground_strike_rate=ground_strike_rate,
                     avg_min_dist=avg_min_dist,
                     avg_ep_len=avg_ep_len, entropy=last_entropy,
                     actor_loss=last_actor_loss, critic_loss=last_critic_loss,
                     approx_kl=last_approx_kl, clip_frac=last_clip_frac, early_stop_kl=int(stop_early),
                     steps_per_sec=steps_per_sec, collect_steps_per_sec=collect_sps,
                     ent_coef=ent_coef, entropy_recovery=int(entropy_recovery),
                     best_collision_rate=best_collision_rate,
                     mean_pairwise=mean_pw, std_pairwise=std_pw, swarm_diameter=diameter,
                     r_track=comp_avgs['track'], r_spread=comp_avgs['spread'],
                     r_safety=comp_avgs['safety'], r_cohesion=comp_avgs['cohesion'],
                     r_collision=comp_avgs['collision'], r_velocity=comp_avgs['velocity'],
                     r_joint=comp_avgs['joint'], r_contact=comp_avgs['contact'],
                     r_brake=comp_avgs['brake'], r_ground=comp_avgs['ground'],
                     r_altitude=comp_avgs['altitude'],
                     log_std_mean=log_std_mean, mean_action_abs=mean_action_abs,
                     mean_brake_reduction=mean_brake_reduction,
                     mean_brake_passes=mean_brake_passes, max_brake_violation=max_brake_violation,
                     mean_brake_solo=mean_brake_solo, mean_brake_multi=mean_brake_multi,
                     critic_lr=CRITIC_LR, critic_v_mean=critic_v_mean, critic_v_std=critic_v_std,
                     critic_v_min=critic_v_min, critic_v_max=critic_v_max,
                     critic_target_mean=critic_target_mean, critic_target_std=critic_target_std,
                     critic_pearson=critic_pearson, critic_spearman=critic_spearman,
                     critic_final_sat_frac=critic_final_sat_frac, critic_final_pre_std=critic_final_pre_std,
                     aux_dir_coef=AUX_DIR_COEF, aux_loss=last_aux_loss, aux_cos=last_aux_cos,
                     aux_frac_insearch=last_aux_frac, aux_coef_mean=last_aux_coef_mean)

        save_checkpoint(actor, critic, opt_actor, opt_critic, total_steps, ep_count, RUN_ID)

    if _stop_requested:
        print(f"Stopped early by interrupt at steps={total_steps}. Checkpoint saved.")
    else:
        os.makedirs("models", exist_ok=True)
        suffix = f"_{RUN_ID}" if RUN_ID else ""
        # One shared actor -> one file (evaluate.py loads it once and reuses
        # the same object for every agent, rather than expecting a
        # per-drone-name file each).
        torch.save(actor.state_dict(), f"models/actor{suffix}.pt")
        print(f"Training done. Final model saved to models/actor{suffix}.pt "
              f"(best model during the run: models/actor_best{suffix}.pt, collision_rate={best_collision_rate:.2f})")


if __name__ == "__main__":
    main()
