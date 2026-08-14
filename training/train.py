import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque

from envs.formation_env import FormationEnv3D
from training.networks import Actor, CentralCritic
from training.buffer import RolloutBuffer
from training.checkpoint import save_checkpoint, load_checkpoint, save_best_actor
from training.logger import init_logger, log_row
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]
ROLLOUT_LEN = 2048
EPOCHS = 10
BATCH_SIZE = 256
LR = 3e-4
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

COMPONENT_KEYS = ["track", "spread", "safety", "cohesion", "collision", "velocity", "joint"]

def joint(obs_dict):
    return np.concatenate([obs_dict[a] for a in AGENTS]).astype(np.float32)


def main():
    global _stop_requested
    print(f"[run] device={DEVICE} seed={SEED}" + (f" run_id={RUN_ID}" if RUN_ID else " (SEED env var unset -> outputs unsuffixed)"))
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
    actor = Actor(OBS_DIM, ACT_DIM).to(DEVICE)
    critic = CentralCritic(OBS_DIM * len(AGENTS), num_agents=len(AGENTS)).to(DEVICE)
    agent_idx = {a: i for i, a in enumerate(AGENTS)}
    opt_actor = optim.Adam(actor.parameters(), lr=LR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR)

    total_steps, ep_count = load_checkpoint(actor, critic, opt_actor, opt_critic, RUN_ID, DEVICE)

    buf = RolloutBuffer(AGENTS, OBS_DIM, ACT_DIM, ROLLOUT_LEN, gamma=GAMMA)
    obs, _ = env.reset(seed=SEED)
    ep_rewards = {a: 0.0 for a in AGENTS}
    ep_components = {k: 0.0 for k in COMPONENT_KEYS}
    ep_len = 0

    recent_rewards = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)
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
    best_score = (float("inf"), float("inf"))

    while total_steps < TOTAL_STEPS and not _stop_requested:
        rollout_start = time.time()
        buf.reset()
        action_abs_sum = 0.0
        action_count = 0
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

            next_obs, rewards, terms, truncs, infos = env.step(act_dict)
            done = any(terms.values()) or any(truncs.values())
            collided = any(terms.values())

            if done and not collided:
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

            buf.store(obs, joint(obs), act_dict, logp_dict, rewards, float(done), value_dict)
            for a in AGENTS:
                ep_rewards[a] += rewards.get(a, 0.0)
            if env.agents:
                any_agent = env.agents[0]
                comps = infos.get(any_agent, {}).get("reward_components", {})
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
        if buf.ptr == ROLLOUT_LEN:
            # Linear LR anneal to 0 over training -- constant with a
            # collapsing action std means the same gradient step moves the
            # policy further and further in probability space as training
            # progresses, which is what let approx_kl/clip_frac drift upward
            # in the back half of every prior run instead of settling.
            frac = min(1.0, total_steps / TOTAL_STEPS)
            lr_now = LR * (1.0 - frac)
            for g in opt_actor.param_groups:
                g["lr"] = lr_now
            for g in opt_critic.param_groups:
                g["lr"] = lr_now

            scheduled_ent_coef = ENT_COEF_START + frac * (ENT_COEF_END - ENT_COEF_START)
            ent_coef = ENTROPY_RECOVERY_ENT_COEF if entropy_recovery else scheduled_ent_coef

            # Pool all agents' rollout data into one batch for the shared
            # actor -- GAE is still computed per-agent (each has its own
            # reward stream and critic head), only the actor's training
            # batch is combined.
            o_list, jo_list, act_list, old_logp_list, adv_list, ret_list, head_list = [], [], [], [], [], [], []
            for a in AGENTS:
                o, jo, act, old_logp, adv, ret = buf.get_tensors(a, last_value_dict[a])
                o_list.append(o); jo_list.append(jo); act_list.append(act)
                old_logp_list.append(old_logp); adv_list.append(adv); ret_list.append(ret)
                head_list.append(torch.full((o.shape[0],), agent_idx[a], dtype=torch.long))

            o = torch.cat(o_list).to(DEVICE)
            jo = torch.cat(jo_list).to(DEVICE)
            act = torch.cat(act_list).to(DEVICE)
            old_logp = torch.cat(old_logp_list).to(DEVICE)
            adv = torch.cat(adv_list).to(DEVICE)
            ret = torch.cat(ret_list).to(DEVICE)
            head_idx = torch.cat(head_list).to(DEVICE)

            n = o.shape[0]
            stop_early = False
            for _ in range(EPOCHS):
                idx = torch.randperm(n)
                epoch_kls = []
                for start in range(0, n, BATCH_SIZE):
                    b = idx[start:start + BATCH_SIZE]
                    logp, entropy = actor.evaluate(o[b], act[b])
                    ratio = torch.exp(logp - old_logp[b])
                    s1 = ratio * adv[b]
                    s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]
                    actor_loss = -torch.min(s1, s2).mean() - ent_coef * entropy.mean()

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

            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            avg_min_dist = float(np.mean(recent_min_dist)) if recent_min_dist else 0.0
            avg_ep_len = float(np.mean(recent_ep_len)) if recent_ep_len else 0.0
            mean_pw = float(np.mean(recent_mean_pw)) if recent_mean_pw else 0.0
            std_pw = float(np.mean(recent_std_pw)) if recent_std_pw else 0.0
            diameter = float(np.mean(recent_diameter)) if recent_diameter else 0.0
            comp_avgs = {k: float(np.mean(recent_components[k])) if recent_components[k] else 0.0
                          for k in COMPONENT_KEYS}

            if recent_collisions:
                current_score = (round(collision_rate, 2), -comp_avgs['track'])
                if current_score < best_score:
                    best_score = current_score
                    save_best_actor(actor, RUN_ID)

            log_std_mean = actor.get_log_std().mean().item()
            mean_action_abs = action_abs_sum / action_count if action_count > 0 else 0.0

            print(f"steps={total_steps:>8} ep={ep_count:>5} "
                  f"avg_rew={avg_reward:>7.1f} coll_rate={collision_rate:.2f} best={best_collision_rate:.2f} "
                  f"min_dist={avg_min_dist:.2f} ep_len={avg_ep_len:.0f} "
                  f"entropy={last_entropy:.2f} log_std={log_std_mean:.3f} act_abs={mean_action_abs:.3f} "
                  f"kl={last_approx_kl:.4f} clip_frac={last_clip_frac:.2f} "
                  f"lr={lr_now:.2e} ent_coef={ent_coef:.4f}{' [RECOVERY]' if entropy_recovery else ''} "
                  f"sps={steps_per_sec:.0f} (collect={collect_sps:.0f}) "
                  f"[track={comp_avgs['track']:.1f} spread={comp_avgs['spread']:.1f} "
                  f"safety={comp_avgs['safety']:.1f} cohesion={comp_avgs['cohesion']:.1f} "
                  f"coll_pen={comp_avgs['collision']:.1f} vel={comp_avgs['velocity']:.1f} "
                  f"joint={comp_avgs['joint']:.1f}]")

            log_row(total_steps=total_steps, episode=ep_count, avg_reward=avg_reward,
                     collision_rate=collision_rate, avg_min_dist=avg_min_dist,
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
                     r_joint=comp_avgs['joint'],
                     log_std_mean=log_std_mean, mean_action_abs=mean_action_abs)

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
