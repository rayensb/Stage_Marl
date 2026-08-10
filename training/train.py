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
from training.checkpoint import save_checkpoint, load_checkpoint
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
TOTAL_STEPS = 600_000   # back to 600k -- 3M was only ever a test of the
                          # entropy floor above, not a new baseline.

# Performance-gated entropy recovery: instead of only a fixed calendar-based
# anneal, if the policy has become confident (entropy < 0) but is still
# performing badly (collision_rate above this threshold), boost exploration
# back toward ENT_COEF_START rather than letting the schedule keep
# suppressing it. The 3M-step run showed unconditional annealing keeps
# shrinking entropy even while collision_rate is stuck high, with no
# mechanism to back off and try something else -- this gives it one.
ENTROPY_RECOVERY_COLLISION_THRESHOLD = 0.10
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

    last_entropy = 0.0   # carried across rollouts for the entropy-recovery check

    while total_steps < TOTAL_STEPS and not _stop_requested:
        rollout_start = time.time()
        buf.reset()
        for t in range(ROLLOUT_LEN):
            with torch.no_grad():
                agents_now = env.agents
                obs_batch = torch.as_tensor(np.stack([obs[a] for a in agents_now])).to(DEVICE)
                act_batch, logp_batch = actor.get_action(obs_batch)
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

        last_actor_loss, last_critic_loss = 0.0, 0.0
        last_approx_kl, last_clip_frac = 0.0, 0.0
        entropy_recovery = False
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
            if last_entropy < 0.0 and collision_rate > ENTROPY_RECOVERY_COLLISION_THRESHOLD:
                ent_coef = ENTROPY_RECOVERY_ENT_COEF
                entropy_recovery = True
            else:
                ent_coef = scheduled_ent_coef

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
                    opt_actor.step()

                    opt_critic.zero_grad()
                    critic_loss.backward()
                    opt_critic.step()

                    with torch.no_grad():
                        last_approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                        last_clip_frac = (torch.abs(ratio - 1) > CLIP).float().mean().item()
                    epoch_kls.append(last_approx_kl)
                    last_actor_loss = actor_loss.item()
                    last_critic_loss = critic_loss.item()
                    last_entropy = entropy.mean().item()

                if np.mean(epoch_kls) > TARGET_KL:
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

            print(f"steps={total_steps:>8} ep={ep_count:>5} "
                  f"avg_rew={avg_reward:>7.1f} coll_rate={collision_rate:.2f} "
                  f"min_dist={avg_min_dist:.2f} ep_len={avg_ep_len:.0f} "
                  f"entropy={last_entropy:.2f} kl={last_approx_kl:.4f} clip_frac={last_clip_frac:.2f} "
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
                     approx_kl=last_approx_kl, clip_frac=last_clip_frac,
                     steps_per_sec=steps_per_sec, collect_steps_per_sec=collect_sps,
                     ent_coef=ent_coef, entropy_recovery=int(entropy_recovery),
                     mean_pairwise=mean_pw, std_pairwise=std_pw, swarm_diameter=diameter,
                     r_track=comp_avgs['track'], r_spread=comp_avgs['spread'],
                     r_safety=comp_avgs['safety'], r_cohesion=comp_avgs['cohesion'],
                     r_collision=comp_avgs['collision'], r_velocity=comp_avgs['velocity'],
                     r_joint=comp_avgs['joint'])

        save_checkpoint(actor, critic, opt_actor, opt_critic, total_steps, ep_count, RUN_ID)

    if _stop_requested:
        print(f"Stopped early by interrupt at steps={total_steps}. Checkpoint saved.")
    else:
        os.makedirs("models", exist_ok=True)
        suffix = f"_{RUN_ID}" if RUN_ID else ""
        # Same shared weights saved once per agent-name file, so evaluate.py
        # (and anything else expecting one file per drone) doesn't need to
        # change at all -- it just happens to load identical weights each time.
        for a in AGENTS:
            torch.save(actor.state_dict(), f"models/actor_{a}{suffix}.pt")
        print("Training done. Final models saved to models/")


if __name__ == "__main__":
    main()
