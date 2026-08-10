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
from config import NUM_AGENTS, K_NEIGHBORS

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]
OBS_DIM = 10 + 7 * K_NEIGHBORS
ACT_DIM = 3
ROLLOUT_LEN = 2048
EPOCHS = 10
BATCH_SIZE = 256
LR = 3e-4
CLIP = 0.2
GAMMA = 0.99
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001
TARGET_KL = 0.02   # standard PPO trust-region safety net (SB3/CleanRL default
                     # range ~0.01-0.03) -- stop an agent's epoch loop early if
                     # an update has already moved the policy this far, instead
                     # of blindly running all EPOCHS regardless of update size.
TOTAL_STEPS = 600_000

# Rollout collection is ~2048 sequential batch-1 forward passes per step
# (4 actors + 1 critic, not vectorized) -- likely host/device-transfer-bound
# on GPU rather than compute-bound, since the network itself is tiny. Only
# the EPOCHS/BATCH_SIZE=256 minibatch training phase is a real batched
# workload that GPU can actually accelerate. Falls back to CPU automatically
# if no CUDA device is visible. DEVICE env var forces a choice regardless
# (e.g. DEVICE=cpu to skip CUDA even if a GPU is attached, without touching
# Kaggle's accelerator setting) -- steps_per_sec is now logged so this is a
# measured decision instead of a guess.
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# SEED env var enables reproducible, non-colliding parallel runs (e.g. one
# Kaggle session per seed): set SEED=1/2/3 per session and each gets its own
# seeded stream plus suffixed log/checkpoint/model filenames. Leaving SEED
# unset keeps the original unsuffixed single-run behavior and paths.
_seed_env = os.environ.get("SEED")
if _seed_env is not None:
    SEED = int(_seed_env)
    RUN_ID = str(SEED)
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

COMPONENT_KEYS = ["track", "spread", "safety", "cohesion", "collision", "velocity"]

def joint(obs_dict):
    return np.concatenate([obs_dict[a] for a in AGENTS]).astype(np.float32)


def main():
    global _stop_requested
    print(f"[run] device={DEVICE} seed={SEED}" + (f" run_id={RUN_ID}" if RUN_ID else " (SEED env var unset -> outputs unsuffixed)"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    init_logger(RUN_ID)
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    actors = {a: Actor(OBS_DIM, ACT_DIM).to(DEVICE) for a in AGENTS}
    critic = CentralCritic(OBS_DIM * len(AGENTS), num_agents=len(AGENTS)).to(DEVICE)
    agent_idx = {a: i for i, a in enumerate(AGENTS)}
    opt_actors = {a: optim.Adam(actors[a].parameters(), lr=LR) for a in AGENTS}
    opt_critic = optim.Adam(critic.parameters(), lr=LR)

    total_steps, ep_count = load_checkpoint(actors, critic, opt_actors, opt_critic, RUN_ID, DEVICE)

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

    while total_steps < TOTAL_STEPS and not _stop_requested:
        rollout_start = time.time()
        buf.reset()
        for t in range(ROLLOUT_LEN):
            act_dict, logp_dict = {}, {}
            with torch.no_grad():
                for a in env.agents:
                    o = torch.as_tensor(obs[a]).unsqueeze(0).to(DEVICE)
                    act, logp = actors[a].get_action(o)
                    act_dict[a] = act.squeeze(0).cpu().numpy()
                    logp_dict[a] = logp.item()
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

        last_actor_loss, last_critic_loss, last_entropy = 0.0, 0.0, 0.0
        last_approx_kl, last_clip_frac = 0.0, 0.0
        if buf.ptr == ROLLOUT_LEN:
            # Linear LR anneal to 0 over training -- constant with a
            # collapsing action std means the same gradient step moves the
            # policy further and further in probability space as training
            # progresses, which is what let approx_kl/clip_frac drift upward
            # in the back half of every prior run instead of settling.
            frac = min(1.0, total_steps / TOTAL_STEPS)
            lr_now = LR * (1.0 - frac)
            for a in AGENTS:
                for g in opt_actors[a].param_groups:
                    g["lr"] = lr_now
            for g in opt_critic.param_groups:
                g["lr"] = lr_now
            ent_coef = ENT_COEF_START + frac * (ENT_COEF_END - ENT_COEF_START)

            for a in AGENTS:
                o, jo, act, old_logp, adv, ret = buf.get_tensors(a, last_value_dict[a])
                o, jo, act, old_logp, adv, ret = (o.to(DEVICE), jo.to(DEVICE), act.to(DEVICE),
                                                     old_logp.to(DEVICE), adv.to(DEVICE), ret.to(DEVICE))
                n = o.shape[0]
                for _ in range(EPOCHS):
                    idx = torch.randperm(n)
                    epoch_kls = []
                    for start in range(0, n, BATCH_SIZE):
                        b = idx[start:start + BATCH_SIZE]
                        logp, entropy = actors[a].evaluate(o[b], act[b])
                        ratio = torch.exp(logp - old_logp[b])
                        s1 = ratio * adv[b]
                        s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]
                        actor_loss = -torch.min(s1, s2).mean() - ent_coef * entropy.mean()

                        value_pred = critic(jo[b])[:, agent_idx[a]]
                        critic_loss = ((value_pred - ret[b]) ** 2).mean()

                        opt_actors[a].zero_grad()
                        actor_loss.backward(retain_graph=True)
                        opt_actors[a].step()

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
            collision_rate = float(np.mean(recent_collisions)) if recent_collisions else 0.0
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
                  f"entropy={last_entropy:.2f} kl={last_approx_kl:.4f} clip_frac={last_clip_frac:.2f} lr={lr_now:.2e} "
                  f"sps={steps_per_sec:.0f} (collect={collect_sps:.0f}) "
                  f"[track={comp_avgs['track']:.1f} spread={comp_avgs['spread']:.1f} "
                  f"safety={comp_avgs['safety']:.1f} cohesion={comp_avgs['cohesion']:.1f} "
                  f"coll_pen={comp_avgs['collision']:.1f} vel={comp_avgs['velocity']:.1f}]")

            log_row(total_steps=total_steps, episode=ep_count, avg_reward=avg_reward,
                     collision_rate=collision_rate, avg_min_dist=avg_min_dist,
                     avg_ep_len=avg_ep_len, entropy=last_entropy,
                     actor_loss=last_actor_loss, critic_loss=last_critic_loss,
                     approx_kl=last_approx_kl, clip_frac=last_clip_frac,
                     steps_per_sec=steps_per_sec, collect_steps_per_sec=collect_sps,
                     mean_pairwise=mean_pw, std_pairwise=std_pw, swarm_diameter=diameter,
                     r_track=comp_avgs['track'], r_spread=comp_avgs['spread'],
                     r_safety=comp_avgs['safety'], r_cohesion=comp_avgs['cohesion'],
                     r_collision=comp_avgs['collision'], r_velocity=comp_avgs['velocity'])

        save_checkpoint(actors, critic, opt_actors, opt_critic, total_steps, ep_count, RUN_ID)

    if _stop_requested:
        print(f"Stopped early by interrupt at steps={total_steps}. Checkpoint saved.")
    else:
        os.makedirs("models", exist_ok=True)
        for a in AGENTS:
            suffix = f"_{RUN_ID}" if RUN_ID else ""
            torch.save(actors[a].state_dict(), f"models/actor_{a}{suffix}.pt")
        print("Training done. Final models saved to models/")


if __name__ == "__main__":
    main()
