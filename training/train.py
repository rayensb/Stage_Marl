import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal
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
TOTAL_STEPS = 600_000
DEVICE = "cpu"

_stop_requested = False

def _handle_interrupt(signum, frame):
    global _stop_requested
    print("\n[!] Interrupt received — will save checkpoint and exit after current rollout.")
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)


def joint(obs_dict):
    return np.concatenate([obs_dict[a] for a in AGENTS]).astype(np.float32)


def main():
    global _stop_requested
    init_logger()
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    actors = {a: Actor(OBS_DIM, ACT_DIM).to(DEVICE) for a in AGENTS}
    critic = CentralCritic(OBS_DIM * len(AGENTS)).to(DEVICE)
    opt_actors = {a: optim.Adam(actors[a].parameters(), lr=LR) for a in AGENTS}
    opt_critic = optim.Adam(critic.parameters(), lr=LR)

    total_steps, ep_count = load_checkpoint(actors, critic, opt_actors, opt_critic)

    buf = RolloutBuffer(AGENTS, OBS_DIM, ACT_DIM, ROLLOUT_LEN)
    obs, _ = env.reset()
    ep_rewards = {a: 0.0 for a in AGENTS}
    ep_len = 0

    recent_rewards = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)
    recent_min_dist = deque(maxlen=50)
    recent_ep_len = deque(maxlen=50)
    recent_mean_pw = deque(maxlen=50)
    recent_std_pw = deque(maxlen=50)
    recent_diameter = deque(maxlen=50)

    while total_steps < TOTAL_STEPS and not _stop_requested:
        buf.reset()
        for t in range(ROLLOUT_LEN):
            act_dict, logp_dict = {}, {}
            with torch.no_grad():
                for a in env.agents:
                    o = torch.as_tensor(obs[a]).unsqueeze(0)
                    act, logp = actors[a].get_action(o)
                    act_dict[a] = act.squeeze(0).numpy()
                    logp_dict[a] = logp.item()
                jobs = torch.as_tensor(joint(obs)).unsqueeze(0)
                value = critic(jobs).item()

            next_obs, rewards, terms, truncs, infos = env.step(act_dict)
            done = any(terms.values()) or any(truncs.values())
            collided = any(terms.values())

            buf.store(obs, joint(obs), act_dict, logp_dict, rewards, float(done), value)
            for a in AGENTS:
                ep_rewards[a] += rewards.get(a, 0.0)
            ep_len += 1
            total_steps += 1

            if done:
                ep_count += 1
                min_d = min((infos[a]["min_dist"] for a in infos), default=0.0)
                swarm = env.get_swarm_stats()

                recent_rewards.append(np.mean(list(ep_rewards.values())))
                recent_collisions.append(1.0 if collided else 0.0)
                recent_min_dist.append(min_d)
                recent_ep_len.append(ep_len)
                recent_mean_pw.append(swarm["mean_pairwise"])
                recent_std_pw.append(swarm["std_pairwise"])
                recent_diameter.append(swarm["swarm_diameter"])

                obs, _ = env.reset()
                ep_rewards = {a: 0.0 for a in AGENTS}
                ep_len = 0
            else:
                obs = next_obs

            if _stop_requested:
                break

        with torch.no_grad():
            jobs = torch.as_tensor(joint(obs)).unsqueeze(0)
            last_value = critic(jobs).item()

        last_actor_loss, last_critic_loss, last_entropy = 0.0, 0.0, 0.0
        if buf.ptr == ROLLOUT_LEN:
            for a in AGENTS:
                o, jo, act, old_logp, adv, ret = buf.get_tensors(a, last_value)
                n = o.shape[0]
                for _ in range(EPOCHS):
                    idx = torch.randperm(n)
                    for start in range(0, n, BATCH_SIZE):
                        b = idx[start:start + BATCH_SIZE]
                        logp, entropy = actors[a].evaluate(o[b], act[b])
                        ratio = torch.exp(logp - old_logp[b])
                        s1 = ratio * adv[b]
                        s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]
                        actor_loss = -torch.min(s1, s2).mean() - 0.01 * entropy.mean()

                        value_pred = critic(jo[b])
                        critic_loss = ((value_pred - ret[b]) ** 2).mean()

                        opt_actors[a].zero_grad()
                        actor_loss.backward(retain_graph=True)
                        opt_actors[a].step()

                        opt_critic.zero_grad()
                        critic_loss.backward()
                        opt_critic.step()

                        last_actor_loss = actor_loss.item()
                        last_critic_loss = critic_loss.item()
                        last_entropy = entropy.mean().item()

            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            collision_rate = float(np.mean(recent_collisions)) if recent_collisions else 0.0
            avg_min_dist = float(np.mean(recent_min_dist)) if recent_min_dist else 0.0
            avg_ep_len = float(np.mean(recent_ep_len)) if recent_ep_len else 0.0
            mean_pw = float(np.mean(recent_mean_pw)) if recent_mean_pw else 0.0
            std_pw = float(np.mean(recent_std_pw)) if recent_std_pw else 0.0
            diameter = float(np.mean(recent_diameter)) if recent_diameter else 0.0

            print(f"steps={total_steps:>8} ep={ep_count:>5} "
                  f"avg_rew={avg_reward:>7.1f} coll_rate={collision_rate:.2f} "
                  f"min_dist={avg_min_dist:.2f} ep_len={avg_ep_len:.0f} "
                  f"mean_pw={mean_pw:.2f} diam={diameter:.2f}")

            log_row(total_steps=total_steps, episode=ep_count, avg_reward=avg_reward,
                     collision_rate=collision_rate, avg_min_dist=avg_min_dist,
                     avg_ep_len=avg_ep_len, entropy=last_entropy,
                     actor_loss=last_actor_loss, critic_loss=last_critic_loss,
                     mean_pairwise=mean_pw, std_pairwise=std_pw, swarm_diameter=diameter)

        save_checkpoint(actors, critic, opt_actors, opt_critic, total_steps, ep_count)

    if _stop_requested:
        print(f"Stopped early by interrupt at steps={total_steps}. Checkpoint saved.")
    else:
        os.makedirs("models", exist_ok=True)
        for a in AGENTS:
            torch.save(actors[a].state_dict(), f"models/actor_{a}.pt")
        print("Training done. Final models saved to models/")


if __name__ == "__main__":
    main()
