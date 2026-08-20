#!/usr/bin/env python3
"""Runs one deterministic episode of a trained policy against the real
FormationEnv3D -- no PX4, no scripted phantoms, every agent is actually
reactive, exactly the environment the policy trained in. Prints each step's
commanded actions live and saves the full trajectory to JSON.

This answers a different, more direct question than deployment/inference_node.py
does: that script tests whether the PX4/ROS2/Gazebo *pipeline* works for one
real vehicle against a scripted stand-in scene; this one tests whether the
*policy itself* tracks a moving target well, in the actual multi-agent
environment it was measured against. No robotics middleware, no coordinate
frames, no arming/failsafe state -- just the policy and the simulation it
was trained in.

Run:
    python3 deployment/sim_demo.py --model models/actor_best_n3_3_20260817-1255_b59c139-3m-seed3.pt --seed 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

import numpy as np
import torch

from envs.formation_env import FormationEnv3D
from training.networks import Actor
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, TARGET_DIST

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="deployment/last_trajectory.json")
    args = parser.parse_args()

    actor = Actor(OBS_DIM, ACT_DIM)
    actor.load_state_dict(torch.load(args.model, map_location="cpu"))
    actor.eval()
    print(f"Loaded {args.model} (NUM_AGENTS={NUM_AGENTS}, OBS_DIM={OBS_DIM}, TARGET_DIST={TARGET_DIST:.2f})")

    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    obs, _ = env.reset(seed=args.seed)
    print(f"Episode seed={args.seed}, agents={env.agents}")

    frames = []
    step = 0
    collided, target_lost = False, False
    while True:
        frame_positions = {a: env.pos[a].tolist() for a in env.agents}
        frame_target = env.pos_t.tolist()

        act_dict = {}
        with torch.no_grad():
            for a in env.agents:
                o = torch.as_tensor(obs[a]).unsqueeze(0)
                act, _ = actor.get_action(o, deterministic=True)
                act_dict[a] = act.squeeze(0).numpy()

        dists = {a: float(np.linalg.norm(env.pos[a] - env.pos_t)) for a in env.agents}
        min_pair = min(
            (float(np.linalg.norm(env.pos[a] - env.pos[b]))
             for i, a in enumerate(env.agents) for b in env.agents[i + 1:]),
            default=float("nan"),
        )
        print(f"step={step:>3} " + " ".join(
            f"{a}:act=[{act_dict[a][0]:+.2f},{act_dict[a][1]:+.2f},{act_dict[a][2]:+.2f}]"
            f" dist_to_target={dists[a]:.2f}" for a in env.agents
        ) + f" min_pairwise={min_pair:.2f}")

        frames.append({
            "step": step, "target": frame_target, "positions": frame_positions,
            "actions": {a: act_dict[a].tolist() for a in act_dict},
        })

        obs, rewards, terms, truncs, infos = env.step(act_dict)
        step += 1
        any_info = next(iter(infos.values()), {})
        collided = bool(any_info.get("collision", False))
        target_lost = bool(any_info.get("target_lost", False))
        if collided or target_lost or any(truncs.values()):
            frames.append({
                "step": step, "target": env.pos_t.tolist(),
                "positions": {a: env.pos[a].tolist() for a in AGENTS}, "actions": {},
            })
            reason = "collision" if collided else ("target_lost" if target_lost else "time limit")
            print(f"Episode ended at step {step}: {reason}")
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "agents": AGENTS, "num_agents": NUM_AGENTS, "target_dist": TARGET_DIST,
            "frames": frames,
            "outcome": {"collided": collided, "target_lost": target_lost, "steps": step},
        }, f)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
