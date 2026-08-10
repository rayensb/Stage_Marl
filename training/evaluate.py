"""Deterministic evaluation of trained actors -- no exploration noise, no
training. Runs a batch of episodes with each actor's mean action (tanh(mu),
not a sampled action), and reports the metrics that actually answer "does
this policy work": success/collision rate, tracking accuracy, formation
spacing, and separation margin. Training-time metrics are confounded by
ongoing exploration noise and a still-changing policy; this is the clean
read that decides whether the current setup is worth building on further.

Usage:
    python training/evaluate.py --episodes 100
    python training/evaluate.py --episodes 100 --run-id 1
    SEED=1 python training/evaluate.py --episodes 100   # picks up run-id from SEED
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import numpy as np
import torch

from envs.formation_env import FormationEnv3D
from training.networks import Actor
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, TARGET_DIST

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]


def load_actors(model_dir, run_id, device):
    """Prefers the final models/actor_*.pt files (only written when training
    completes TOTAL_STEPS naturally), falls back to the training checkpoint
    (written every rollout, including on Ctrl+C) so an interrupted run can
    still be evaluated without waiting for it to finish."""
    suffix = f"_{run_id}" if run_id else ""
    actors = {a: Actor(OBS_DIM, ACT_DIM).to(device) for a in AGENTS}

    missing = [a for a in AGENTS if not os.path.exists(os.path.join(model_dir, f"actor_{a}{suffix}.pt"))]
    if not missing:
        for a in AGENTS:
            path = os.path.join(model_dir, f"actor_{a}{suffix}.pt")
            actors[a].load_state_dict(torch.load(path, map_location=device))
    else:
        ckpt_path = f"checkpoints/latest{suffix}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No final models/actor_*{suffix}.pt and no checkpoint at {ckpt_path} either -- "
                f"check --model-dir/--run-id match how training saved this run."
            )
        print(f"[eval] final models/ files not found (run likely interrupted) -- loading from checkpoint {ckpt_path} instead")
        state = torch.load(ckpt_path, map_location=device)
        for a in AGENTS:
            actors[a].load_state_dict(state["actors"][a])

    for a in AGENTS:
        actors[a].eval()
    return actors


def run_episode(env, actors, device, seed, record=False):
    obs, _ = env.reset(seed=seed)
    step = 0
    collided = False
    track_errors, spacing_stds, diameters, speeds = [], [], [], []
    min_dist_ever = float("inf")
    trajectory = [] if record else None

    while True:
        act_dict = {}
        with torch.no_grad():
            for a in env.agents:
                o = torch.as_tensor(obs[a]).unsqueeze(0).to(device)
                act, _ = actors[a].get_action(o, deterministic=True)
                act_dict[a] = act.squeeze(0).cpu().numpy()

        for a in env.agents:
            track_errors.append(abs(float(np.linalg.norm(env.pos[a] - env.pos_t)) - TARGET_DIST))
            speeds.append(float(np.linalg.norm(env.vel[a])))
        swarm = env.get_swarm_stats()
        spacing_stds.append(swarm["std_pairwise"])
        diameters.append(swarm["swarm_diameter"])
        min_dist_ever = min(min_dist_ever, swarm["min_pairwise"])
        if record:
            trajectory.append({"target": env.pos_t.copy(), **{a: env.pos[a].copy() for a in AGENTS}})

        obs, rewards, terms, truncs, infos = env.step(act_dict)
        step += 1
        collided = any(terms.values())
        if collided or any(truncs.values()):
            break

    metrics = {
        "collided": collided,
        "episode_len": step,
        "min_dist": min_dist_ever,
        "tracking_rmse": float(np.sqrt(np.mean(np.square(track_errors)))) if track_errors else 0.0,
        "avg_spacing_std": float(np.mean(spacing_stds)) if spacing_stds else 0.0,
        "avg_diameter": float(np.mean(diameters)) if diameters else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
    }
    return metrics, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-id", default=os.environ.get("SEED", ""))
    parser.add_argument("--seed", type=int, default=0, help="Eval-scenario base seed, independent of the training seed")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--save-trajectory", default=None, help="Optional path to save episode 0's position history as .npz, for a future 3D replay viewer")
    args = parser.parse_args()

    print(f"[eval] device={args.device} run_id={args.run_id or '(none)'} episodes={args.episodes}")
    actors = load_actors(args.model_dir, args.run_id, args.device)
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)

    rows = []
    for ep in range(args.episodes):
        metrics, traj = run_episode(env, actors, args.device, seed=args.seed + ep,
                                      record=(ep == 0 and args.save_trajectory is not None))
        rows.append(metrics)
        if traj is not None:
            np.savez(args.save_trajectory,
                     **{k: np.array([t[k] for t in traj]) for k in traj[0]})
            print(f"[eval] saved episode 0 trajectory -> {args.save_trajectory}")

    collided = np.array([r["collided"] for r in rows], dtype=float)
    ep_lens = np.array([r["episode_len"] for r in rows], dtype=float)
    min_dists = np.array([r["min_dist"] for r in rows], dtype=float)
    tracking_rmses = np.array([r["tracking_rmse"] for r in rows], dtype=float)
    spacing_stds = np.array([r["avg_spacing_std"] for r in rows], dtype=float)
    diameters = np.array([r["avg_diameter"] for r in rows], dtype=float)
    speeds = np.array([r["avg_speed"] for r in rows], dtype=float)

    summary = {
        "episodes": args.episodes,
        "success_rate": float(1.0 - collided.mean()),
        "collision_rate": float(collided.mean()),
        "avg_episode_len": float(ep_lens.mean()),
        "avg_min_dist": float(min_dists.mean()),
        "worst_min_dist": float(min_dists.min()),
        "avg_tracking_rmse": float(tracking_rmses.mean()),
        "avg_spacing_std": float(spacing_stds.mean()),
        "avg_swarm_diameter": float(diameters.mean()),
        "avg_speed": float(speeds.mean()),
    }

    print("\n=== Evaluation summary (deterministic, no exploration) ===")
    for k, v in summary.items():
        print(f"  {k:>20}: {v:.3f}" if isinstance(v, float) else f"  {k:>20}: {v}")

    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/eval_{args.run_id}.csv" if args.run_id else "logs/eval.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode"] + list(rows[0].keys()))
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"episode": i, **r})
    print(f"\nPer-episode results saved to {out_path}")


if __name__ == "__main__":
    main()
