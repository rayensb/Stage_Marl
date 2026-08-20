"""Sustained-flight diagnostic + top-down trajectory plot for an already-
trained checkpoint. Added 2026-08-20: training episodes are MAX_STEPS=200
(10 simulated seconds at DT=0.05/20Hz) by design -- that's what the
termination-on-collision/target_lost mechanism needs to stay short and
frequent for PPO to learn from (see DECISIONS.md/EXPERIMENT_LOG.md for why
that's not a bug to remove). This script does NOT change training. It runs
an already-trained, frozen policy for much longer than it ever saw during
training, and does not stop the run on the first collision/target_lost --
both are recorded as events, but the loop keeps going (managed by this
script overriding env.agents after termination, not by changing step()) --
so degradation past the training horizon is actually visible instead of
being cut off at the first failure.

Usage:
    python training/diagnose_horizon.py --model models/actor_best_1_20260820-1255_3f9069c-n4-5m-phase2-seed1.pt --seconds 60
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from envs.formation_env import FormationEnv3D
from training.networks import Actor
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, TARGET_DIST, DT

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to an actor .pt checkpoint")
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated flight duration (training episodes are 10s; default matches the ~1 minute this was built to check)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Output PNG path (default: logs/horizon_<model-basename>.png)")
    args = parser.parse_args()

    steps = int(round(args.seconds / DT))
    actor = Actor(OBS_DIM, ACT_DIM).to(args.device)
    actor.load_state_dict(torch.load(args.model, map_location=args.device))
    actor.eval()
    actors = {a: actor for a in AGENTS}

    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    obs, _ = env.reset(seed=args.seed)

    traj = {a: [] for a in AGENTS}
    target_traj = []
    track_err_by_step = []
    events = []  # (step, "collision"|"target_lost")

    for step in range(steps):
        act_dict = {}
        with torch.no_grad():
            for a in env.agents:
                o = torch.as_tensor(obs[a]).unsqueeze(0).to(args.device)
                act, _ = actors[a].get_action(o, deterministic=True)
                act_dict[a] = act.squeeze(0).cpu().numpy()

        for a in AGENTS:
            traj[a].append(env.pos[a].copy())
        target_traj.append(env.pos_t.copy())
        track_err_by_step.append(float(np.mean([
            abs(float(np.linalg.norm(env.pos[a] - env.pos_t)) - TARGET_DIST) for a in env.agents
        ])) if env.agents else float("nan"))

        obs, rewards, terms, truncs, infos = env.step(act_dict)
        any_info = next(iter(infos.values()), {})
        if any_info.get("collision", False):
            events.append((step, "collision"))
        if any_info.get("target_lost", False):
            events.append((step, "target_lost"))
        # Deliberately override termination here (diagnostic-only, does not
        # touch step()/reset() -- see module docstring) so a single failure
        # doesn't cut the run short; we want to see what happens after.
        if not env.agents:
            env.agents = env.possible_agents[:]

    print(f"[diagnose_horizon] ran {steps} steps ({args.seconds:.0f}s simulated) -- "
          f"{len([e for e in events if e[1]=='collision'])} collision events, "
          f"{len([e for e in events if e[1]=='target_lost'])} target_lost events")
    if events:
        print(f"[diagnose_horizon] first event: step {events[0][0]} "
              f"({events[0][0]*DT:.1f}s) -- {events[0][1]} "
              f"(training horizon ends at step 200 / 10.0s)")

    # Windowed tracking error -- the actual "does it hold up past the
    # training horizon" number, in 10s buckets (matching MAX_STEPS) so it's
    # directly comparable to a training episode's own scale.
    window = int(round(10.0 / DT))
    print("[diagnose_horizon] mean tracking error by 10s window:")
    for w0 in range(0, steps, window):
        chunk = [e for e in track_err_by_step[w0:w0 + window] if not np.isnan(e)]
        if chunk:
            print(f"    {w0*DT:5.0f}s - {min(w0+window, steps)*DT:5.0f}s: {np.mean(chunk):.3f}")

    out = args.out or f"logs/horizon_{os.path.splitext(os.path.basename(args.model))[0]}.png"
    os.makedirs("logs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(AGENTS)))
    t = np.arange(steps) * DT
    for a, color in zip(AGENTS, colors):
        pts = np.array(traj[a])
        segs = np.stack([pts[:-1, :2], pts[1:, :2]], axis=1)
        lc = LineCollection(segs, array=t[:-1], cmap='viridis', linewidth=1.5, alpha=0.85)
        ax.add_collection(lc)
        ax.plot(pts[0, 0], pts[0, 1], 'o', color=color, markersize=9, label=f"{a} start")
        ax.plot(pts[-1, 0], pts[-1, 1], 's', color=color, markersize=9)

    tgt = np.array(target_traj)
    ax.plot(tgt[:, 0], tgt[:, 1], '--', color='black', linewidth=1, alpha=0.6, label="target path")
    ax.plot(tgt[0, 0], tgt[0, 1], '*', color='gold', markersize=16, markeredgecolor='black', label="target start", zorder=5)

    for step_i, kind in events:
        p = target_traj[step_i]
        ax.plot(p[0], p[1], 'x', color='red' if kind == 'collision' else 'orange', markersize=12, markeredgewidth=3, zorder=6)

    ax.axvline(0, color='none')  # keep autoscale sane before adding text
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Top-down trajectory, {args.seconds:.0f}s simulated "
                 f"({os.path.basename(args.model)})\ncircle=start, square=end, "
                 f"line color=time elapsed, x=collision/target_lost event")
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', fontsize=8)
    ax.autoscale()
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, t[-1])), ax=ax, label="time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[diagnose_horizon] saved top-down plot -> {out}")


if __name__ == "__main__":
    main()
