#!/usr/bin/env python3
"""Plot a deployment/run_live_test.py run: reads every droneN.csv in the
given run directory and produces distance-to-target, min inter-agent
distance, and top-down trajectory plots, saved as summary.png alongside the
CSVs. Also prints a compact per-agent numeric summary.

Run: python3 deployment/plot_run.py deployment/runs/run_YYYYMMDD_HHMMSS/
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def load_run(run_dir):
    data = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "drone*.csv"))):
        agent = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            data[agent] = rows
    return data


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir>")
        return 1
    run_dir = sys.argv[1]

    data = load_run(run_dir)
    if not data:
        print(f"No per-agent CSVs with data found in {run_dir}")
        return 1

    config_path = os.path.join(run_dir, "config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    summary_path = os.path.join(run_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    agents = sorted(data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for a in agents:
        rows = data[a]
        t0 = float(rows[0]["wall_time"])
        t = [float(r["wall_time"]) - t0 for r in rows]
        dist = [float(r["dist_to_target"]) for r in rows]
        ax.plot(t, dist, label=a, linewidth=1)
    ax.set_xlabel("time (s, since first policy cycle)")
    ax.set_ylabel("distance to target (m)")
    ax.set_title("Distance to target per agent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for a in agents:
        rows = data[a]
        t0 = float(rows[0]["wall_time"])
        t = [float(r["wall_time"]) - t0 for r in rows]
        min_d = [float(r["min_dist_to_neighbor"]) for r in rows]
        ax.plot(t, min_d, label=a, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("min distance to a neighbor (m)")
    ax.set_title("Min inter-agent distance (per agent's own view)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(agents), 2)))
    for a, c in zip(agents, colors):
        rows = data[a]
        xs = [float(r["pos_x"]) for r in rows]
        ys = [float(r["pos_y"]) for r in rows]
        ax.plot(xs, ys, color=c, alpha=0.7, linewidth=1, label=a)
        ax.scatter([xs[0]], [ys[0]], color=c, marker="o", s=40)
        ax.scatter([xs[-1]], [ys[-1]], color=c, marker="s", s=40)
    first = data[agents[0]]
    tx = [float(r["target_x"]) for r in first]
    ty = [float(r["target_y"]) for r in first]
    ax.plot(tx, ty, color="black", linestyle=":", linewidth=1.5, label=f"target ({agents[0]}'s belief)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down trajectory (o=start, sq=end)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    title = os.path.basename(os.path.normpath(run_dir))
    if config:
        title += f" -- N={config.get('num_agents')} model={os.path.basename(config.get('model', ''))}"
    if summary.get("outcome"):
        title += f"\n{summary['outcome']}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "summary.png")
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")

    print("\nPer-agent summary:")
    for a in agents:
        rows = data[a]
        dist = [float(r["dist_to_target"]) for r in rows]
        min_d = [float(r["min_dist_to_neighbor"]) for r in rows]
        alt = [float(r["pos_z"]) for r in rows]
        print(f"  {a}: n_cycles={len(rows)} dist_to_target final={dist[-1]:.2f} range=[{min(dist):.2f},{max(dist):.2f}] "
              f"altitude range=[{min(alt):.2f},{max(alt):.2f}] min_dist_to_neighbor min={min(min_d):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
