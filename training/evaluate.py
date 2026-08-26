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
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, TARGET_DIST, LOST_TIMEOUT_STEPS, DT, ACTOR_HIDDEN

AGENTS = [f"drone{i+1}" for i in range(NUM_AGENTS)]


def load_actors(model_dir, run_id, device, best=False):
    """One shared actor -> one file. Prefers models/actor[_best]_<run_id>.pt
    (actor_best is only written when a new best collision_rate is reached
    during training; actor is only written when TOTAL_STEPS completes
    naturally), falls back to the training checkpoint (written every
    rollout, including on Ctrl+C) so an interrupted run can still be
    evaluated without waiting for it to finish. Returns the same actor
    object under every agent key, since it's one shared policy."""
    suffix = f"_{run_id}" if run_id else ""
    tag = "_best" if best else ""
    path = os.path.join(model_dir, f"actor{tag}{suffix}.pt")

    actor = Actor(OBS_DIM, ACT_DIM, hidden=ACTOR_HIDDEN).to(device)
    if os.path.exists(path):
        actor.load_state_dict(torch.load(path, map_location=device))
    else:
        ckpt_path = f"checkpoints/latest{suffix}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No {path} and no checkpoint at {ckpt_path} either -- "
                f"check --model-dir/--run-id match how training saved this run."
            )
        print(f"[eval] {path} not found (run likely interrupted, or --best requested before any best was saved) "
              f"-- loading from checkpoint {ckpt_path} instead")
        state = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(state["actor"])

    actor.eval()
    return {a: actor for a in AGENTS}


def run_episode(env, actors, device, seed, record=False):
    obs, _ = env.reset(seed=seed)
    step = 0
    collided = False
    target_lost = False
    ground_struck = False
    # Ground truth (env.pos_t), deliberately NOT the swarm's own tracked
    # estimate -- this is the objective "how well did it actually do"
    # measure, independent of what the swarm could or couldn't perceive at
    # the time (which is what the reward, not the eval metric, is scored
    # against). See envs/formation_env.py's module docstring.
    track_errors, spacing_stds, diameters, speeds, confidences = [], [], [], [], []
    estimation_errors = []
    min_dist_ever = float("inf")
    trajectory = [] if record else None

    # Contact/recovery diagnostics (2026-08-21, added for the no-target_lost-
    # termination training ablation -- see config.py's DISABLE_TARGET_LOST_
    # TERMINATION comment). contact_fraction is the primary metric: what
    # fraction of the episode did the swarm actually have the target, not
    # just "did it eventually fail". Streak/event counts distinguish "loses
    # contact briefly and often" from "loses it once and never gets it back"
    # -- both can produce the same target_lost_rate but mean very different
    # things about what to fix.
    contact_steps = 0
    current_contact_streak = 0
    current_loss_streak = 0
    longest_contact_streak = 0
    longest_loss_streak = 0
    loss_events = 0
    reacquisitions_total = 0
    reacquisitions_within_timeout = 0
    # Per-event steps-to-reacquire (2026-08-22) -- reacquisitions_total/
    # reacquisitions_within_timeout above are counts, not durations. This is
    # the actual "how long did search take" distribution: one entry per
    # successful reacquisition this episode, pooled across all episodes in
    # main(). Deliberately does NOT include events that never reacquired
    # (the episode ended in target_lost/truncation first) -- those are
    # already fully captured by target_lost_rate/reacquire_rate_*, and
    # mixing an undefined "how long would it have taken" duration into this
    # list would bias it in a way that's hard to interpret.
    reacquisition_times = []
    was_in_contact = True  # episodes always start in contact by construction (see reset())

    # Productive-agent diagnostic (2026-08-26, added to evaluate the
    # search-direction auxiliary objective) -- per loss event, an agent
    # counts as "productive" if its own true distance to the target ever
    # dropped below its distance at the moment contact was lost
    # (start_dist - min_dist > 0), matching the definition used throughout
    # the actor search-dynamics investigation (see EXPERIMENT_LOG.md).
    # Ground-truth-based, diagnostic/eval-only, same as tracking_rmse above
    # -- never fed into reward or observation.
    productive_agent_counts = []  # one entry per loss event this episode, 0..NUM_AGENTS
    event_start_dist, event_min_dist = {}, {}

    while True:
        act_dict = {}
        with torch.no_grad():
            for a in env.agents:
                o = torch.as_tensor(obs[a]).unsqueeze(0).to(device)
                act, _ = actors[a].get_action(o, deterministic=True)
                act_dict[a] = act.squeeze(0).cpu().numpy()

        true_dists_this_step = {}
        for a in env.agents:
            d = float(np.linalg.norm(env.pos[a] - env.pos_t))
            true_dists_this_step[a] = d
            track_errors.append(abs(d - TARGET_DIST))
            speeds.append(float(np.linalg.norm(env.vel[a])))
        swarm = env.get_swarm_stats()
        spacing_stds.append(swarm["std_pairwise"])
        diameters.append(swarm["swarm_diameter"])
        min_dist_ever = min(min_dist_ever, swarm["min_pairwise"])
        if record:
            trajectory.append({"target": env.pos_t.copy(), **{a: env.pos[a].copy() for a in AGENTS}})

        obs, rewards, terms, truncs, infos = env.step(act_dict)
        confidences.append(env._track_confidence)
        step += 1
        any_info = next(iter(infos.values()), {})
        collided = bool(any_info.get("collision", False))
        ground_struck = ground_struck or bool(any_info.get("ground_strike", False))
        target_lost = bool(any_info.get("target_lost", False))
        estimation_errors.append(float(any_info.get("target_estimation_error", 0.0)))

        in_contact_now = (env._steps_since_contact == 0)
        if in_contact_now:
            if not was_in_contact:
                # Just reacquired -- current_loss_streak still holds the
                # just-ended loss period's full length (checked before it's
                # reset below), which is exactly what decides whether this
                # recovery would have beaten the real LOST_TIMEOUT_STEPS.
                reacquisitions_total += 1
                reacquisition_times.append(current_loss_streak)
                if current_loss_streak <= LOST_TIMEOUT_STEPS:
                    reacquisitions_within_timeout += 1
                if event_start_dist:
                    productive_agent_counts.append(sum(
                        1 for a in event_start_dist
                        if event_start_dist[a] - event_min_dist.get(a, event_start_dist[a]) > 0))
                event_start_dist, event_min_dist = {}, {}
            current_contact_streak += 1
            current_loss_streak = 0
            contact_steps += 1
        else:
            if was_in_contact:
                loss_events += 1
                event_start_dist = dict(true_dists_this_step)
                event_min_dist = dict(true_dists_this_step)
            else:
                for a, d in true_dists_this_step.items():
                    event_min_dist[a] = min(event_min_dist.get(a, d), d)
            current_loss_streak += 1
            current_contact_streak = 0
        longest_contact_streak = max(longest_contact_streak, current_contact_streak)
        longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        was_in_contact = in_contact_now

        # env.agents is authoritative for "did the episode actually end" --
        # respects DISABLE_TARGET_LOST_TERMINATION automatically, unlike
        # re-deriving the stop condition from collided/target_lost/truncs
        # directly (which would stop this loop even when the env itself
        # was told to keep going).
        if not env.agents:
            # Episode ended mid-loss-event (never reacquired) -- finalize
            # that event's productive-agent count too, same as the normal
            # reacquisition path above, so it isn't silently dropped from
            # the distribution.
            if event_start_dist:
                productive_agent_counts.append(sum(
                    1 for a in event_start_dist
                    if event_start_dist[a] - event_min_dist.get(a, event_start_dist[a]) > 0))
                event_start_dist, event_min_dist = {}, {}
            break

    track_errors_arr = np.array(track_errors, dtype=float)
    metrics = {
        "collided": collided,
        "target_lost": target_lost,
        "ground_strike": ground_struck,
        "episode_len": step,
        "min_dist": min_dist_ever,
        "tracking_rmse": float(np.sqrt(np.mean(np.square(track_errors)))) if track_errors else 0.0,
        # Per-episode true-tracking-error tail stats (2026-08-22) -- RMSE
        # above already exists for continuity with prior runs, but it
        # smooths over exactly the tail behavior (occasional bad excursions)
        # that RMSE alone can mask. Pooled, run-level mean/P95/max (the more
        # statistically meaningful version -- see main()) are computed from
        # the raw per-step samples returned below, not from these
        # per-episode values.
        "track_err_p95": float(np.percentile(track_errors_arr, 95)) if len(track_errors_arr) else 0.0,
        "track_err_max": float(track_errors_arr.max()) if len(track_errors_arr) else 0.0,
        "avg_spacing_std": float(np.mean(spacing_stds)) if spacing_stds else 0.0,
        "avg_diameter": float(np.mean(diameters)) if diameters else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_target_estimation_error": float(np.mean(estimation_errors)) if estimation_errors else 0.0,
        "contact_fraction": float(contact_steps / step) if step else 0.0,
        "longest_contact_streak": longest_contact_streak,
        "longest_loss_streak": longest_loss_streak,
        "loss_events": loss_events,
        "reacquisitions_total": reacquisitions_total,
        "reacquisitions_within_timeout": reacquisitions_within_timeout,
    }
    # Raw per-step/per-event samples, kept separate from the CSV-bound
    # per-episode `metrics` dict above -- main() pools these across every
    # episode in the batch to compute run-level percentiles (see its
    # docstring note on why "average of per-episode P95s" is the wrong
    # statistic).
    raw = {
        "track_errors": track_errors_arr,
        "reacquisition_times": np.array(reacquisition_times, dtype=float),
        "productive_agent_counts": np.array(productive_agent_counts, dtype=float),
    }
    return metrics, trajectory, raw


def _default_run_id():
    """Mirrors train.py's RUN_ID scheme exactly (SEED, prefixed with agent
    count when NUM_AGENTS is overridden away from the default) so a plain
    `python training/evaluate.py` run in the same session finds the right
    files without the user needing to reconstruct the id by hand."""
    seed = os.environ.get("SEED")
    if seed is None:
        return ""
    return f"n{NUM_AGENTS}_{seed}" if NUM_AGENTS != 4 else seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--seed", type=int, default=0, help="Eval-scenario base seed, independent of the training seed")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--save-trajectory", default=None, help="Optional path to save episode 0's position history as .npz, for a future 3D replay viewer")
    parser.add_argument("--best", action="store_true", help="Evaluate the best-checkpointed model (by training-time collision_rate) instead of the final one")
    args = parser.parse_args()

    print(f"[eval] device={args.device} run_id={args.run_id or '(none)'} episodes={args.episodes}" + (" model=best" if args.best else " model=final"))
    actors = load_actors(args.model_dir, args.run_id, args.device, best=args.best)
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)

    rows = []
    all_track_errors = []
    all_reacquisition_times = []
    all_productive_agent_counts = []
    for ep in range(args.episodes):
        metrics, traj, raw = run_episode(env, actors, args.device, seed=args.seed + ep,
                                           record=(ep == 0 and args.save_trajectory is not None))
        rows.append(metrics)
        all_track_errors.append(raw["track_errors"])
        all_reacquisition_times.append(raw["reacquisition_times"])
        all_productive_agent_counts.append(raw["productive_agent_counts"])
        if traj is not None:
            np.savez(args.save_trajectory,
                     **{k: np.array([t[k] for t in traj]) for k in traj[0]})
            print(f"[eval] saved episode 0 trajectory -> {args.save_trajectory}")

    collided = np.array([r["collided"] for r in rows], dtype=float)
    target_lost = np.array([r["target_lost"] for r in rows], dtype=float)
    ground_strikes = np.array([r["ground_strike"] for r in rows], dtype=float)
    ep_lens = np.array([r["episode_len"] for r in rows], dtype=float)
    min_dists = np.array([r["min_dist"] for r in rows], dtype=float)
    tracking_rmses = np.array([r["tracking_rmse"] for r in rows], dtype=float)
    spacing_stds = np.array([r["avg_spacing_std"] for r in rows], dtype=float)
    diameters = np.array([r["avg_diameter"] for r in rows], dtype=float)
    speeds = np.array([r["avg_speed"] for r in rows], dtype=float)
    confidences = np.array([r["avg_confidence"] for r in rows], dtype=float)
    estimation_errors = np.array([r["avg_target_estimation_error"] for r in rows], dtype=float)
    contact_fractions = np.array([r["contact_fraction"] for r in rows], dtype=float)
    longest_contact_streaks = np.array([r["longest_contact_streak"] for r in rows], dtype=float)
    longest_loss_streaks = np.array([r["longest_loss_streak"] for r in rows], dtype=float)

    # Aggregated as counts across all episodes, not an average of per-episode
    # ratios -- a single episode's loss-event count is often 0 or 1, which
    # makes a per-episode fraction noisy/undefined; summing first and
    # dividing once is the statistically sound way to get P(reacquire | lost).
    total_loss_events = sum(r["loss_events"] for r in rows)
    total_reacq = sum(r["reacquisitions_total"] for r in rows)
    total_reacq_in_time = sum(r["reacquisitions_within_timeout"] for r in rows)

    # Pooled raw samples across the whole eval batch (2026-08-22) -- the
    # statistically sound way to get run-level tail statistics, same
    # "pool first, then compute the statistic once" reasoning already used
    # above for total_loss_events/total_reacq (a per-episode P95 of ~18
    # samples, then averaged over 100 episodes, is a different and noisier
    # statistic than the true P95 over every sample actually observed).
    pooled_track_errors = np.concatenate(all_track_errors) if all_track_errors else np.array([])
    pooled_reacq_times = np.concatenate(all_reacquisition_times) if all_reacquisition_times else np.array([])
    pooled_productive_counts = np.concatenate(all_productive_agent_counts) if all_productive_agent_counts else np.array([])

    summary = {
        "episodes": args.episodes,
        "success_rate": float(1.0 - collided.mean() - target_lost.mean()),
        "collision_rate": float(collided.mean()),
        "target_lost_rate": float(target_lost.mean()),
        "ground_strike_rate": float(ground_strikes.mean()),
        "avg_episode_len": float(ep_lens.mean()),
        "avg_min_dist": float(min_dists.mean()),
        "worst_min_dist": float(min_dists.min()),
        "avg_tracking_rmse": float(tracking_rmses.mean()),
        "avg_spacing_std": float(spacing_stds.mean()),
        "avg_swarm_diameter": float(diameters.mean()),
        "avg_speed": float(speeds.mean()),
        "avg_track_confidence": float(confidences.mean()),
        "avg_target_estimation_error": float(estimation_errors.mean()),
        # --- pooled true-tracking-error tail stats (2026-08-22) ---
        # Ground-truth-based (env.pos_t, not the swarm's estimate -- see
        # run_episode's track_errors comment), pooled over every step of
        # every episode -- distinct from avg_tracking_rmse above (an
        # average of 100 per-episode RMSEs, which understates tail risk).
        "mean_true_track_err": float(pooled_track_errors.mean()) if len(pooled_track_errors) else 0.0,
        "p95_true_track_err": float(np.percentile(pooled_track_errors, 95)) if len(pooled_track_errors) else 0.0,
        "max_true_track_err": float(pooled_track_errors.max()) if len(pooled_track_errors) else 0.0,
        # --- contact/recovery diagnostics (2026-08-21) ---
        "contact_fraction": float(contact_fractions.mean()),
        "median_contact_streak": float(np.median(longest_contact_streaks)),
        "p90_contact_streak": float(np.percentile(longest_contact_streaks, 90)),
        "max_contact_streak": float(longest_contact_streaks.max()) if len(longest_contact_streaks) else 0.0,
        "frac_episodes_streak_gt_100": float((longest_contact_streaks > 100).mean()),
        "frac_episodes_streak_gt_200": float((longest_contact_streaks > 200).mean()),
        "median_loss_streak": float(np.median(longest_loss_streaks)),
        "max_loss_streak": float(longest_loss_streaks.max()) if len(longest_loss_streaks) else 0.0,
        "total_loss_events": total_loss_events,
        "avg_loss_events_per_episode": float(total_loss_events / args.episodes) if args.episodes else 0.0,
        "reacquire_rate_eventual": float(total_reacq / total_loss_events) if total_loss_events else 0.0,
        "reacquire_rate_within_timeout": float(total_reacq_in_time / total_loss_events) if total_loss_events else 0.0,
        # --- reacquisition-time distribution (2026-08-22) ---
        # Steps/seconds actually taken to regain contact, over successful
        # reacquisitions only (see reacquisition_times' comment in
        # run_episode for why failed/never-reacquired events are excluded
        # rather than biasing this with an undefined duration).
        # n_reacquisitions_observed makes the sample size explicit --
        # important since a "good" seed can paradoxically have very few
        # loss events to compute this distribution from.
        "n_reacquisitions_observed": int(len(pooled_reacq_times)),
        "mean_reacquisition_steps": float(pooled_reacq_times.mean()) if len(pooled_reacq_times) else 0.0,
        "median_reacquisition_steps": float(np.median(pooled_reacq_times)) if len(pooled_reacq_times) else 0.0,
        "p95_reacquisition_steps": float(np.percentile(pooled_reacq_times, 95)) if len(pooled_reacq_times) else 0.0,
        "max_reacquisition_steps": float(pooled_reacq_times.max()) if len(pooled_reacq_times) else 0.0,
        "mean_reacquisition_sec": float(pooled_reacq_times.mean() * DT) if len(pooled_reacq_times) else 0.0,
        "p95_reacquisition_sec": float(np.percentile(pooled_reacq_times, 95) * DT) if len(pooled_reacq_times) else 0.0,
        # --- productive-agent diagnostic (2026-08-26) ---
        # One sample per loss event (pooled across the whole batch, same
        # "pool first" reasoning as reacquisition times above): how many of
        # NUM_AGENTS ever closed on the true target during that event.
        # productive_agent_fraction is mean_productive_agents / NUM_AGENTS,
        # a 0-1 scale matching the other rate metrics in this summary.
        "mean_productive_agents": float(pooled_productive_counts.mean()) if len(pooled_productive_counts) else 0.0,
        "productive_agent_fraction": float(pooled_productive_counts.mean() / NUM_AGENTS) if len(pooled_productive_counts) else 0.0,
        "frac_events_zero_productive": float((pooled_productive_counts == 0).mean()) if len(pooled_productive_counts) else 0.0,
    }

    print("\n=== Evaluation summary (deterministic, no exploration) ===")
    for k, v in summary.items():
        print(f"  {k:>20}: {v:.3f}" if isinstance(v, float) else f"  {k:>20}: {v}")

    os.makedirs("logs", exist_ok=True)
    name_parts = [p for p in (["best"] if args.best else []) + ([args.run_id] if args.run_id else []) if p]
    out_path = f"logs/eval_{'_'.join(name_parts)}.csv" if name_parts else "logs/eval.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode"] + list(rows[0].keys()))
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"episode": i, **r})
    print(f"\nPer-episode results saved to {out_path}")


if __name__ == "__main__":
    main()
