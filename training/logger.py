import os
import csv

LOG_PATH = "logs/training_log.csv"
FIELDS = ["total_steps", "episode", "avg_reward", "collision_rate", "target_lost_rate",
          "avg_min_dist", "avg_ep_len", "entropy", "actor_loss", "critic_loss",
          "approx_kl", "clip_frac", "early_stop_kl", "steps_per_sec", "collect_steps_per_sec",
          "ent_coef", "entropy_recovery", "best_collision_rate",
          "mean_pairwise", "std_pairwise", "swarm_diameter",
          "r_track", "r_spread", "r_safety", "r_cohesion", "r_collision", "r_velocity", "r_joint", "r_contact",
          "r_brake",
          "log_std_mean", "mean_action_abs", "mean_brake_reduction",
          "mean_brake_passes", "max_brake_violation", "mean_brake_solo", "mean_brake_multi"]

def init_logger(run_id=""):
    """run_id suffixes the log filename so parallel runs (e.g. different
    seeds on separate Kaggle sessions) don't overwrite each other's logs if
    later downloaded into the same folder. Empty run_id keeps the original
    unsuffixed path."""
    global LOG_PATH
    LOG_PATH = f"logs/training_log_{run_id}.csv" if run_id else "logs/training_log.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def log_row(**kwargs):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(kwargs)
