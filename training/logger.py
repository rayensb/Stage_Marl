import os
import csv

LOG_PATH = "logs/training_log.csv"
FIELDS = ["total_steps", "episode", "avg_reward", "collision_rate",
          "avg_min_dist", "avg_ep_len", "entropy", "actor_loss", "critic_loss",
          "approx_kl", "clip_frac", "steps_per_sec", "collect_steps_per_sec",
          "mean_pairwise", "std_pairwise", "swarm_diameter",
          "r_track", "r_spread", "r_safety", "r_cohesion", "r_collision", "r_velocity", "r_joint"]

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
