import os
import csv

LOG_PATH = "logs/training_log.csv"
FIELDS = ["total_steps", "episode", "avg_reward", "collision_rate",
          "avg_min_dist", "avg_ep_len", "entropy", "actor_loss", "critic_loss",
          "mean_pairwise", "std_pairwise", "swarm_diameter",
          "r_track", "r_spread", "r_safety", "r_diverge", "r_collision", "r_velocity"]

def init_logger():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def log_row(**kwargs):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(kwargs)
