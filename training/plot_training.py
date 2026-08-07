import sys, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Matches logger.py's run_id suffixing: pass a run id as the first arg, or
# rely on the same SEED env var used to launch training (so a plain re-run
# in the same Kaggle session picks up the right file automatically).
run_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SEED", "")
LOG_PATH = f"logs/training_log_{run_id}.csv" if run_id else "logs/training_log.csv"
OUT_PATH = f"logs/training_curves_{run_id}.png" if run_id else "logs/training_curves.png"

df = pd.read_csv(LOG_PATH)
fig, axes = plt.subplots(5, 2, figsize=(12, 20))

axes[0,0].plot(df.total_steps, df.avg_reward); axes[0,0].set_title("Avg Reward")
axes[0,1].plot(df.total_steps, df.collision_rate, color='red')
axes[0,1].set_title("Collision Rate (should trend -> 0)")
axes[0,1].axhline(0, color='green', ls='--', alpha=0.5)
axes[1,0].plot(df.total_steps, df.avg_min_dist); axes[1,0].set_title("Avg Min Dist")
axes[1,1].plot(df.total_steps, df.entropy, color='purple')
axes[1,1].set_title("Entropy (should trend DOWN, not up)")
axes[2,0].plot(df.total_steps, df.mean_pairwise, label="mean")
axes[2,0].plot(df.total_steps, df.std_pairwise, label="std")
axes[2,0].set_title("Pairwise Distance"); axes[2,0].legend()
axes[2,1].plot(df.total_steps, df.swarm_diameter, color='brown')
axes[2,1].set_title("Swarm Diameter")
for k, c in [("r_track","blue"),("r_spread","green"),("r_safety","orange"),
             ("r_cohesion","red"),("r_collision","black"),("r_velocity","gray")]:
    axes[3,0].plot(df.total_steps, df[k], label=k, color=c)
axes[3,0].set_title("Reward Components"); axes[3,0].legend(fontsize=7)
axes[3,1].plot(df.total_steps, df.critic_loss, color='orange'); axes[3,1].set_title("Critic Loss")
axes[4,0].plot(df.total_steps, df.approx_kl, color='teal'); axes[4,0].set_title("Approx KL (watch for spikes)")
axes[4,1].plot(df.total_steps, df.clip_frac, color='magenta'); axes[4,1].set_title("Clip Fraction")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Saved {OUT_PATH}")
