import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import NUM_AGENTS

# Matches train.py/evaluate.py's run_id scheme exactly: SEED, prefixed with
# agent count when NUM_AGENTS is overridden away from the default. Without
# this it silently looked for the wrong (unprefixed) filename whenever
# NUM_AGENTS was set, e.g. NUM_AGENTS=2 SEED=1 -> training_log_n2_1.csv, but
# this used to look for training_log_1.csv and crash with FileNotFoundError.
if len(sys.argv) > 1:
    run_id = sys.argv[1]
else:
    _seed = os.environ.get("SEED")
    if _seed is None:
        run_id = ""
    else:
        run_id = f"n{NUM_AGENTS}_{_seed}" if NUM_AGENTS != 4 else _seed
LOG_PATH = f"logs/training_log_{run_id}.csv" if run_id else "logs/training_log.csv"
OUT_PATH = f"logs/training_curves_{run_id}.png" if run_id else "logs/training_curves.png"

df = pd.read_csv(LOG_PATH)
fig, axes = plt.subplots(7, 2, figsize=(12, 28))

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
             ("r_cohesion","red"),("r_collision","black"),("r_velocity","gray"),
             ("r_joint","purple")]:
    axes[3,0].plot(df.total_steps, df[k], label=k, color=c)
axes[3,0].set_title("Reward Components"); axes[3,0].legend(fontsize=7)
axes[3,1].plot(df.total_steps, df.critic_loss, color='orange'); axes[3,1].set_title("Critic Loss")
axes[4,0].plot(df.total_steps, df.approx_kl, color='teal'); axes[4,0].set_title("Approx KL (watch for spikes)")
axes[4,1].plot(df.total_steps, df.clip_frac, color='magenta'); axes[4,1].set_title("Clip Fraction")
axes[5,0].plot(df.total_steps, df.steps_per_sec, label="total", color='darkgreen')
axes[5,0].plot(df.total_steps, df.collect_steps_per_sec, label="rollout collection only", color='olive')
axes[5,0].set_title("Throughput (steps/sec)"); axes[5,0].legend(fontsize=7)
axes[5,1].plot(df.total_steps, df.ent_coef, color='indigo')
axes[5,1].set_title("Entropy Coefficient (watch for recovery spikes)")
axes[6,0].plot(df.total_steps, df.entropy_recovery, color='crimson', drawstyle='steps-post')
axes[6,0].set_title("Entropy Recovery Active (1=triggered)")
axes[6,1].axis('off')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Saved {OUT_PATH}")
