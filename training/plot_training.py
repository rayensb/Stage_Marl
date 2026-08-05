import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("logs/training_log.csv")
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

axes[0,0].plot(df.total_steps, df.avg_reward); axes[0,0].set_title("Avg Reward")
axes[0,1].plot(df.total_steps, df.collision_rate, color='red')
axes[0,1].set_title("Collision Rate (should trend -> 0)")
axes[0,1].axhline(0, color='green', ls='--', alpha=0.5)
axes[1,0].plot(df.total_steps, df.avg_min_dist); axes[1,0].set_title("Avg Min Dist")
axes[1,1].plot(df.total_steps, df.critic_loss, color='orange')
axes[1,1].set_title("Critic Loss")
axes[2,0].plot(df.total_steps, df.mean_pairwise, label="mean")
axes[2,0].plot(df.total_steps, df.std_pairwise, label="std")
axes[2,0].set_title("Pairwise Distance (mean/std)"); axes[2,0].legend()
axes[2,1].plot(df.total_steps, df.swarm_diameter, color='purple')
axes[2,1].set_title("Swarm Diameter")

plt.tight_layout()
plt.savefig("logs/training_curves.png", dpi=120)
print("Saved logs/training_curves.png")
