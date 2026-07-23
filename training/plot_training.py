import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("logs/training_log.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df.total_steps, df.avg_reward); axes[0,0].set_title("Avg Reward")
axes[0,1].plot(df.total_steps, df.collision_rate, color='red')
axes[0,1].set_title("Collision Rate (should trend -> 0)")
axes[0,1].axhline(0, color='green', ls='--', alpha=0.5)
axes[1,0].plot(df.total_steps, df.avg_min_dist); axes[1,0].set_title("Avg Min Dist")
axes[1,1].plot(df.total_steps, df.critic_loss, color='orange')
axes[1,1].set_title("Critic Loss (should stabilize/decrease)")

plt.tight_layout()
plt.savefig("logs/training_curves.png", dpi=120)
print("Saved logs/training_curves.png")
