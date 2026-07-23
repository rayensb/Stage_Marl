"""Sanity check — run random actions, confirm no crashes, print shapes."""
import numpy as np
from envs.formation_env import FormationEnv

env = FormationEnv(scenario="danger")
obs, infos = env.reset(seed=0)
print("agents:", env.agents)
print("obs['drone1'] shape:", obs["drone1"].shape, "sample:", obs["drone1"])

for step in range(50):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if step % 10 == 0:
        print(f"step={step} inter_dist={infos.get('drone1',{}).get('inter_dist'):.2f} "
              f"rew_d1={rewards.get('drone1', 0):.2f}")
    if not env.agents:
        print("Episode ended at step", step)
        break

print("Test finished OK.")
