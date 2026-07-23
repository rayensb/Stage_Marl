from envs.formation_env import FormationEnv3D

env = FormationEnv3D(num_agents=4, k_neighbors=2, scenario=None)
obs, infos = env.reset(seed=0)
print("agents:", env.agents)
print("obs_dim:", obs["drone1"].shape)

for step in range(200):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if step % 20 == 0:
        min_d = min(infos[a]["min_dist"] for a in env.agents)
        print(f"step={step} min_dist={min_d:.2f} rew_d1={rewards.get('drone1', 0):.2f}")
    if not env.agents:
        print("ended at", step, "terms:", terms)
        break
else:
    print("reached max steps OK")
