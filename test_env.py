from envs.formation_env import FormationEnv3D
from config import NUM_AGENTS, K_NEIGHBORS

env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
obs, infos = env.reset(seed=0)
print("agents:", env.agents)
print("obs_dim:", obs["drone1"].shape)

for step in range(200):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if step % 20 == 0:
        min_d = min(infos[a]["min_dist"] for a in env.agents)
        any_a = next(iter(infos))
        print(f"step={step} min_dist={min_d:.2f} rew_d1={rewards.get('drone1', 0):.2f} "
              f"contact={infos[any_a].get('target_lost', '?')==False} conf={env._track_confidence:.2f}")
    if not env.agents:
        any_a = next(iter(infos))
        print("ended at", step, "terms:", terms, "target_lost:", infos[any_a]["target_lost"])
        break
else:
    print("reached max steps OK")
