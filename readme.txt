stage/
├── config.py                    # All shared constants (distances, speeds, N agents, K neighbors)
│
├── envs/
│   ├── __init__.py               # empty, makes envs a package
│   └── formation_env.py          # FormationEnv3D — N-agent PettingZoo env, 3D space,
│                                   #   k-NN neighbor locking (RECON/LOCKED), obs/reward/step logic
│
├── training/
│   ├── __init__.py
│   ├── networks.py               # Actor (per-agent policy) + CentralCritic (sees joint obs)
│   ├── buffer.py                 # RolloutBuffer — stores transitions, computes GAE per agent
│   └── train.py                  # Main CTDE-PPO training loop, saves models/actor_droneN.pt
│
├── models/                       # (created by training) saved actor weights, one per drone
│
└── test_env.py                   # Quick sanity check script — random actions, no training
