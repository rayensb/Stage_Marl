stage/
├── config.py                    # All shared constants. Most are derived, not hand-picked:
│                                 #   safety-zone distances from a reaction-time/closing-speed
│                                 #   model, TARGET_DIST from tetrahedron packing + margin,
│                                 #   COHESION_LIMIT proportional to the ideal formation size.
│                                 #   See the comments in the file for the reasoning behind each.
│
├── envs/
│   ├── __init__.py               # empty, makes envs a package
│   └── formation_env.py          # FormationEnv3D — 4-agent PettingZoo ParallelEnv, 3D space.
│                                 #   Mutual k-NN neighbor locking + connectivity repair
│                                 #   (decentralized execution, not decentralized topology
│                                 #   maintenance -- see the module docstring), obs/reward/step
│                                 #   logic. Safety reward checks all agents, not just locked
│                                 #   neighbors (collision termination is global). Joint bonus
│                                 #   rewards tracking accuracy AND safety being true at once.
│
├── training/
│   ├── __init__.py
│   ├── networks.py               # Actor (tanh-squashed Gaussian, per-agent) + CentralCritic
│   │                             #   (joint obs in, one value head per agent out)
│   ├── buffer.py                 # RolloutBuffer — stores transitions, computes GAE per agent
│   ├── train.py                  # CTDE-PPO training loop: LR annealing, target-KL early
│   │                             #   stopping, truncation-vs-termination bootstrap distinction,
│   │                             #   checkpointing, interrupt-safe, SEED-based reproducible
│   │                             #   runs (suffixes log/checkpoint/model filenames so multiple
│   │                             #   seeds don't collide), steps/sec throughput logging.
│   │                             #   Saves models/actor_droneN[_SEED].pt on completion.
│   ├── evaluate.py               # Deterministic evaluation (no exploration noise) of trained
│   │                             #   actors -- success/collision rate, tracking RMSE, formation
│   │                             #   spacing, separation margin. Falls back to the training
│   │                             #   checkpoint if a run was interrupted before saving final
│   │                             #   models. This is the real read on policy quality; training
│   │                             #   curves are confounded by ongoing exploration noise.
│   ├── checkpoint.py             # save/load full training state (resumable), SEED-suffixed
│   ├── logger.py                 # per-update CSV metrics logging, SEED-suffixed
│   └── plot_training.py          # generates logs/training_curves[_SEED].png from the csv
│
├── models/ checkpoints/ logs/    # gitignored, runtime outputs only
│
└── test_env.py                   # Quick sanity check script — random actions, no training

Workflow: edit locally -> commit/push to GitHub -> pull inside a Kaggle notebook (CPU is the
measured-faster default; GPU is ~40% slower for this network size) ->
`SEED=<n> python training/train.py` -> `python training/evaluate.py --episodes 100 --run-id <n>`
-> download/inspect logs/training_log_<n>.csv, logs/training_curves_<n>.png, logs/eval_<n>.csv.
