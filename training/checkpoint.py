import os
import torch


def _ckpt_path(run_id=""):
    return f"checkpoints/latest_{run_id}.pt" if run_id else "checkpoints/latest.pt"


def save_checkpoint(actors, critic, opt_actors, opt_critic, total_steps, ep_count, run_id=""):
    os.makedirs("checkpoints", exist_ok=True)
    state = {
        "actors": {a: actors[a].state_dict() for a in actors},
        "critic": critic.state_dict(),
        "opt_actors": {a: opt_actors[a].state_dict() for a in opt_actors},
        "opt_critic": opt_critic.state_dict(),
        "total_steps": total_steps,
        "ep_count": ep_count,
    }
    torch.save(state, _ckpt_path(run_id))

def load_checkpoint(actors, critic, opt_actors, opt_critic, run_id="", device="cpu"):
    path = _ckpt_path(run_id)
    if not os.path.exists(path):
        return 0, 0
    state = torch.load(path, map_location=device)
    for a in actors:
        actors[a].load_state_dict(state["actors"][a])
        opt_actors[a].load_state_dict(state["opt_actors"][a])
    critic.load_state_dict(state["critic"])
    opt_critic.load_state_dict(state["opt_critic"])
    print(f"Resumed from checkpoint: steps={state['total_steps']} ep={state['ep_count']}")
    return state["total_steps"], state["ep_count"]
