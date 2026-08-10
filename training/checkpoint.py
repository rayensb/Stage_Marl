import os
import torch


def _ckpt_path(run_id=""):
    return f"checkpoints/latest_{run_id}.pt" if run_id else "checkpoints/latest.pt"


def save_checkpoint(actor, critic, opt_actor, opt_critic, total_steps, ep_count, run_id=""):
    os.makedirs("checkpoints", exist_ok=True)
    state = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "opt_actor": opt_actor.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "total_steps": total_steps,
        "ep_count": ep_count,
    }
    torch.save(state, _ckpt_path(run_id))

def load_checkpoint(actor, critic, opt_actor, opt_critic, run_id="", device="cpu"):
    path = _ckpt_path(run_id)
    if not os.path.exists(path):
        return 0, 0
    state = torch.load(path, map_location=device)
    actor.load_state_dict(state["actor"])
    opt_actor.load_state_dict(state["opt_actor"])
    critic.load_state_dict(state["critic"])
    opt_critic.load_state_dict(state["opt_critic"])
    print(f"Resumed from checkpoint: steps={state['total_steps']} ep={state['ep_count']}")
    return state["total_steps"], state["ep_count"]
