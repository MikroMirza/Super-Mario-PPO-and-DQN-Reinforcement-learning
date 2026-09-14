import argparse

import torch

import params
from wrappers import make_env
from Agents.PPO import PPOAgent


def transfer(source_path, target_scenario, out_path):
    cfg, hp = params.SCENARIOS[target_scenario]
    env = make_env(
        env_id=cfg["env_id"], skip=cfg["frame_skip"], shape=cfg["frame_size"],
        stack=cfg["frame_stack"], clip_rewards=cfg["clip_rewards"],
        max_episode_steps=cfg["max_ep_steps"], window_visible=cfg["window_visible"],
        reward_mode=cfg["reward_mode"], kill_reward=cfg["kill_reward"],
        distance_scale=cfg["distance_scale"], health_scale=cfg["health_scale"],
        aim_reward=cfg["aim_reward"], aim_penalty=cfg["aim_penalty"],
        longevity_reward=cfg["longevity_reward"],
        distance_discount=cfg.get("distance_discount", 1.0),
        gate_advance_on_enemy=cfg.get("gate_advance_on_enemy", False),
        gated_discount=cfg.get("gated_discount", 0.05),
        allowed_button_indices=cfg.get("allowed_button_indices"),
        extra_combos=cfg.get("extra_combos"),
        # gate_distance=cfg.get("gate_distance",250.0)
    )

    target_agent = PPOAgent(env, hp) 
    source_ckpt = torch.load(source_path, map_location=target_agent.device, weights_only=False)
    source_state = source_ckpt["actor_critic"]
    target_state = target_agent.actor_critic.state_dict()

    transferred, skipped = [], []
    for name, tensor in source_state.items():
        if name in target_state and target_state[name].shape == tensor.shape:
            target_state[name] = tensor
            transferred.append(name)
        else:
            skipped.append(name)

    target_agent.actor_critic.load_state_dict(target_state)

    print(f"Preneseno {len(transferred)} slojeva: {transferred}")
    print(f"Preskočeno (nova, nasumično inicijalizovana glava): {skipped}")

    target_agent.save(out_path)
    print(f"Sacuvano: {out_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                         help="Checkpoint iz kog vucemo CNN/shared_visual tezine")
    parser.add_argument("--target-scenario", type=str, required=True, choices=list(params.SCENARIOS.keys()),
                         help="Scenario za koji pravimo novi checkpoint")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    transfer(args.source, args.target_scenario, args.out)