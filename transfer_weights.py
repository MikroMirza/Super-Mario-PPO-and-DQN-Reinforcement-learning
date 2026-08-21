"""
KORAK 2: Transfer learning izmedju scenarija sa razlicitim brojem akcija.

Problem: standardni agent.load() ne radi izmedju defend_the_center (3 akcije)
i deadly_corridor (7 akcija) jer se oblik actor/critic izlaznog sloja ne
poklapa (torch baca RuntimeError o mismatch-u velicina tenzora).

Resenje: CNN + shared_visual slojevi (feature extractor) ne zavise od broja
akcija - isti su bez obzira na scenario. Ovaj skript ucitava SAMO te slojeve
iz jednog checkpoint-a u svez model za drugi scenario, i nasumicno
inicijalizuje actor/critic glave (koje MORAJU biti nove jer je n_actions
razlicit). Ideja: agent koji je vec naucio da prepoznaje monstrume i
gadja u defend_the_center ne mora da uci "vid" od nule za deadly_corridor -
samo mu treba nova "odluka sta da radi sa tim vidom" (actor/critic glave).

Upotreba:
    python3 transfer_weights.py --source checkpoints/PPOAgent_defend_best.pt \
                                 --target-scenario corridor \
                                 --out checkpoints/corridor_init_from_defend.pt
"""
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
                         help="Checkpoint iz kog vucemo CNN/shared_visual tezine (npr. defend checkpoint)")
    parser.add_argument("--target-scenario", type=str, required=True, choices=list(params.SCENARIOS.keys()),
                         help="Scenario za koji pravimo novi (delimicno prenet) checkpoint")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    transfer(args.source, args.target_scenario, args.out)