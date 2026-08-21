"""
KORAK 1: Sanity-check pipeline-a na vise scenarija.

Puni PPO trening na svih 5 mapa bi trajao predugo, ali kratka evaluacija
(par hiljada nasumicnih/malo-treniranih koraka) ti brzo kaze da li je
problem u tvom PPO kodu/arhitekturi (los rezultat svuda) ili specificno
u tezini deadly_corridor-a (dobar rezultat na lakim mapama, los na
corridor-u).

Upotreba:
    python3 eval_multi_scenario.py                          # nasumicna politika, brz sanity check
    python3 eval_multi_scenario.py --checkpoint path.pt --scenario defend
    python3 eval_multi_scenario.py --all --episodes 20       # sve mape, isti (nasumicni) agent
"""

import argparse

import numpy as np

import params
from wrappers import make_env
from Agents.PPO import PPOAgent


def build_env(scenario):
    cfg, _ = params.SCENARIOS[scenario]
    return make_env(
        env_id=cfg["env_id"], skip=cfg["frame_skip"], shape=cfg["frame_size"],
        stack=cfg["frame_stack"], clip_rewards=cfg["clip_rewards"],
        max_episode_steps=cfg["max_ep_steps"], window_visible=cfg["window_visible"],
        reward_mode=cfg["reward_mode"], kill_reward=cfg["kill_reward"],
        distance_scale=cfg["distance_scale"], health_scale=cfg["health_scale"],
        aim_reward=cfg["aim_reward"], aim_penalty=cfg["aim_penalty"],
        longevity_reward=cfg["longevity_reward"],
    )


def evaluate(scenario, n_episodes=10, checkpoint=None):
    env = build_env(scenario)
    _, hp = params.SCENARIOS[scenario]
    agent = PPOAgent(env, hp)
    if checkpoint:
        try:
            agent.load(checkpoint)
        except RuntimeError as e:
            print(f"[{scenario}] Ne moze da ucita checkpoint {e}")
            return None

    rewards, lengths, kills, deaths = [], [], [], []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_r, ep_len, ep_k = 0.0, 0, 0
        while not done:
            action = agent.select_action(state)
            state, r, done, info = env.step(action)
            ep_r += r
            ep_len += 1
            ep_k = info.get("kills", ep_k)
        rewards.append(ep_r)
        lengths.append(ep_len)
        kills.append(ep_k)
        deaths.append(int(info.get("died", False)))
    env.close()

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths)),
        "avg_kills":  float(np.mean(kills)),
        "death_rate": float(np.mean(deaths)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None, choices=list(params.SCENARIOS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Putanja do .pt checkpoint-a. Radi SAMO ako je checkpoint "
                              "treniran na scenariju sa istim brojem akcija (vidi napomenu ispod).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Testiraj sve scenarije redom")
    args = parser.parse_args()

    scenarios = list(params.SCENARIOS.keys()) if args.all else [args.scenario or "defend"]

    print(f"{'Scenario':<10} {'AvgReward':>10} {'AvgLen':>8} {'AvgKills':>9} {'DeathRate':>10}")
    for s in scenarios:
        res = evaluate(s, n_episodes=args.episodes, checkpoint=args.checkpoint)
        if res is None:
            continue
        print(f"{s:<10} {res['avg_reward']:>10.2f} {res['avg_length']:>8.1f} "
              f"{res['avg_kills']:>9.2f} {res['death_rate']:>10.2f}")