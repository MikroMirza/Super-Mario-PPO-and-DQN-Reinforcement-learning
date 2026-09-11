"""
train_recurrent.py - isto sto i train.py, samo za PPORecurrentAgent (LSTM).
Odvojen fajl da ne diramo tvoj postojeci train.py.

Upotreba:
    python3 train_recurrent.py --scenario corridor
    python3 train_recurrent.py --scenario corridor --resume checkpoints/recurrent_corridor_step_50000.pt
    python3 train_recurrent.py --scenario corridor --eval checkpoints/recurrent_corridor_step_50000.pt --episodes 5
"""
import argparse
import csv
import os

import numpy as np

import params
from wrappers import make_env
from Agents.PPORecurrent import PPORecurrentAgent


def moving_average(values, window=100):
    if len(values) < window:
        return np.mean(values) if values else 0.0
    return np.mean(values[-window:])


def build_env(scenario):
    cfg, hp = params.SCENARIOS[scenario]
    env = make_env(
        env_id=cfg["env_id"], skip=cfg["frame_skip"], shape=cfg["frame_size"],
        stack=cfg["frame_stack"], clip_rewards=cfg["clip_rewards"],
        max_episode_steps=cfg["max_ep_steps"], window_visible=cfg["window_visible"],
        reward_mode=cfg["reward_mode"], kill_reward=cfg["kill_reward"],
        distance_scale=cfg["distance_scale"], health_scale=cfg["health_scale"],
        aim_reward=cfg["aim_reward"], aim_penalty=cfg["aim_penalty"],
        longevity_reward=cfg["longevity_reward"], gate_distance=cfg.get("gate_distance", 250.0),
    )
    return env, hp


def train(agent, env, cfg, scenario, resume_path=None):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    if resume_path:
        agent.load(resume_path)
        print(f"Ucitan checkpoint: {resume_path} (total_steps={agent.total_steps})")

    csv_path = os.path.join(cfg["log_dir"], f"training_episodes_Recurrent_{scenario}.csv")
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    ep_writer = csv.writer(csv_file)
    if not file_exists:
        ep_writer.writerow(["step", "episode", "ep_reward", "ep_length", "ep_kills", "ep_max_x", "died"])

    episode_rewards, episode_lengths, episode_kills, episode_max_x = [], [], [], []
    best_avg_reward = -float("inf")

    state = env.reset()
    ep_reward, ep_length, ep_num, ep_kills, ep_max_x, total_deaths = 0, 0, 0, 0, 0, 0
    step = agent.total_steps

    while step < cfg["max_steps"]:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        step += 1
        agent.total_steps = step
        ep_reward += reward
        ep_length += 1
        ep_kills = info.get("kills", ep_kills)
        ep_max_x = max(ep_max_x, info.get("x_pos", 0))

        if done:
            died = info.get("died", False)
            if died:
                total_deaths += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_kills.append(ep_kills)
            episode_max_x.append(ep_max_x)
            ep_num += 1

            ep_writer.writerow([step, ep_num, f"{ep_reward:.2f}", ep_length, ep_kills, f"{ep_max_x:.1f}", int(died)])

            agent.on_episode_end()
            ep_reward, ep_length, ep_kills, ep_max_x = 0, 0, 0, 0
            state = env.reset()

        if step % cfg["csv_flush_freq"] == 0:
            csv_file.flush()

        if step % cfg["log_freq"] == 0 and episode_rewards:
            avg_r = moving_average(episode_rewards)
            avg_kills = moving_average(episode_kills)
            avg_x = moving_average(episode_max_x)
            extras = agent.extra_metrics()
            extras_str = " | ".join(f"{k}: {v:.3f}" for k, v in extras.items())
            print(f"Step {step:>8,} | Ep {ep_num:>5} | Deaths: {total_deaths:>4} | "
                  f"Avg R(100): {avg_r:>7.2f} | Avg Kills: {avg_kills:>6.2f} | "
                  f"Avg X: {avg_x:>7.1f} | {extras_str}")

            # cuvaj best checkpoint odvojeno od periodicnog - stiti te od
            # kolapsa/regresije koju smo vec vise puta videli u ovoj sesiji
            if avg_r > best_avg_reward and ep_num > 20:
                best_avg_reward = avg_r
                best_path = os.path.join(cfg["checkpoint_dir"], f"Recurrent_{scenario}_BEST.pt")
                agent.save(best_path)
                print(f"  -> Novi best checkpoint ({avg_r:.2f}): {best_path}")

        if step % cfg["save_freq"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"Recurrent_{scenario}_step_{step}.pt")
            agent.save(ckpt_path)

    csv_file.close()


def evaluate(agent, env, checkpoint, n_episodes=5):
    agent.load(checkpoint)
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep}: reward = {total_reward:.1f}  |  kills = {info.get('kills', 0)}  |  "
              f"x_pos = {info.get('x_pos', 0):.1f}  |  died = {info.get('died', False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="corridor", choices=list(params.SCENARIOS.keys()))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                         help="Override za checkpoints folder (npr. checkpoints/corridor_lstm)")
    args = parser.parse_args()

    env, hp = build_env(args.scenario)
    agent = PPORecurrentAgent(env, hp)

    training_cfg = dict(params.training_params)
    if args.checkpoint_dir:
        training_cfg["checkpoint_dir"] = args.checkpoint_dir

    if args.eval:
        evaluate(agent, env, args.eval, n_episodes=args.episodes)
    else:
        train(agent, env, training_cfg, args.scenario, resume_path=args.resume)