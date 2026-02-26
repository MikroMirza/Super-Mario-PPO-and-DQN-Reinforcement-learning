import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import params
from wrappers import make_env
from Agents.Agent import Agent
from Agents.dqn import DQNAgent
from Agents.PPO import PPOAgent

#POMOCNE FUNKCIJE
def moving_average(values, window=100):
    if len(values) < window:
        return np.mean(values)
    return np.mean(values[-window:])

def train(agent: Agent, env, cfg: dict, resume_path: str = None):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    if resume_path:
        agent.load(resume_path)

    print(f"Agent type  : {type(agent).__name__}")
    print(f"Training on : {agent.device}")
    print(f"Max steps   : {cfg['max_steps']:,}")
    print("─" * 50)

    ep_log_path  = os.path.join(cfg["log_dir"], f"training_episodes_{type(agent).__name__}.csv")
    ep_file_exists = resume_path and os.path.exists(ep_log_path)
    ep_csv_file  = open(ep_log_path, "a" if ep_file_exists else "w", newline="")
    ep_writer    = csv.writer(ep_csv_file)

    if not ep_file_exists:
        ep_writer.writerow([
            "step", "episode", "ep_reward", "ep_length",
            "ep_max_x", "flag_get",
        ])

    episode_rewards = []
    episode_lengths = []
    episode_x_positions = []
    metric_history = defaultdict(list)

    state = env.reset()
    ep_reward = 0
    ep_length = 0
    ep_num = 0
    ep_max_x = 0
    total_wins = 0
    t_start = time.time()
    loaded_step = agent.total_steps

    for step in range(loaded_step + 1, cfg["max_steps"] + 1):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        metrics = agent.step(state, action, reward, next_state, done)

        if metrics is not None:
            for k, v in metrics.items():
                metric_history[k].append(v)
        state = next_state
        ep_reward += reward
        ep_length += 1

        current_x = info.get("x_pos", 0)
        if current_x > ep_max_x:
            ep_max_x = current_x

        if done:
            flag = info.get("flag_get", False)
            if flag:
                total_wins += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_x_positions.append(ep_max_x)
            ep_num += 1

            ep_writer.writerow([
                step, ep_num, f"{ep_reward:.2f}", ep_length,
                ep_max_x, int(flag),
            ])

            agent.on_episode_end()

            ep_reward = 0
            ep_length = 0
            ep_max_x  = 0
            state = env.reset()

        if step % cfg["log_freq"] == 0:
            elapsed = time.time() - t_start
            fps = step / elapsed
            avg_r = moving_average(episode_rewards)
            avg_x = moving_average(episode_x_positions) if episode_x_positions else 0

            train_parts = " | ".join(
                f"{k}: {moving_average(v):.4f}"
                for k, v in metric_history.items()
            )

            extras = agent.extra_metrics()
            extras_str = " | ".join(f"{k}: {v:.3f}" for k, v in extras.items())

            parts = [
                f"Step {step:>8,}",
                f"Ep {ep_num:>5}",
                f"Wins: {total_wins:>4}",
                f"Avg R(100): {avg_r:>7.2f}",
                f"Avg X pos: {avg_x:>6.0f}",
            ]
            if train_parts:
                parts.append(train_parts)
            if extras_str:
                parts.append(extras_str)
            parts.append(f"FPS: {fps:>5.0f}")

            print(" | ".join(parts))

        if step % cfg.get("csv_flush_freq", 10_000) == 0:
            ep_csv_file.flush()

        if step % cfg["save_freq"] == 0:
            ckpt = os.path.join(cfg["checkpoint_dir"], f"{type(agent).__name__}_step_{step}.pt")
            agent.save(ckpt)

    env.close()
    ep_csv_file.close()
    final = os.path.join(cfg["checkpoint_dir"], f"{type(agent).__name__}_final.pt")
    agent.save(final)
    print("Training complete.")

#Replay
def evaluate(agent: Agent, env, checkpoint_path, n_episodes=10, render=True):
    info = {}

    agent.load(checkpoint_path)

    FPS = 30
    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done  = False
        total_reward = 0.0

        while not done:
            start = time.time()
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            
            total_reward += reward
            while time.time() - start < 1/FPS:
                pass
        
        agent.on_episode_end()

        print(f"Episode {ep}: reward = {total_reward:.1f}  |  flag = {info.get('flag_get', False)}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Mario Bros - DQN & PPO")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="Which algorithm to run (dqn or ppo)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    args = parser.parse_args()

    env = make_env(
        env_id = params.env_params["env_id"],
        skip = params.env_params["frame_skip"],
        shape = params.env_params["frame_size"],
        stack = params.env_params["frame_stack"],
        clip_rewards = params.env_params["clip_rewards"],
        max_episode_steps = params.env_params["max_ep_steps"]
    )
    state_shape = env.observation_space.shape
    n_actions   = env.action_space.n
    
    agent = None
    if args.algo == "dqn":
        agent = DQNAgent(
                state_shape = state_shape,
                n_actions = n_actions,
                lr = params.dqn_params["lr"],
                gamma = params.dqn_params["gamma"],
                buffer_capacity = params.dqn_params["buffer_capacity"],
                batch_size = params.dqn_params["batch_size"],
                eps_start = params.dqn_params["eps_start"],
                eps_end = params.dqn_params["eps_end"],
                eps_decay_steps = params.dqn_params["eps_decay_steps"],
                target_update_freq = params.dqn_params["target_update_freq"],
                train_freq = params.dqn_params["train_freq"],
                learning_starts = params.dqn_params["learning_starts"]
            )
    else:
        agent = PPOAgent(env, params.params2ndgo)

    if args.eval:
        evaluate(agent, env, args.eval, n_episodes=args.episodes)
    else:
        train(agent, env, params.training_params, args.eval)