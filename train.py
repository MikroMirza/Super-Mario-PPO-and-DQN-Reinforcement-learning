import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import params
from wrappers import make_env
from Agents.Agent import Agent
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

    ep_log_path  = os.path.join(cfg["log_dir"], f"training_episodes_{type(agent).__name__}_transferred_corridor.csv")
    ep_file_exists = resume_path and os.path.exists(ep_log_path)
    ep_csv_file  = open(ep_log_path, "a" if ep_file_exists else "w", newline="")
    ep_writer    = csv.writer(ep_csv_file)

    if not ep_file_exists:
        ep_writer.writerow([
            "step",
            "episode",
            "ep_reward",
            "ep_length",
            "ep_kills",
            "ep_max_x",
            "healthpacks_picked",
            "health_gained",
            "final_health",
            "died",
        ])

    episode_rewards = []
    episode_lengths = []
    episode_kills = []
    episode_max_x = []
    episode_healthpacks = []
    episode_health_gained = []
    metric_history = defaultdict(list)

    state = env.reset()
    ep_reward = 0
    ep_length = 0
    ep_num = 0
    ep_kills = 0
    ep_max_x = 0
    total_deaths = 0

    ep_healthpacks = 0
    ep_health_gained = 0.0
    final_health = 0.0
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

        ep_kills = info.get("kills", ep_kills)
        ep_max_x = max(ep_max_x, info.get("x_pos", 0))

        ep_healthpacks = info.get("healthpacks_picked", ep_healthpacks)
        ep_health_gained = info.get("health_gained", ep_health_gained)
        final_health = info.get("health", final_health)

        if done:
            died = info.get("died", False)
            if died:
                total_deaths += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_kills.append(ep_kills)
            episode_max_x.append(ep_max_x)
            episode_healthpacks.append(ep_healthpacks)
            episode_health_gained.append(ep_health_gained)  
            ep_num += 1

            ep_writer.writerow([
                step,
                ep_num,
                f"{ep_reward:.2f}",
                ep_length,
                ep_kills,
                f"{ep_max_x:.1f}",
                ep_healthpacks,
                f"{ep_health_gained:.1f}",
                f"{final_health:.1f}",
                int(died),
            ])

            agent.on_episode_end()

            ep_reward = 0
            ep_length = 0
            ep_kills  = 0
            ep_max_x  = 0

            ep_healthpacks = 0
            ep_health_gained = 0.0
            final_health = 0.0
            state = env.reset()

        if step % cfg["log_freq"] == 0:
            elapsed = time.time() - t_start
            fps = step / elapsed
            avg_r = moving_average(episode_rewards)
            avg_kills = moving_average(episode_kills) if episode_kills else 0
            avg_x = moving_average(episode_max_x) if episode_max_x else 0
            avg_healthpacks = (
            moving_average(episode_healthpacks)
                if episode_healthpacks else 0
            )

            avg_health_gained = (
                moving_average(episode_health_gained)
                if episode_health_gained else 0
            )
            train_parts = " | ".join(
                f"{k}: {moving_average(v):.4f}"
                for k, v in metric_history.items()
            )

            extras = agent.extra_metrics()
            extras_str = " | ".join(f"{k}: {v:.3f}" for k, v in extras.items())

            parts = [
                f"Step {step:>8,}",
                f"Ep {ep_num:>5}",
                f"Deaths: {total_deaths:>4}",
                f"Avg R(100): {avg_r:>7.2f}",
                f"Avg Kills: {avg_kills:>6.2f}",
                f"Avg X: {avg_x:>7.1f}",
                f"Avg HP Packs: {avg_healthpacks:>5.2f}",
                f"Avg HP Gain: {avg_health_gained:>6.1f}",
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

        print(f"Episode {ep}: reward = {total_reward:.1f}  |  kills = {info.get('kills', 0)}  |  died = {info.get('died', False)}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doom PPO Agent")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "ppo"],
                        help="Algoritam za treniranje/evaluaciju (dqn ili ppo)")
    # parser.add_argument("--scenario", type=str, default="defend", choices=["defend", "corridor"],
    #                     help="defend = defend_the_center (kills) | corridor = deadly_corridor (presi hodnik)")
    parser.add_argument("--scenario", type=str, default="defend",
                     choices=list(params.SCENARIOS.keys()))
    args = parser.parse_args()

    env_cfg, hyperparams = params.SCENARIOS[args.scenario]

    env = make_env(
        env_id=env_cfg["env_id"],
        skip=env_cfg["frame_skip"],
        shape=env_cfg["frame_size"],
        stack=env_cfg["frame_stack"],
        clip_rewards=env_cfg["clip_rewards"],
        max_episode_steps=env_cfg["max_ep_steps"],
        window_visible=env_cfg["window_visible"],

        reward_mode=env_cfg["reward_mode"],
        kill_reward=env_cfg["kill_reward"],
        distance_scale=env_cfg["distance_scale"],
        health_scale=env_cfg["health_scale"],
        aim_reward=env_cfg["aim_reward"],
        aim_penalty=env_cfg["aim_penalty"],

        distance_discount=env_cfg.get("distance_discount", 1.0),
        gate_advance_on_enemy=env_cfg.get("gate_advance_on_enemy", False),
        gated_discount=env_cfg.get("gated_discount", 0.05),
        allowed_button_indices=env_cfg.get("allowed_button_indices"),
        extra_combos=env_cfg.get("extra_combos"),
        longevity_reward=env_cfg.get("longevity_reward", 0.0),
    )
    state_shape = env.observation_space.shape
    n_actions   = env.action_space.n
    
    agent = None
    if args.algo == "dqn":
        pass
    else:
        agent = PPOAgent(env, hyperparams)

    if args.eval:
        evaluate(agent, env, args.eval, n_episodes=args.episodes)
    else:
        train(agent, env, params.training_params, args.resume)