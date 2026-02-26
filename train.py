"""
CLI Argumenti:
    python train.py
    python train.py --resume checkpoint.pt
    python train.py --eval checkpoint.pt
"""

import argparse
import csv
import os
import time
from collections import deque
from collections import defaultdict

import numpy as np
import params
import torch

from wrappers import make_env
from Agent import Agent
from dqn import DQNAgent
from params import hyperparameters as PPOhyperparameters
from PPO import PPOAgent

CFG = dict(
    env_id            = "SuperMarioBros-1-1-v0",
    frame_skip        = 4,
    frame_size        = 84,
    frame_stack       = 4,
    clip_rewards      = False,

    #DQN parametri
    lr                  = 2.5e-5,
    gamma               = 0.99,
    buffer_capacity     = 200_000,
    batch_size          = 256,
    eps_start           = 1.0,
    eps_end             = 0.1,
    eps_decay_steps = 500_000,
    target_update_freq = 10_000,
    max_ep_steps = 10_000,

    #Trening konfiguracija
    max_steps = 100_000_000,
    learning_starts = 10_000,   #Koliko koraka treba baferovati pre nego sto trening pocne
    train_freq = 4,        #Na koliko koraka se azurira

    #Logovanje
    # log_freq = 1_000,    #Frekvencija ispisivanja statistike
    log_freq = 1,    #Frekvencija ispisivanja statistike
    save_freq = 50_000,   #Frekvencija cuvanja checkpointa
    ppo_save_freq = 102400,   #Frekvencija cuvanja checkpointa
    csv_flush_freq = 10_000,   #Koliko cesto se upisuje CSV na disk
    checkpoint_dir = "checkpoints",
    log_dir = "logs",
)

#POMOCNE FUNKCIJE

def moving_average(values, window=100):
    if len(values) < window:
        return np.mean(values)
    return np.mean(values[-window:])


#TRENING
def train_ppo(resume_path=None):
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"], exist_ok=True)

    env = make_env(
        env_id=CFG["env_id"],
        skip=CFG["frame_skip"],
        shape=CFG["frame_size"],
        stack=CFG["frame_stack"],
        clip_rewards=CFG["clip_rewards"],
        max_episode_steps=CFG["max_ep_steps"]
    )

    agent = PPOAgent(env, PPOhyperparameters)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=agent.device)
        agent.actor_critic.load_state_dict(checkpoint["actor_critic"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        total_steps_done = checkpoint.get("total_steps", 0)
        print(f"[PPO] Resumed from step {total_steps_done}")
    else:
        total_steps_done = 0

    # Logovanje
    log_path = os.path.join(CFG["log_dir"], "ppo_episodes.csv")
    file_exists = resume_path and os.path.exists(log_path)
    csv_file = open(log_path, "a" if file_exists else "w", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["step", "episode", "ep_reward", "ep_max_x", "flag_get"])
    ep_num = 0
    total_wins = 0
    t_start = time.time()

    while total_steps_done < CFG["max_steps"]:

        agent.collect_data()
        total_steps_done += PPOhyperparameters['n_steps']
        agent.total_steps = total_steps_done #checkpointi
        #Loguju se epizode koje zavrse
        for ep_reward, ep_max_x, flag in agent.finished_episodes:
            ep_num += 1
            if flag:
                total_wins += 1
            writer.writerow([total_steps_done, ep_num, f"{ep_reward:.2f}", ep_max_x, int(flag)])

        last_state = torch.tensor(agent.current_state, dtype=torch.float32).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            _, last_value = agent.actor_critic(last_state)

        values_with_bootstrap = agent.values + [last_value.squeeze()]
        advantages, returns = agent.compute_advantage(agent.rewards, values_with_bootstrap, agent.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states        = torch.stack(agent.states).to(agent.device)
        actions       = torch.stack(agent.actions).to(agent.device)
        old_log_probs = torch.stack(agent.log_probs).to(agent.device)

        agent.update_policy(states, actions, old_log_probs, advantages, returns)
        #Logovanje opet
        # if total_steps_done % CFG["log_freq"] == 0:
        #     elapsed           = time.time() - t_start
        #     fps               = total_steps_done / elapsed
        #     recent_rewards    = [ep[0] for ep in agent.finished_episodes[-100:]]
        #     recent_x          = [ep[1] for ep in agent.finished_episodes[-100:]]
        #     avg_r             = np.mean(recent_rewards) if recent_rewards else float("nan")
        #     avg_x             = np.mean(recent_x)       if recent_x      else 0
        #     avg_l             = moving_average(agent.recent_losses) if agent.recent_losses else float("nan")

        #     print(
        #         f"Step             {total_steps_done:>8,} | "
        #         f"Ep               {ep_num:>5}            | "
        #         f"Wins:            {total_wins:>4}        | "
        #         f"Avg R(100):      {avg_r:>7.2f}          | "
        #         f"Avg X pos:       {avg_x:>6.0f}          | "
        #         f"Loss:            {avg_l:.4f}            | "
        #         f"FPS:             {fps:>5.0f}"
        #     )

        if total_steps_done % CFG["ppo_save_freq"] == 0:
            ckpt_path = os.path.join(CFG["checkpoint_dir"], f"ppo_step_{total_steps_done}.pt")
            torch.save({
                "actor_critic": agent.actor_critic.state_dict(),
                "optimizer":    agent.optimizer.state_dict(),
                "total_steps":  total_steps_done,
            }, ckpt_path)
        csv_file.flush()
    env.close()
    csv_file.close()

    torch.save({
        "actor_critic": agent.actor_critic.state_dict(),
        "optimizer":    agent.optimizer.state_dict(),
        "total_steps":  total_steps_done,
    }, os.path.join(CFG["checkpoint_dir"], "ppo_final.pt"))
    print("PPO Training complete.")

def train(agent: Agent, env, cfg: dict, resume_path: str = None):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    if resume_path:
        agent.load(resume_path)

    print(f"Agent type  : {type(agent).__name__}")
    print(f"Training on : {agent.device}")
    print(f"Max steps   : {cfg['max_steps']:,}")
    print("─" * 50)

    ep_log_path  = os.path.join(cfg["log_dir"], "training_episodes.csv")
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

def evaluate(checkpoint_path, n_episodes=10, render=True):
    env = make_env(
        env_id       = CFG["env_id"],
        skip         = CFG["frame_skip"],
        shape        = CFG["frame_size"],
        stack        = CFG["frame_stack"],
        clip_rewards = False,
    )

    state_shape = env.observation_space.shape
    n_actions   = env.action_space.n

    agent = DQNAgent(
        state_shape = state_shape,
        n_actions   = n_actions,
        eps_start   = 0.10,
        eps_end     = 0.10,
    )
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
            state, reward, done, info = env.step(action)
            total_reward += reward
            while time.time() - start < 1/FPS:
                s = 1

        print(f"Episode {ep}: reward = {total_reward:.1f}  |  flag = {info.get('flag_get', False)}")

    env.close()

def evaluate_ppo(checkpoint_path, n_episodes=10, render=True):
    env = make_env(
        env_id       = CFG["env_id"],
        skip         = CFG["frame_skip"],
        shape        = CFG["frame_size"],
        stack        = CFG["frame_stack"],
        clip_rewards = False,
    )

    agent = PPOAgent(env, PPOhyperparameters)

    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.actor_critic.load_state_dict(checkpoint["actor_critic"])
    agent.actor_critic.eval()

    print(f"[PPO] Loaded checkpoint: {checkpoint_path}")

    FPS = 30

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            start = time.time()

            if render:
                env.render()
            #No sampling, justthe greediest action
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                distribution, _ = agent.actor_critic(state_t)
                action = distribution.probs.argmax(dim=1).item()

            state, reward, done, info = env.step(action)
            total_reward += reward

            while time.time() - start < 1 / FPS:
                pass

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
        env_id=CFG["env_id"],
        skip=CFG["frame_skip"],
        shape=CFG["frame_size"],
        stack=CFG["frame_stack"],
        clip_rewards=CFG["clip_rewards"],
        max_episode_steps=CFG["max_ep_steps"]
    )

    if args.algo == "dqn":
        if args.eval:
            evaluate(args.eval, n_episodes=args.episodes)
        else:
            state_shape = env.observation_space.shape
            n_actions   = env.action_space.n
            agent = DQNAgent(
                state_shape        = state_shape,
                n_actions          = n_actions,
                lr                 = CFG["lr"],
                gamma              = CFG["gamma"],
                buffer_capacity    = CFG["buffer_capacity"],
                batch_size         = CFG["batch_size"],
                eps_start          = CFG["eps_start"],
                eps_end            = CFG["eps_end"],
                eps_decay_steps    = CFG["eps_decay_steps"],
                target_update_freq = CFG["target_update_freq"],
            )
            train(agent, env, CFG)

    elif args.algo == "ppo":
        if args.eval:
            evaluate_ppo(args.eval, n_episodes=args.episodes)
        else:
            agent = PPOAgent(env, params.params)
            train(agent, env, CFG)