from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from Agent import Agent

params = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.1,
    'n_epochs':        4,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.05,
    'value_loss_coef': 0.5,
}

class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(ActorCritic, self).__init__()

        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("C1",    nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)),
                    ("ReLU1", nn.ReLU()),
                    ("C2",    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                    ("ReLU2", nn.ReLU()),
                    ("C3",    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                    ("ReLU3", nn.ReLU()),
                    ("flat",  nn.Flatten())
                ]
            )
        )

        cnn_output_size = 64 * 7 * 7

        self.shared_visual = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU()
        )

        self.actor  = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, state):
        # Normalize pixels from [0,255] to [0,1]
        if state.dtype == torch.uint8:
            state = state.float() / 255.0
        feature = self.cnn(state)
        feature = self.shared_visual(feature)

        logits = self.actor(feature)   # raw scores for each of the 7 actions
        value  = self.critic(feature)  # estimated total future reward from this state
        distribution = Categorical(logits=logits)  # Uses softmax to normalize to [0,1]

        return distribution, value


class PPOAgent(Agent):
    def __init__(self, env, hyperparameters):
        state_shape = env.observation_space.shape  # (4, 84, 84)
        n_actions   = env.action_space.n

        super().__init__(state_shape, n_actions)

        self.hyperparams  = hyperparameters
        self.actor_critic = ActorCritic(state_shape[0], n_actions).to(self.device)
        self.optimizer    = optim.Adam(self.actor_critic.parameters(), lr=hyperparameters['learning_rate'])

        # Every n_steps, refilled
        self.states    = []
        self.actions   = []
        self.values    = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []

        self.finished_episodes = []  # list of (ep_reward, ep_max_x, flag_get)
        self.current_ep_reward = 0
        self.ep_max_x          = 0
        self.recent_losses     = []

        # Store last state/action/value between select_action() and step() calls
        self._last_state_tensor = None
        self._last_action       = None
        self._last_log_prob     = None
        self._last_value        = None

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distribution, value = self.actor_critic(state_tensor)

        action   = distribution.sample()
        log_prob = distribution.log_prob(action)

        # Cache these so step() can store them in buffers
        self._last_state_tensor = state_tensor.squeeze(0)
        self._last_action       = action.squeeze(0)
        self._last_log_prob     = log_prob.squeeze(0)
        self._last_value        = value.squeeze()

        return action.item()

    def step(self, state, action, reward, next_state, done) -> dict | None:
        # Tracking x-es for training/logging
        self.current_ep_reward += reward

        self.states.append(self._last_state_tensor)
        self.actions.append(self._last_action)
        self.log_probs.append(self._last_log_prob)
        self.rewards.append(reward)
        self.values.append(self._last_value)
        self.dones.append(int(done))

        # Only update when we have a full batch of n_steps
        if len(self.states) < self.hyperparams['n_steps']:
            return None

        # Bootstrap value of whatever state we're currently in
        next_state_tensor = torch.tensor(
            next_state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, last_value = self.actor_critic(next_state_tensor)

        values_with_bootstrap = self.values + [last_value.squeeze()]
        advantages, returns   = self.compute_advantage(self.rewards, values_with_bootstrap, self.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states        = torch.stack(self.states).to(self.device)
        actions       = torch.stack(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)

        self.update_policy(states, actions, old_log_probs, advantages, returns)

        # New 'state' new data, so we clear everything
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

        avg_loss = np.mean(self.recent_losses[-16:]) if self.recent_losses else float("nan")
        return {"ppo_loss": avg_loss}

    def on_episode_end(self) -> None:
        self.current_ep_reward = 0
        self.ep_max_x          = 0

    def extra_metrics(self) -> dict:
        return {"loss": self.recent_losses[-1] if self.recent_losses else float("nan")}

    def compute_advantage(self, rewards, values, dones):
        n             = len(rewards)
        advantages    = torch.zeros(n).to(self.device)
        gae           = 0.0
        values_tensor = torch.stack(values)  #(n_steps+1,)

        for t in reversed(range(n)):
            delta = (rewards[t] + self.hyperparams['gamma'] * values_tensor[t + 1].item() * (1 - dones[t]) - values_tensor[t].item())

            gae = delta + self.hyperparams['gamma'] * self.hyperparams['gae_lambda'] * (1 - dones[t]) * gae
            advantages[t] = gae

        # This trains the critic. It's what he SHOULD predict.
        returns = advantages + values_tensor[:-1]

        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        n = states.shape[0]

        for _ in range(self.hyperparams['n_epochs']):
            indices = torch.randperm(n)

            for start in range(0, n, self.hyperparams['batch_size']):
                batch_idx = indices[start:start + self.hyperparams['batch_size']]

                batch_states     = states[batch_idx]
                batch_actions    = actions[batch_idx]
                batch_old_lp     = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns    = returns[batch_idx]

                distribution, new_values = self.actor_critic(batch_states)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy       = distribution.entropy().mean()

                #This tracks how much the policy changed after collecting the data
                probability_ratio = torch.exp(new_log_probs - batch_old_lp)

                #Updates too large get clipped so it doesn't corrupt the cnn
                clipped_ratio = torch.clamp(
                    probability_ratio,
                    1 - self.hyperparams['clip_epsilon'],
                    1 + self.hyperparams['clip_epsilon']
                )
                actor_loss = -torch.min(
                    probability_ratio * batch_advantages,
                    clipped_ratio     * batch_advantages
                ).mean()

                new_values = new_values.squeeze(-1)
                value_loss = nn.MSELoss()(new_values, batch_returns)

                loss = (actor_loss
                        + self.hyperparams['value_loss_coef'] * value_loss
                        - self.hyperparams['entropy_coef']    * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

                self.recent_losses.append(loss.item())

    def save(self, path: str) -> None:
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "total_steps":  self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
