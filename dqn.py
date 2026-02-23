import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # -> 7x7
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_output_size(in_channels)

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        v = self.value(x)
        a = self.advantage(x)
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        return v + (a - a.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_shape,        # (C, H, W)
        n_actions: int,
        lr: float           = 2.5e-5,
        gamma: float        = 0.99,
        buffer_capacity: int = 200_000,
        batch_size: int     = 256,
        eps_start: float    = 1.0,
        eps_end: float      = 0.05,
        eps_decay_steps: int = 500_000,
        target_update_freq: int = 10_000,
        device: str         = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        in_channels = state_shape[0]
        self.online_net = QNetwork(in_channels, n_actions).to(self.device)
        self.target_net = QNetwork(in_channels, n_actions).to(self.device)
        self._sync_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)

        self.total_steps = 0

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @property
    def epsilon(self) -> float:
        fraction = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.from_numpy(states).to(self.device, non_blocking=True)
        next_states_t = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        actions_t     = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards_t     = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        dones_t       = torch.from_numpy(dones).to(self.device, non_blocking=True)

        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        current_q = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )
        print(f"[Agent] Checkpoint saved: {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        print(f"[Agent] Checkpoint loaded: {path}  (step {self.total_steps})")