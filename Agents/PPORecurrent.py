"""
KORAK 3: PPO sa LSTM slojem posle CNN-a.

Zasto: trenutni agent ima "pamcenje" samo kroz frame_stack=4 poslednja
frejma. Kod deadly_corridor-a sa 4 neprijatelja u 2 para (levo/desno),
agent gubi informaciju o neprijatelju cim mu izadje iz vidokruga (npr.
kad se okrene da gadja drugu stranu) - nema nacina da "zapamti" da je
tamo iza njega jos jedan protivnik. LSTM cuva skriveno stanje kroz CELU
epizodu (ne samo poslednja 4 frejma), sto bi trebalo da pomogne bas
kod ovakvih "flanking" scenarija.

VAZNA NAPOMENA O ISPRAVNOSTI: LSTM zahteva da se batch obradjuje
HRONOLOSKI (ne sme se mesati/shuffle-ovati kao kod obicnog PPO), i
skriveno stanje mora da se resetuje na pocetku svake nove epizode
(kad je done=True). Ako se to ne uradi ispravno, trening je tiho
pokvaren (radi bez greske, ali uci pogresne stvari). Ovaj fajl to
resava tako sto:
  1. Pamti POCETNO skriveno stanje na pocetku svakog rollout-a (n_steps)
  2. Pri update-u, kroz ceo rollout ide REDOM (bez safl-ovanja),
     resetujuci skriveno stanje na nulu gde god je dones[t-1] == True.
"""


from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from Agents.Agent import Agent


class ActorCriticLSTM(nn.Module):
    def __init__(self, input_channels, n_actions, lstm_hidden=256):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        self.cnn = nn.Sequential(
            OrderedDict([
                ("C1",    nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
                ("ReLU1", nn.ReLU()),
                ("C2",    nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                ("ReLU2", nn.ReLU()),
                ("C3",    nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                ("ReLU3", nn.ReLU()),
                ("flat",  nn.Flatten()),
            ])
        )
        cnn_output_size = 64 * 7 * 7

        self.shared_visual = nn.Sequential(nn.Linear(cnn_output_size, 512), nn.ReLU())
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden, batch_first=True)

        self.actor  = nn.Linear(lstm_hidden, n_actions)
        self.critic = nn.Linear(lstm_hidden, 1)

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def forward_step(self, state, hidden):
        """Jedan vremenski korak (za select_action tokom rollout-a). state: (1, C, H, W)."""
        if state.dtype == torch.uint8:
            state = state.float() / 255.0
        feat = self.shared_visual(self.cnn(state))
        feat = feat.unsqueeze(1)
        lstm_out, hidden = self.lstm(feat, hidden)
        lstm_out = lstm_out.squeeze(1)

        logits = self.actor(lstm_out)
        value  = self.critic(lstm_out)
        return Categorical(logits=logits), value, hidden

    def forward_sequence(self, states, dones, initial_hidden):
        """
        Obradjuje CEO rollout hronoloski, resetujuci hidden state na
        pocetku svake epizode (gde je prethodni done=True).
        """
        T = states.shape[0]
        h, c = initial_hidden
        h, c = h.clone(), c.clone()

        if states.dtype == torch.uint8:
            states = states.float() / 255.0
        feats = self.shared_visual(self.cnn(states))

        outputs = []
        for t in range(T):
            x_t = feats[t].view(1, 1, -1)
            out_t, (h, c) = self.lstm(x_t, (h, c))
            outputs.append(out_t.squeeze(0).squeeze(0))
            if dones[t] == 1 and t < T - 1:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)

        lstm_out = torch.stack(outputs, dim=0)
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out).squeeze(-1)
        return Categorical(logits=logits), values


class PPORecurrentAgent(Agent):
    def __init__(self, env, hyperparameters, lstm_hidden=256):
        state_shape = env.observation_space.shape
        n_actions   = env.action_space.n
        super().__init__(state_shape, n_actions)

        self.hyperparams  = hyperparameters
        self.actor_critic = ActorCriticLSTM(state_shape[0], n_actions, lstm_hidden).to(self.device)
        self.optimizer    = optim.Adam(self.actor_critic.parameters(), lr=hyperparameters["learning_rate"])

        self.states, self.actions, self.values = [], [], []
        self.log_probs, self.rewards, self.dones = [], [], []

        self.recent_losses, self.recent_entropies = [], []
        self.current_ep_reward = 0

        self._hidden = self.actor_critic.init_hidden(device=self.device)
        self._rollout_start_hidden = self._hidden

        self._last_state_tensor = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, value, new_hidden = self.actor_critic.forward_step(state_tensor, self._hidden)

        action   = dist.sample()
        log_prob = dist.log_prob(action)

        self._last_state_tensor = state_tensor.squeeze(0)
        self._last_action       = action.squeeze(0)
        self._last_log_prob     = log_prob.squeeze(0)
        self._last_value        = value.squeeze()
        self._hidden = new_hidden

        return action.item()

    def step(self, state, action, reward, next_state, done):
        self.current_ep_reward += reward

        self.states.append(self._last_state_tensor)
        self.actions.append(self._last_action)
        self.log_probs.append(self._last_log_prob)
        self.rewards.append(reward)
        self.values.append(self._last_value)
        self.dones.append(int(done))

        if done:
            self._hidden = self.actor_critic.init_hidden(device=self.device)

        if len(self.states) < self.hyperparams["n_steps"]:
            return None

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value, _ = self.actor_critic.forward_step(next_state_tensor, self._hidden)

        values_with_bootstrap = self.values + [last_value.squeeze()]
        advantages, returns = self.compute_advantage(self.rewards, values_with_bootstrap, self.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states  = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        dones_arr = self.dones

        self.update_policy(states, actions, old_log_probs, advantages, returns, dones_arr)

        self._rollout_start_hidden = (self._hidden[0].clone(), self._hidden[1].clone())

        self.states.clear(); self.actions.clear(); self.values.clear()
        self.log_probs.clear(); self.rewards.clear(); self.dones.clear()

        avg_loss = np.mean(self.recent_losses[-16:]) if self.recent_losses else float("nan")
        return {"ppo_loss": avg_loss}

    def on_episode_end(self):
        self.current_ep_reward = 0

    def extra_metrics(self):
        return {
            "loss": self.recent_losses[-1] if self.recent_losses else float("nan"),
            "entropy": np.mean(self.recent_entropies[-16:]) if self.recent_entropies else float("nan"),
        }

    def compute_advantage(self, rewards, values, dones):
        n = len(rewards)
        advantages = torch.zeros(n).to(self.device)
        gae = 0.0
        values_tensor = torch.stack(values)
        for t in reversed(range(n)):
            delta = rewards[t] + self.hyperparams["gamma"] * values_tensor[t + 1].item() * (1 - dones[t]) - values_tensor[t].item()
            gae = delta + self.hyperparams["gamma"] * self.hyperparams["gae_lambda"] * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values_tensor[:-1]
        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, advantages, returns, dones):
        # NAPOMENA: LSTM zahteva HRONOLOSKI redosled - nema shuffle-a kao u PPO.py
        for _ in range(self.hyperparams["n_epochs"]):
            dist, new_values = self.actor_critic.forward_sequence(states, dones, self._rollout_start_hidden)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.hyperparams["clip_epsilon"], 1 + self.hyperparams["clip_epsilon"])
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_loss = nn.MSELoss()(new_values, returns)

            loss = (actor_loss
                    + self.hyperparams["value_loss_coef"] * value_loss
                    - self.hyperparams["entropy_coef"] * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

            self.recent_losses.append(loss.item())
            self.recent_entropies.append(entropy.item())

    def save(self, path: str) -> None:
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)