from typing import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(ActorCritic, self).__init__()

        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1",  nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)),
                    ("relu1",  nn.ReLU()),
                    ("conv2"), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), #(64,9,9)
                    ("relu2"), nn.ReLU(),
                    ("conv3"), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), #(64,7,7),
                    ("relu3"), nn.ReLU(),
                    ("flatten", nn.Flatten())
                ]
            )
        )

        cnn_output_size = 64*7*7

        self.shared_visual = nn.Sequential(
            nn.Linear(cnn_output_size,512),
            nn.ReLU
        )
        
        self.actor_head = nn.Linear(512,n_actions)
        self.critic_head = nn.Linear(512,1)
        pass   

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() /255.0
        feature = self.cnn(x)
        feature = self.shared_visual(feature)

        #Both actor and critic observe same state
        logits = self.actor_head(feature)
        value = self.critic_head(feature)
        distribution = Categorical(logits = logits)
        
        return distribution, value
        

class PPO:
    def __init__(self, env, hyperparameters):
        self.environment = env
        self.hyperparams = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_actions = env.action_space.n
        input_channels = env.observation_space.shape[0] 

        self.actor_critic = ActorCritic(input_channels, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=hyperparameters['learning_rate'])
        
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

        self.current_state = env.reset()
        self.episode_reward = []
        self.current_ep_reward = 0
        pass

    def collect_data(self ):
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

        for _ in range(self.hyperparams['n_steps']):
            state = torch.tensor(self.current_state, dtype=torch.float32).unsqueeze(0)

            while torch.no_grad():
                distribution, value = self.actor_critic(state)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            next_state, reward, done, info = self.env.step(action.item())
            self.current_ep_reward+=reward

            self.states.append(state.squeeze(0))
            self.actions.append(action.squeeze(0))
            self.log_probs.append(log_prob.squeeze(0))
            self.rewards.append(reward)
            self.values.append(value.squeeze())
            self.dones.append(int(done))

            if done:
                self.episode_reward.append(self.current_ep_reward)
                self.current_ep_reward=0
                self.current_state = self.env.reset()
            else:
                self.current_state = next_state
        pass

    def compute_advantage(self, rewards, values, done):

        n=len(rewards)
        advantages = torch.zeros(n).to(self.device)
        GAE = 0.0

        values_tensor = torch.stack(values)

        for t in reversed(range(n)):
            delta = rewards[t] + self.hyperparams['gamma'] * values_tensor[t+1].item()* (1-done[t]) - values_tensor[t].item()
            gae = delta + self.hp['gamma'] * self.hp['gae_lambda'] * (1 - done[t]) * gae
            advantages[t] = gae

        returns = advantages + values_tensor[:-1]

        return advantages, returns
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        n = states.shape[0]

        for _ in range(self.hyperparams['n_epochs']):
            indices = torch.randperm(n)

            for start in range(0,n,self.hyperparams['batch_size']):
                end = start + self.hp['batch_size']
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_lp     = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns    = returns[batch_idx]

                distribution, new_values = self.actor_critic(batch_states)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean() 

                probability_ratio = torch.exp(new_log_probs - batch_old_lp)
                clipped_ratio = torch.clamp(probability_ratio, 1 - self.hp['clip_epsilon'], 1 + self.hp['clip_epsilon'])

                actor_loss = -torch.min(probability_ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                new_values = new_values.squeeze(-1)
                value_loss = nn.MSELoss()(new_values, batch_returns)
                loss = actor_loss + self.hp['value_loss_coef'] * value_loss - self.hp['entropy_coef'] * entropy

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        pass
    
    def learn(self, total_timestamps):
        learned_timestamps = 0
        while learned_timestamps<total_timestamps:
            self.collect_data()
            learned_timestamps +=self.hyperparams['n_steps']

            last_state = torch.tensor(self.current_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value = self.actor_critic(last_state)
            bootstrap_value = last_value.squeeze()

            values_with_bootstrap = self.values + [bootstrap_value]
            advantages, returns = self.compute_advantage(self.rewards, values_with_bootstrap, self.dones)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states     = torch.stack(self.states).to(self.device)
            actions    = torch.stack(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs).to(self.device)

            self.update_policy(states, actions, old_log_probs, advantages, returns)

            if len(self.episode_rewards) > 0:
                print(f"Timesteps: {learned_timestamps}/{total_timestamps}") #random log
        pass