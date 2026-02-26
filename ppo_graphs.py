import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/ppo_episodes.csv")

# Smooth reward
df['reward_smooth'] = df['ep_reward'].rolling(window=500, min_periods=1).mean()

df_plot = df.iloc[::50, :]

early_success = df[(df['flag_get']==1)]

plt.figure(figsize=(12,6))

plt.plot(df_plot['episode'], df_plot['reward_smooth'], color='blue', linewidth=2, label='Reward (smoothed)')

plt.scatter(early_success['episode'], early_success['reward_smooth'],
            color='red', s=100, marker='*', label='Goal Reached (early)', zorder=5)

plt.xlabel("Episode", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.title("PPO Training: Reward and Goal Achievements", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()