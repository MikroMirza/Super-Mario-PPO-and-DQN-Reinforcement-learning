import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_ppo(df):
    df['reward_smooth'] = df['ep_reward'].rolling(window=500, min_periods=1).mean()
    df_plot = df.iloc[::50, :]
    early_success = df[df['flag_get'] == 1]

    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['episode'], df_plot['reward_smooth'], color='blue', linewidth=2, label='Reward (smoothed)')
    plt.scatter(early_success['episode'], early_success['reward_smooth'],
                color='red', s=100, marker='*', label='Goal Reached (early)', zorder=5)
    plt.title("PPO Training: Reward and Goal Achievements", fontsize=16)


def plot_ddqn(df):
    df['reward_smooth'] = df['ep_reward'].rolling(window=500, min_periods=1).mean()
    df_plot = df.iloc[::50, :]

    early_cutoff = 5500
    late_cutoff = 9000
    early_success = df[(df['flag_get'] == 1) & (df['episode'] <= early_cutoff)]
    late_success = df[(df['flag_get'] == 1) & (df['episode'] > early_cutoff) & (df['episode'] < late_cutoff)]
    very_late_success = df[(df['flag_get'] == 1) & (df['episode'] > late_cutoff)]

    late_success = late_success.iloc[::4]
    very_late_success = very_late_success.iloc[::60]

    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['episode'], df_plot['reward_smooth'], color='blue', linewidth=2, label='Reward (smoothed)')
    plt.scatter(early_success['episode'], early_success['reward_smooth'],
                color='red', s=100, marker='*', label='Goal Reached (early)', zorder=5)
    plt.scatter(late_success['episode'], late_success['reward_smooth'],
                color='red', s=100, marker='*', label='Goal Reached (sampled later)', zorder=5, alpha=0.7)
    plt.scatter(very_late_success['episode'], very_late_success['reward_smooth'],
                color='red', s=100, marker='*', label='Goal Reached (sampled later)', zorder=5, alpha=0.7)
    plt.title("DDQN Training: Reward and Goal Achievements", fontsize=16)


def main():
    parser = argparse.ArgumentParser(description="Visualize PPO or DDQN training logs.")
    parser.add_argument("algorithm", choices=["ppo", "ddqn"], help="Algorithm to visualize")
    parser.add_argument("--log", required=True, help="Path to the CSV log file")
    args = parser.parse_args()

    df = pd.read_csv(args.log)

    if args.algorithm == "ppo":
        plot_ppo(df)
    else:
        plot_ddqn(df)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()