params = {
    'learning_rate':  2.5e-4,
    'gamma':          0.99,
    'gae_lambda':     0.95,
    'clip_epsilon':   0.2,
    'n_epochs':       4,
    'batch_size':     256,
    'n_steps':        512,
    'entropy_coef':   0.01,
    'value_loss_coef': 0.5,
}

hyperparameters = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.3,
    'n_epochs':        4,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.05,
    'value_loss_coef': 0.5,
}

# Za defend_the_center (3 akcije) - manji entropy_coef jer isti koeficijent
# daje jaci relativni bonus kod manjeg akcionog prostora (ln(3) vs ln(7)).
hyperparameters_defend = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.1,
    'n_epochs':        3,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.01,
    'value_loss_coef': 0.25,
}

# Za deadly_corridor (7 akcija).
hyperparameters_corridor = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.1,
    'n_epochs':        3,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.025,
    'value_loss_coef': 0.25,
}


dqn_params = {
    'lr': 2.5e-5,
    'gamma': 0.99,
    'buffer_capacity': 200_000,
    'batch_size': 256,
    'eps_start': 1.0,
    'eps_end': 0.1,
    'eps_decay_steps': 500_000,
    'target_update_freq': 10_000,
    'train_freq': 4,
    'learning_starts': 10_000,
}

# ---------------------------------------------------------------------
# Scenario configs. Svaki dict ide u make_env(**cfg minus par kljuceva).
# reward_mode "kills" je bezopasan default za scenarije bez neprijatelja
# (KILLCOUNT ostaje 0, aim/health shaping se prirodno iskljucuje ako
# scenario nema ATTACK dugme).
# ---------------------------------------------------------------------

env_params = {
    'env_id': "assets/defend_the_center.cfg",
    'frame_skip': 4,
    'frame_size': 84,
    'frame_stack': 4,
    'clip_rewards': False,
    'max_ep_steps': 2_100,
    'window_visible': False,
    'reward_mode': 'kills',
    'kill_reward': 1.3,
    'distance_scale': 0.03,
    'health_scale': 0.1,
    'longevity_reward': 0.01,
    'aim_reward': 0.6,
    'aim_penalty': 0.2,
}

env_params_corridor = {
    'env_id': "assets/deadly_corridor.cfg",
    'frame_skip': 4,
    'frame_size': 84,
    'frame_stack': 4,
    'clip_rewards': False,
    'max_ep_steps': 2_100,
    'window_visible': True,
    'reward_mode': 'distance',
    'kill_reward': 3.0,
    'distance_scale': 0.015,
    'health_scale': 0.15,
    'longevity_reward': 0.0,
    'aim_reward': 1.2,
    'aim_penalty': 0.15,
}

env_params_basic = {
    'env_id': "assets/basic.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 300, 'window_visible': False,
    'reward_mode': 'kills', 'kill_reward': 1.0, 'distance_scale': 0.0,
    'health_scale': 0.0, 'longevity_reward': 0.0, 'aim_reward': 0.3, 'aim_penalty': 0.05,
}

env_params_health = {
    'env_id': "assets/health_gathering.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 2_100, 'window_visible': False,
    'reward_mode': 'kills', 'kill_reward': 0.0, 'distance_scale': 0.0,
    'health_scale': 0.02, 'longevity_reward': 0.01, 'aim_reward': 0.0, 'aim_penalty': 0.0,
}

env_params_myhome = {
    'env_id': "assets/my_way_home.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 2_100, 'window_visible': False,
    'reward_mode': 'kills', 'kill_reward': 0.0, 'distance_scale': 0.0,
    'health_scale': 0.0, 'longevity_reward': 0.0, 'aim_reward': 0.0, 'aim_penalty': 0.0,
}

SCENARIOS = {
    "defend":   (env_params,          hyperparameters_defend),
    "corridor": (env_params_corridor, hyperparameters_corridor),
    "basic":    (env_params_basic,    hyperparameters_defend),
    "health":   (env_params_health,   hyperparameters_corridor),
    "myhome":   (env_params_myhome,   hyperparameters_corridor),
}

training_params = {
    'max_steps': 100_000_000,
    'log_freq': 1_000,
    'save_freq': 25_000,
    'csv_flush_freq': 10_000,
    'checkpoint_dir': "checkpoints",
    'log_dir': "logs",
}