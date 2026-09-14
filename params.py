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

env_params = {
    'env_id': "assets/defend_the_center.cfg",
    'frame_skip': 4,
    'frame_size': 84,
    'frame_stack': 4,
    'clip_rewards': False,
    'max_ep_steps': 2_100,
    'window_visible': True,
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
    'kill_reward': 10.0,
    'distance_scale': 0.02,
}

env_params_basic = {
    'env_id': "assets/basic.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 300, 'window_visible': True,
    'reward_mode': 'kills', 'kill_reward': 1.0, 'distance_scale': 0.0,
    'health_scale': 0.0, 'longevity_reward': 0.0, 'aim_reward': 0.3, 'aim_penalty': 0.05,
    # 'extra_combos': [(0, 2), (1, 2)],   # MOVE_LEFT+ATTACK, MOVE_RIGHT+ATTACK
}

env_params_health = {
    'env_id': "assets/health_gathering.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 2_100, 'window_visible': False,
    'reward_mode': 'kills', 'kill_reward': 0.0, 'distance_scale': 0.0,
    'health_scale': 0.05, 'longevity_reward': 0.00, 'aim_reward': 0.0, 'aim_penalty': 0.0, 'allowed_button_indices': [0, 1, 2],
}
hyperparameters_health = {
    'learning_rate':   2.5e-4,
    'gamma':           0.995,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.2,

    'n_epochs':        4,
    'batch_size':      256,
    'n_steps':         4096,

    'entropy_coef':    0.03,
    'value_loss_coef': 0.5,
}

hyperparameters_basic = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.15,
    'n_epochs':        3,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.06,
    'value_loss_coef': 0.25,
}

env_params_myhome = {
    'env_id': "assets/my_way_home.cfg",
    'frame_skip': 4, 'frame_size': 84, 'frame_stack': 4, 'clip_rewards': False,
    'max_ep_steps': 2_100, 'window_visible': True,
    'reward_mode': 'explore',
    'kill_reward': 0.0, 'distance_scale': 0.02,
    'health_scale': 0.0, 'longevity_reward': 0.0, 'aim_reward': 0.0, 'aim_penalty': 0.0,
    'stuck_penalty': 0.03,
}

hyperparameters_myhome = {
    'learning_rate':   1e-4,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_epsilon':    0.15,
    'n_epochs':        3,
    'batch_size':      256,
    'n_steps':         1024,
    'entropy_coef':    0.06,
    'value_loss_coef': 0.25,
}

training_params = {
    'max_steps': 100_000_000,
    'log_freq': 1_000,
    'save_freq': 25_000,
    'csv_flush_freq': 10_000,
    'checkpoint_dir': "checkpoints",
    'log_dir': "logs",
}









env_params = {
    'env_id': "assets/defend_the_center.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),
    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 2_100,
    'window_visible': True,

    'reward_mode': 'kills',
    'kill_reward': 1.3,
    'distance_scale': 0.03,
    'health_scale': 0.1,
    'longevity_reward': 0.01,
    'aim_reward': 0.6,
    'aim_penalty': 0.2,
}


env_params = {
    'env_id': "assets/defend_the_center.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),
    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 2_100,
    'window_visible': True,

    'reward_mode': 'kills',
    'kill_reward': 1.3,
    'distance_scale': 0.03,
    'health_scale': 0.1,
    'longevity_reward': 0.01,
    'aim_reward': 0.6,
    'aim_penalty': 0.2,
}

env_params_basic = {
    'env_id': "assets/basic.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),
    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 300,
    'window_visible': True,

    'reward_mode': 'kills',
    'kill_reward': 1.0,
    'distance_scale': 0.0,

    'health_scale': 0.0,
    'longevity_reward': 0.0,

    'aim_reward': 0.3,
    'aim_penalty': 0.05,
}

env_params_health = {
    'env_id': "assets/health_gathering.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),
    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 2_100,
    'window_visible': False,

    'reward_mode': 'kills',

    'kill_reward': 0.0,
    'distance_scale': 0.0,

    'health_scale': 0.05,
    'exploration_scale': 0.25,

    'longevity_reward': 0.05,

    'aim_reward': 0.0,
    'aim_penalty': 0.0,

    'allowed_button_indices': [0, 1, 2],
}

env_params_myhome = {
    'env_id': "assets/my_way_home.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),

    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 2_100,
    'window_visible': True,

    'reward_mode': 'explore',

    'kill_reward': 0.0,
    'distance_scale': 0.02,

    'health_scale': 0.0,
    'longevity_reward': 0.0,

    'aim_reward': 0.0,
    'aim_penalty': 0.0,

    'stuck_penalty': 0.1,
}

hyperparameters = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.2,

    'n_epochs': 4,
    'batch_size': 256,
    'n_steps': 1024,

    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
}

hyperparameters_defend = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.1,

    'n_epochs': 3,
    'batch_size': 256,
    'n_steps': 1024,

    'entropy_coef': 0.05,
    'value_loss_coef': 0.25,
}
hyperparameters_corridor = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.1,

    'n_epochs': 3,
    'batch_size': 256,
    'n_steps': 1024,

    'entropy_coef': 0.02, #0.025
    'value_loss_coef': 0.2,
}

hyperparameters_basic = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.15,

    'n_epochs': 3,
    'batch_size': 256,
    'n_steps': 1024,

    'entropy_coef': 0.06,
    'value_loss_coef': 0.25,
}

hyperparameters_health = {
    'learning_rate': 5e-4,
    'gamma': 0.995,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.2,

    'n_epochs': 4,
    'batch_size': 256,
    'n_steps': 4096,

    'entropy_coef': 0.3,
    'value_loss_coef': 0.5,
}

hyperparameters_myhome = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    'clip_epsilon': 0.15,

    'n_epochs': 3,
    'batch_size': 512,
    'n_steps': 2048,

    'entropy_coef': 0.04,
    'value_loss_coef': 0.25,
}

env_params_corridor = {
    'env_id': "assets/deadly_corridor.cfg",

    'frame_skip': 4,
    'frame_size': (84, 84),

    'frame_stack': 4,

    'clip_rewards': False,

    'max_ep_steps': 2_100,
    'window_visible': False,

    'reward_mode': 'distance',
    'distance_scale': 0.00005,
    'distance_discount': 0.002,

    'gate_advance_on_enemy': True,
    'gated_discount': 0.05,

    'kill_reward': 25,
    'aim_reward': 0.015,
    'aim_penalty': 0.001,

    'health_scale': 0.0025,
    'longevity_reward': 0.0,

    'allowed_button_indices': [0, 1, 2, 3, 4],

    'extra_combos': [
        (3, 1),
        (4, 1),
        (1, 0),
        (3, 0),
        (4, 0),
    ],
}

SCENARIOS = {
    "defend":   (env_params,          hyperparameters_defend),
    "corridor": (env_params_corridor , hyperparameters_corridor),
    "basic":    (env_params_basic,    hyperparameters_basic),
    "health":   (env_params_health,   hyperparameters_health),
    "myhome":   (env_params_myhome,   hyperparameters_myhome),
}