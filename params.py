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
    'gamma':           0.99,     # discount
    'gae_lambda':      0.95,     # high variance Generalized Advantage Estimation
    'clip_epsilon':    0.1,      # PPO clipping range ~20%
    'n_epochs':        4,        # num of updates per batch
    'batch_size':      256,
    'n_steps':         1024,     # n steps before update
    'entropy_coef':    0.05,     # entropy = exploration
    'value_loss_coef': 0.5,      # used for critic loss 
}

params2ndgo = {
    'learning_rate':  2.5e-5,
    'gamma':          0.99,
    'gae_lambda':     0.95,
    'clip_epsilon':   0.15,
    'n_epochs':       3,
    'batch_size':     256,
    'n_steps':        2048,
    'entropy_coef':   0.02,
    'value_loss_coef': 0.5,
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
    'train_freq': 4,        #Na koliko koraka se azurira
    'learning_starts': 10_000,   #Koliko koraka treba baferovati pre nego sto trening pocne
}

env_params = {
    'env_id': "SuperMarioBros-1-1-v0",
    'frame_skip': 4,
    'frame_size': 84,
    'frame_stack': 4,
    'clip_rewards': False,
    'max_ep_steps': 10_000
}

training_params = {
    'max_steps': 100_000_000,
    'log_freq': 1_000,    #Frekvencija ispisivanja statistike
    'save_freq': 50_000,   #Frekvencija cuvanja checkpointa
    'csv_flush_freq': 10_000,   #Koliko cesto se upisuje CSV na disk
    'checkpoint_dir': "checkpoints",
    'log_dir': "logs",
}