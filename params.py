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