import os

import gym
import numpy as np
import vizdoom as vzd
from gym import spaces


class VizDoomEnv(gym.Env):
    """Tanak gym.Env omotac oko sirovog ViZDoom API-ja (stari gym 4-tuple API)."""

    def __init__(self, config_path, frame_repeat=1, window_visible=False,
                 reward_mode="kills", kill_reward=1.0, distance_scale=0.03,
                 health_scale=0.05, aim_reward=0.5, aim_penalty=0.1,
                 longevity_reward=0.01):
        super().__init__()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ViZDoom config nije pronadjen: {config_path}")

        self.frame_repeat = frame_repeat
        self.reward_mode = reward_mode
        self.kill_reward = kill_reward
        self.distance_scale = distance_scale
        self.health_scale = health_scale
        self.aim_reward = aim_reward
        self.aim_penalty = aim_penalty
        self.longevity_reward = longevity_reward

        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(window_visible)
        self.game.set_labels_buffer_enabled(True)
        self.game.init()

        n_buttons = self.game.get_available_buttons_size()
        self._actions = [list(a) for a in np.eye(n_buttons, dtype=np.int32).tolist()]
        self.action_space = spaces.Discrete(n_buttons)

        buttons = self.game.get_available_buttons()
        self._attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None

        h = self.game.get_screen_height()
        w = self.game.get_screen_width()
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

        self._prev_kills = 0
        self._prev_x = 0.0
        self._prev_health = 100.0
        self._prev_hits = 0.0

    def _get_state_and_obs(self):
        state = self.game.get_state()
        if state is None:
            return None, np.zeros(self.observation_space.shape, dtype=np.uint8)
        return state, state.screen_buffer

    def reset(self, **kwargs):
        self.game.new_episode()
        self._prev_kills = 0
        self._prev_x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        self._prev_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self._prev_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        _, obs = self._get_state_and_obs()
        return obs

    def step(self, action):
        base_reward = self.game.make_action(self._actions[action], self.frame_repeat)

        done = self.game.is_episode_finished()
        state, obs = self._get_state_and_obs()

        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        died = bool(done and health <= 0)
        attacked = bool(self._attack_idx is not None and self._actions[action][self._attack_idx] == 1)

        reward = base_reward

        health_delta = health - self._prev_health
        reward += self.health_scale * health_delta
        self._prev_health = health

        kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        kill_delta = kills - self._prev_kills
        self._prev_kills = kills
        reward += self.kill_reward * kill_delta

        if not died:
            reward += self.longevity_reward

        aimed = False
        if attacked and not done:
            hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            hit_landed = hits > self._prev_hits
            self._prev_hits = hits
            aimed = hit_landed
            reward += self.aim_reward if hit_landed else -self.aim_penalty
        else:
            self._prev_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)

        info = {"died": died, "kills": int(self._prev_kills), "attacked": attacked, "aimed": aimed}

        if self.reward_mode == "distance":
            x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            reward += self.distance_scale * (x - self._prev_x)
            self._prev_x = x
            info["x_pos"] = float(x)

        return obs, reward, done, info

    def render(self):
        pass

    def close(self):
        self.game.close()