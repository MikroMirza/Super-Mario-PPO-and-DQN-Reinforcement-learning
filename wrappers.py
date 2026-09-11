from doom_gym_wrappers import DoomMaxAndSkipEnv, DoomNormalizeObservation, DoomNormalizeReward, DoomObservation, DoomRandomStart
import numpy as np
import gym
from gym import spaces
from collections import deque
import cv2

class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
            
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GrayScaleObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # H x W
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs


class ResizeObservation(gym.ObservationWrapper):

    def __init__(self, env, shape=(60, 80)):
        super().__init__(env)

        self.shape = tuple(shape)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        width = self.shape[1]
        height = self.shape[0]

        return cv2.resize(
            obs,
            (width, height),
            interpolation=cv2.INTER_AREA
        )


class FrameStack(gym.Wrapper):

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.zeros(
            (num_stack, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        high = np.full(
            (num_stack, *env.observation_space.shape),
            255,
            dtype=env.observation_space.dtype,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return np.array(self.frames, dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info


class ClipReward(gym.RewardWrapper):

    def reward(self, reward):
        return np.sign(reward)


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=15.0):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, reward):
        return reward / self.scale
def make_env(env_id="assets/defend_the_center.cfg", skip=4, shape=84, stack=4, clip_rewards=True,
             max_episode_steps=500, window_visible=False, reward_mode="kills",
             kill_reward=1.0, distance_scale=0.03, health_scale=0.05,
             aim_reward=0.5, aim_penalty=0.1, longevity_reward=0.01,
             distance_discount=1.0, gate_advance_on_enemy=False, gated_discount=0.05,
             allowed_button_indices=None, extra_combos=None):
    try:
        from Environments.vizdoom_env import VizDoomEnv
    except ImportError:
        raise ImportError(
            "Install dependencies:\n"
            "  pip install vizdoom opencv-python"
        )

    env = VizDoomEnv(
        config_path=env_id, window_visible=window_visible, reward_mode=reward_mode,
        kill_reward=kill_reward, distance_scale=distance_scale, health_scale=health_scale,
        aim_reward=aim_reward, aim_penalty=aim_penalty, longevity_reward=longevity_reward,
        distance_discount=distance_discount, gate_advance_on_enemy=gate_advance_on_enemy,
        gated_discount=gated_discount,
        extra_combos=extra_combos, allowed_button_indices=allowed_button_indices
    )
    env = SkipFrame(env, skip=skip)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=stack)
    env = NormalizeObservation(env)
    env = DoomRandomStart(env, max_turn_steps=5)
    if clip_rewards:
        env = ClipReward(env)
    return env