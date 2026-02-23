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

    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs


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

def make_env(env_id="SuperMarioBros-v0", skip=4, shape=84, stack=4, clip_rewards=True, max_episode_steps=500):
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    except ImportError:
        raise ImportError(
            "Install dependencies:\n"
            "  pip install gym-super-mario-bros nes-py opencv-python"
        )

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=skip)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=stack)
    env = NormalizeObservation(env)
    if clip_rewards:
        env = ClipReward(env)
    else:
        env = ScaleRewardWrapper(env)
    return env
