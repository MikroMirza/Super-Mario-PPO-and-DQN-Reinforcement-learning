from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, state_shape, n_actions: int, device: str | None = None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.total_steps = 0

        import torch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        pass

    #Vrati metrike koje zelis da ispises
    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict | None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def extra_metrics(self) -> dict:
        return {}