from abc import ABC, abstractmethod
import numpy as np
import torch

class Agent(ABC):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr,
                 seed=None):
        self.obs_space = obs_space
        self.action_space = action_space
        self.discount = discount
        self.lr = lr
        self.rng = np.random.default_rng(seed=seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def policy(self, state, available, eval=False):
        """
        This function returns the action given the observation.
        """

    @abstractmethod
    def train(self, buffer):
        """
        Train the agent
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the agent.
        """

    @abstractmethod
    def save_state(self) -> None:
        """
        Use this method to save the current state of your agent to the agent_directory.
        """

    @abstractmethod
    def load_state(self) -> None:
        """ 
        Use this method to load the agent state from the self.agent_directory
        """

