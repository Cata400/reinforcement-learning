from gym import Env
from gym.spaces import Discrete, Box
import random


class ShowerEnv(Env):
    """
    Environment for a shower minigame. The goal is to build an agent that gives us the best shower temperature.
    As other people in our apartment building may use the shower as well, the temperature will randomly change in time.
    The goal is to get a temperature between 37 and 39 degrees Celsius for as much time as possible.

    Parameters
    ----------
    
    """
    def __init__(self):
        self.action_space = Discrete(3) # tap down, tap unchanged, tap up
        self.observation_space = Box(low=0, high=100)
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60 # episode length
    
    def step(self, action):
        self.state += action - 1
        self.shower_length -= 1
        
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
            
        # terminated = False
        # truncated = self.shower_length <= 0
        done = self.shower_length <= 0
            
        info = {}
        
        return self.state, reward, done, info
    
    def render(self):
        # Implements visuals
        pass
    
    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        
        return self.state