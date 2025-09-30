import numpy as np
import gymnasium as gym
from itertools import count
from tqdm import tqdm
from monte_carlo_prediction import decay_schedule


def temporal_differnce(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3, n_episodes=500):
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes), leave=False):
        (state, _), terminated, truncated = env.reset(), False, False
        
        while not (terminated or truncated):
            action = pi(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not (terminated or truncated))
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error
            state = next_state
            
        V_track[e] = V
        
    return V.copy(), V_track