import numpy as np
import gymnasium as gym
from itertools import count
from tqdm import tqdm
from monte_carlo_prediction import decay_schedule


def n_step_temporal_differnce(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3, n_step=3, n_episodes=500):
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    discounts = np.logspace(0, n_step+1, num=n_step+1, base=gamma, endpoint=False)
    
    for e in tqdm(range(n_episodes), leave=False):
        (state, _), terminated, truncated, path = env.reset(), False, False, []
        
        while not (terminated or truncated) or path is not None:
            path = path[1:]
            
            while not (terminated or truncated) and len(path) < n_step:
                action = pi(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                experience = (state, reward, next_state, terminated, truncated)
                path.append(experience)
                state = next_state
                
                if terminated or truncated:
                    break
                
            n = len(path)
            estimated_state = path[0][0]
            
            rewards = np.array(path)[:, 1]
            partial_return = discounts[:n] * rewards
            bootstrapping_value = discounts[-1] * V[next_state] * (not (terminated or truncated))
            
            ntd_target = np.sum(np.append(partial_return, bootstrapping_value))
            ntd_error = ntd_target - V[estimated_state]
            V[estimated_state] = V[estimated_state] + alphas[e] * ntd_error
            
            if len(path) == 1 and (path[0][3] or path[0][4]):
                path = None
            
        V_track[e] = V

    return V.copy(), V_track