import numpy as np
from tqdm import tqdm


def optimistic_initialization_strategy(env, n_episodes=5000, optimistic_estimate=1.0, initial_count=100):
    Q = np.full((env.action_space.n), optimistic_estimate, dtype=np.float64)
    N = np.full((env.action_space.n), initial_count, dtype=np.float64)
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)
    
    name = f"Optimistic {optimistic_estimate} {initial_count}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        action = np.argmax(Q)
        
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
        
    return name, returns, Qe, actions