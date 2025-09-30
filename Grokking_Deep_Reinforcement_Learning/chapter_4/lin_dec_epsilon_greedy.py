import numpy as np
from tqdm import tqdm


def lin_dec_epsilon_greedy(env, n_episodes=5000, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.05):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)
    
    decay_episodes = n_episodes * decay_ratio
    
    name = f"Linear Epsilon Greedy {init_epsilon} {min_epsilon} {decay_ratio}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        epsilon = (1 - e / decay_episodes) * (init_epsilon - min_epsilon) + min_epsilon
        epsilon = np.clip(epsilon, min_epsilon, init_epsilon)
        
        p = np.random.rand()
        if p < epsilon:
            action = np.random.randint(len(Q))
        else:
            action = np.argmax(Q)
        
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
        
    return name, returns, Qe, actions