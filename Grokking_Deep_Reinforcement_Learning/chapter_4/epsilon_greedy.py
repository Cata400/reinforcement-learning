import numpy as np
from tqdm import tqdm


def epsilon_greedy(env, n_episodes=5000, epsilon=0.01):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)
    
    name = f"Epsilon Greedy {epsilon}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
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