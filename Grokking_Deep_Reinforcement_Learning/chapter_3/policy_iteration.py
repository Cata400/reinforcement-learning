from greedy_policy_improvement import greedy_policy_improvement
from iterative_policy_evaluation import policy_evaluation
import numpy as np
import gymnasium as gym

def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
    
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = greedy_policy_improvement(V, P, gamma)
        
        if old_pi == {s:pi(s) for s in range(len(P))}:
            break
        
    return V, pi


env = gym.make('FrozenLake-v1')
base_env = env.unwrapped
P = base_env.P
V, pi = policy_iteration(P)
for s, v in enumerate(V):
    print(s, v, pi(s))