import numpy as np
from tqdm import tqdm
import gymnasium as gym, gym_bandits


def thompson_sampling_strategy(env, n_episodes=5000, alpha=1, beta=0):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int64)
        
    name = f"Thompson Sampling {alpha} {beta}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        samples = np.random.normal(loc=Q, scale=alpha/(np.sqrt(N) + beta))
        action = np.argmax(samples)

        _, reward, _, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
        
    return name, returns, Qe, actions


if __name__ == '__main__': 
    env = gym.make("BanditTwoArmedUniform-v0")
    env.reset()
    name, returns, Qe, actions = thompson_sampling_strategy(env)
    print(f"{name}:\nReturns: {returns[-1]} \nQ-values: {Qe[-1]} \nActions: {actions[-1]}")