import numpy as np
from tqdm import tqdm
import gymnasium as gym, gym_bandits


def ucb_strategy(env, n_episodes=5000, c=2):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int64)

    name = f"UCB {c}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        if e < len(Q): # We first select all actions one by one to avoid division by 0
            action = e
        else:
            U = np.sqrt(c * np.log(e) / N)
            action = np.argmax(Q + U)
        
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
    name, returns, Qe, actions = ucb_strategy(env)
    print(f"{name}:\nReturns: {returns[-1]} \nQ-values: {Qe[-1]} \nActions: {actions[-1]}")