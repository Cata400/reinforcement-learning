import numpy as np
from tqdm import tqdm
import gymnasium as gym, gym_bandits


def exp_dec_epsilon_greedy(env, n_episodes=5000, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.1):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int64)
    
    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')
    
    name = f"Exponential Epsilon Greedy {init_epsilon} {min_epsilon} {decay_ratio}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        p = np.random.rand()
        if p < epsilons[e]:
            action = np.random.randint(len(Q))
        else:
            action = np.argmax(Q)
        
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
    name, returns, Qe, actions = exp_dec_epsilon_greedy(env)
    print(f"{name}:\nReturns: {returns[-1]} \nQ-values: {Qe[-1]} \nActions: {actions[-1]}")