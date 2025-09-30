import numpy as np
from tqdm import tqdm
import gymnasium as gym, gym_bandits


def softmax_strategy(env, n_episodes=5000, init_temp=1000.0, min_temp=0.01, decay_ratio=0.04):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int64)
    
    decay_episodes = n_episodes * decay_ratio
    
    name = f"Softmax {init_temp} {min_temp} {decay_ratio}"
    for e in tqdm(range(n_episodes), desc=f"Episodes for: {name}", leave=False):
        temp = (1 - e / decay_episodes) * (init_temp - min_temp) + min_temp
        temp = np.clip(temp, min_temp, init_temp)
        
        scaled_Q = Q / temp
        norm_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(norm_Q)
        probs = exp_Q / np.sum(exp_Q)
        
        assert np.isclose(probs.sum(), 1.0)
        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        
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
    name, returns, Qe, actions = softmax_strategy(env)
    print(f"{name}:\nReturns: {returns[-1]} \nQ-values: {Qe[-1]} \nActions: {actions[-1]}")