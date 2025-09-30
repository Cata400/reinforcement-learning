import numpy as np
import gymnasium as gym
from itertools import count
from tqdm import tqdm
from monte_carlo_control import decay_schedule, use_policy

def q_learning(
    env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
    init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, 
    n_episodes=3000):
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    pi_track = []
    Q = np.zeros((nS, nA))
    Q_track = np.zeros((n_episodes, nS, nA))
    
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    for e in tqdm(range(n_episodes), leave=False):
        state, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action = select_action(state, Q, epsilons[e])
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            td_target = reward + gamma * Q[next_state].max() * (not (terminated or truncated))
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            
            state = next_state
            
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
        
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
    return Q, V, pi, Q_track, pi_track


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    Q, V, pi, _, _ = q_learning(env)

    env = gym.make('FrozenLake-v1', render_mode="human")
    avg_reward = use_policy(env, pi, 100)