import numpy as np
import gymnasium as gym
from itertools import count
from tqdm import tqdm
from sarsa_lambda import decay_schedule, use_policy

def dyna_q(
    env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
    init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, 
    n_planning=3, n_episodes=3000):
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    pi_track = []
    Q = np.zeros((nS, nA))
    Q_track = np.zeros((n_episodes, nS, nA))
    
    T_count = np.zeros((nS, nA, nS))
    R_model = np.zeros((nS, nA, nS))
    
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    for e in tqdm(range(n_episodes), leave=False):
        state, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action = select_action(state, Q, epsilons[e])
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            T_count[state, action, next_state] += 1
            r_diff = reward - R_model[state, action, next_state]
            R_model[state, action, next_state] += (r_diff / T_count[state, action, next_state])
            
            td_target = reward + gamma * Q[next_state].max() * (not (terminated or truncated))
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alphas[e] * td_error
            
            backup_next_state = next_state
            for _ in range(n_planning):
                if Q.sum() == 0:
                    break
                
                visited_states = np.where(np.sum(T_count, axis=(1, 2)) > 0)[0]
                state = np.random.choice(visited_states)
                
                actions_taken = np.where(np.sum(T_count[state], axis=1) > 0)[0]
                action = np.random.choice(actions_taken)
                
                probs = T_count[state, action] / T_count[state, action].sum()
                next_state = np.random.choice(np.arange(nS), size=1, p=probs)[0]
                
                reward = R_model[state, action, next_state]
                td_target = reward + gamma * Q[next_state].max()
                td_error = td_target - Q[state, action]
                Q[state, action] = Q[state, action] + alphas[e] * td_error
                
            state = backup_next_state
            
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
        
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
    return Q, V, pi, Q_track, pi_track


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    Q, V, pi, _, _ = dyna_q(env)

    env = gym.make('FrozenLake-v1', render_mode="human")
    avg_reward = use_policy(env, pi, 100)