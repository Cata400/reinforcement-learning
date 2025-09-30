import numpy as np
import gymnasium as gym
from itertools import count
from tqdm import tqdm


def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values

def sarsa_lambda(
    env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
    init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, 
    lambda_=0.5, replacing_traces=True, n_episodes=3000):
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    E = np.zeros((nS, nA))
    
    pi_track = []
    Q = np.zeros((nS, nA))
    Q_track = np.zeros((n_episodes, nS, nA))
    
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    for e in tqdm(range(n_episodes), leave=False):
        E.fill(0)
        state, _ = env.reset()
        terminated, truncated = False, False
        action = select_action(state, Q, epsilons[e])
        
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            
            td_target = reward + gamma * Q[next_state][next_action] * (not (terminated or truncated))
            td_error = td_target - Q[state][action]
            E[state][action] = E[state][action] + 1
            if replacing_traces: E.clip(0, 1, out=E)
            Q = Q + alphas[e] * td_error * E
            E = gamma * lambda_ * E
            
            state, action = next_state, next_action
            
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
        
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        
    return Q, V, pi, Q_track, pi_track

def use_policy(env, pi, n_episodes):
    avg_reward = 0
    for e in tqdm(range(n_episodes), leave=False):
        episode_reward = 0
        state, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action = pi(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
                        
            state = next_state
            
        print(f"Episode reward: {episode_reward}")
        avg_reward += episode_reward
        
    avg_reward = avg_reward / n_episodes
    print(f"Average episode reward: {avg_reward}")


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    Q, V, pi, _, _ = sarsa_lambda(env)

    env = gym.make('FrozenLake-v1', render_mode="human")
    avg_reward = use_policy(env, pi, 100)