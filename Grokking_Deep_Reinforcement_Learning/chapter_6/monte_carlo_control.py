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

def generate_trajectory(select_action, Q, epsilon, env, max_steps=20):
    terminated, truncated, trajectory = False, False, []
    
    while not (terminated or truncated):
        state, _ = env.reset()
        for t in count():
            action = select_action(state, Q, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            experience = (state, action, reward, next_state, terminated, truncated)
            trajectory.append(experience)
            
            if terminated or truncated:
                break
            
            if t >= max_steps - 1:
                trajectory = []
                break
            
            state = next_state
            
    return np.array(trajectory, object)

def mc_control(
    env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
    init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, 
    n_episodes=3000, max_steps=200, first_visit=True):
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    pi_track = []
    Q = np.zeros((nS, nA))
    Q_track = np.zeros((n_episodes, nS, nA))
    
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)
        
        for t, (state, action, reward, _, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])
            
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
    Q, V, pi, _, _ = mc_control(env)

    env = gym.make('FrozenLake-v1', render_mode="human")
    avg_reward = use_policy(env, pi, 100)
