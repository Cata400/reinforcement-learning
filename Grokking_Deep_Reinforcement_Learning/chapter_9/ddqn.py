import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
import random
from tqdm import tqdm
from itertools import count

class FCQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
        
    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0)
            
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        new_states = torch.from_numpy(new_states).float()
        rewards = torch.from_numpy(rewards).float()
        is_terminals = torch.from_numpy(is_terminals).float()
        return states, actions, rewards, new_states, is_terminals
    
class GreedyStrategy:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def select_action(self, model, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)
    
class EpsilonGreedyExpStrategy:
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = model(state).cpu().detach()
            q_values = q_values.data.numpy().squeeze()
            
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        
        self._epsilon_update()
        return action
    
class ReplayBuffer:
    def __init__(self, m_size=50_000, batch_size=64):
        self.ss_mem = np.empty(shape=(m_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(m_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(m_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(m_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(m_size), dtype=np.ndarray)
        
        self.m_size = m_size
        self.batch_size = batch_size
        self._idx, self.size = 0, 0
        
    def store(self, sample):
        s, a, r, p, d = sample
        
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.m_size
        
        self.size += 1
        self.size = min(self.size, self.m_size)
        
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                    np.vstack(self.as_mem[idxs]), \
                    np.vstack(self.rs_mem[idxs]), \
                    np.vstack(self.ps_mem[idxs]), \
                    np.vstack(self.ds_mem[idxs])
                    
        return experiences
    
    def __len__(self):
        return self.size
        
class DDQN:
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn, evaluation_strategy_fn):
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        is_terminals = is_terminals.to(self.device)
        
        argmax_a_q_sp = self.online_model(next_states).detach().max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[torch.arange(batch_size), argmax_a_q_sp].unsqueeze(1) * (1 - is_terminals)
        target_q_sa = rewards + self.gamma * max_a_q_sp
        
        q_sa = self.online_model(states).gather(1, actions)
        td_errors = q_sa - target_q_sa
        value_loss = torch.mean(td_errors**2)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=self.max_gradient_norm) # Huber loss through gradient clipping
        self.value_optimizer.step()
        
    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated and not truncated
        experience = (state, action, reward, next_state, float(is_failure))
        
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return next_state, terminated
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, update_steps, replay_buffer, min_buffer_size, max_gradient_norm):
        env = gym.make(env_name, render_mode=env_render_mode)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        
        self.online_model = self.value_model_fn(nS, nA)
        self.target_model = self.value_model_fn(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(self.online_model, lr=self.value_optimizer_lr)
        self.training_strategy = self.training_strategy_fn()
        self.gamma = gamma
        self.online_model.to(self.device)
        self.target_model.to(self.device)
        self.update_steps = update_steps
        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.max_gradient_norm = max_gradient_norm
        
        for episode in tqdm(range(max_episodes)):
            state, _ = env.reset()
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                            
                if len(self.replay_buffer) > self.min_buffer_size:
                    batch = self.replay_buffer.sample()  
                    batch = self.online_model.load(batch)    
                    self.optimize_model(batch)
                    
                if np.sum(self.episode_timestep) % self.update_steps == 0:
                    self.update_network()
                    
                if is_terminal:
                    gc.collect()
                    break
                
            # stats
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Timestep {np.sum(self.episode_timestep)}: "
                        f"mean_10_reward: {mean_10_reward:.2f}, "
                        f"std_10_reward: {std_10_reward:.2f}, "
                        f"mean_100_reward: {mean_100_reward:.2f}, "
                        f"std_100_reward: {std_100_reward:.2f})")
                
                
            if (episode + 1) % 100 == 0:
                torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, f'ddqn_model_{episode + 1}.pth'))

        torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, 'ddqn_model_final.pth'))
        
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'ddqn_model_final.pth'))

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.online_model = self.value_model_fn(nS, nA)
        self.online_model.load_state_dict(model_state_dict)
        self.online_model.to(self.device)
        self.online_model.eval()
        
        self.evaluation_strategy = self.evaluation_strategy_fn()

        for e in range(n_episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            
            for t in count():
                action = self.evaluation_strategy.select_action(self.online_model, state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    rs.append(episode_reward)
                    print(f"Episode {e+1}: reward: {episode_reward}")
                    break
                
        return np.mean(rs), np.std(rs)
                
                

if __name__ == '__main__':
    value_model_fn = lambda ns, na: FCQ(ns, na, hidden_dims=(512, 128))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    training_strategy_fn = lambda: EpsilonGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.3, decay_steps=20_000)
    evaluation_strategy_fn = lambda: GreedyStrategy()
        
    batch_size = 1024
    epochs = 40
    
    env_name = 'CartPole-v1'
    seed = 42
    gamma = 1
    update_steps = 15
    max_gradient_norm = float('inf')
    
    agent = DDQN(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
    )
    
    replay_buffer = ReplayBuffer(m_size=50_000, batch_size=batch_size)

    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=1000,
        update_steps=update_steps,
        replay_buffer=replay_buffer,
        min_buffer_size=5*batch_size,
        max_gradient_norm=max_gradient_norm,
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")