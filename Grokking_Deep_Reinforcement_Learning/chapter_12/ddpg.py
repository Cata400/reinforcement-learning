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

class FCQV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim += output_dim
            hidden_layer = nn.Linear(in_features=in_dim, out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        
    def forward(self, state, action):
        x, u = state, action
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                            device=self.device, 
                            dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, 
                            device=self.device, 
                            dtype=torch.float32)
            u = u.unsqueeze(0)
            
        x = self.activation_fc(self.input_layer(x))
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        new_states = torch.from_numpy(new_states).float()
        rewards = torch.from_numpy(rewards).float()
        is_terminals = torch.from_numpy(is_terminals).float()
        return states, actions, rewards, new_states, is_terminals
    

class FCDP(nn.Module):
    def __init__(self, input_dim, action_bounds, device, hidden_dims=(32, 32), activation_fc=F.relu, out_activation_fc=F.tanh):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds
        self.device = device
        
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=len(self.env_max))
        
        self.env_min = torch.tensor(self.env_min, dtype=torch.float32).to(self.device)
        self.env_max = torch.tensor(self.env_max, dtype=torch.float32).to(self.device)

        self.nn_min = self.out_activation_fc(torch.Tensor([float('-inf')]).to(self.device))
        self.nn_max = self.out_activation_fc(torch.Tensor([float('inf')]).to(self.device))
        
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min
        
    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0)
            
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
            
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        
        return self.rescale_fn(x)


class GreedyStrategy:
    def __init__(self, bounds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bounds = bounds

    def select_action(self, model, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        
        action = np.clip(greedy_action, self.bounds[0], self.bounds[1])
        return action
        
    
class NormalNoiseStrategy:
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.noise_ratio_injected = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high
            
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            greedy_action = model(state).cpu().detach().data
            greedy_action = greedy_action.numpy().squeeze()
            
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action) / (self.high - self.low)))
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
        
class DDPG:
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr,
                training_strategy_fn, evaluation_strategy_fn):
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
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
        
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        
        q_sa = self.online_value_model(states, actions)
        td_errors = q_sa - target_q_sa.detach()
        value_loss = torch.mean(td_errors**2)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.online_value_model.parameters(), max_norm=self.value_max_gradient_norm) # Huber loss through gradient clipping
        self.value_optimizer.step()
        
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), max_norm=self.policy_max_gradient_norm) # Huber loss through gradient clipping
        self.policy_optimizer.step()
        
    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_policy_model, state, len(self.replay_buffer) < self.min_buffer_size)
        
        new_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated and not truncated
        experience = (state, action, reward, new_state, float(is_failure))
        
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return new_state, terminated or truncated
    
    def update_networks(self):
        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, update_steps, replay_buffer, min_buffer_size,
            value_max_grad_norm, policy_max_grad_norm, tau):
        env = gym.make(env_name, render_mode=env_render_mode)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_value_model = self.value_model_fn(nS, nA)

        self.online_policy_model = self.policy_model_fn(nS, action_bounds, self.device)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds, self.device)

        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, lr=self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, lr=self.policy_optimizer_lr)
        
        self.online_value_model.to(self.device)
        self.target_value_model.to(self.device)
        self.online_policy_model.to(self.device)
        self.target_policy_model.to(self.device)
        
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.gamma = gamma
        self.update_steps = update_steps
        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.value_max_gradient_norm = value_max_grad_norm
        self.policy_max_gradient_norm = policy_max_grad_norm
        self.tau = tau
        
        for episode in tqdm(range(max_episodes)):
            (state, _), is_terminal = env.reset(), False
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                # print(f"Step {step}, state: {state.shape}, replay_buffer size: {len(self.replay_buffer)}", end='\r')
                            
                if len(self.replay_buffer) > self.min_buffer_size:
                    batch = self.replay_buffer.sample()  
                    batch = self.online_value_model.load(batch)    
                    self.optimize_model(batch)
                    
                if np.sum(self.episode_timestep) % self.update_steps == 0:
                    self.update_networks()
                    
                if is_terminal:
                    gc.collect()
                    break
                
            # stats
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}, Timestep {np.sum(self.episode_timestep)}: "
                        f"mean_10_reward: {mean_10_reward:.2f}, "
                        f"std_10_reward: {std_10_reward:.2f}, "
                        f"mean_100_reward: {mean_100_reward:.2f}, "
                        f"std_100_reward: {std_100_reward:.2f})")
                
                
            if (episode + 1) % 50 == 0:
                torch.save(self.online_policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'ddpg_model_{episode + 1}.pth'))

        torch.save(self.online_policy_model.state_dict(), os.path.join(self.checkpoint_dir, 'ddpg_model_final.pth'))
        
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'ddpg_model_final.pth'))

        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.online_policy_model = self.policy_model_fn(nS, action_bounds, self.device)
        self.online_policy_model.load_state_dict(model_state_dict)
        self.online_policy_model.to(self.device)
        self.online_policy_model.eval()

        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)

        for e in range(n_episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            
            for t in count():
                action = self.evaluation_strategy.select_action(self.online_policy_model, state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    rs.append(episode_reward)
                    print(f"Episode {e+1}: reward: {episode_reward}")
                    break
                
        return np.mean(rs), np.std(rs)
                
                

if __name__ == '__main__':
    value_model_fn = lambda ns, na: FCQV(ns, na, hidden_dims=(256, 256))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0003

    policy_model_fn = lambda ns, bounds, device: FCDP(ns, bounds, device, hidden_dims=(256, 256))
    policy_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds=bounds, exploration_noise_ratio=0.1)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)
        
    batch_size = 256
    
    env_name = 'Pendulum-v1'
    seed = 42
    gamma = 0.99
    update_steps = 1
    tau = 0.005
    
    value_max_grad_norm = float('inf')
    policy_max_grad_norm = float('inf')

    agent = DDPG(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
    )
    
    replay_buffer = ReplayBuffer(m_size=50_000, batch_size=batch_size)

    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=500,
        update_steps=update_steps,
        replay_buffer=replay_buffer,
        min_buffer_size=5*batch_size,
        value_max_grad_norm=value_max_grad_norm,
        policy_max_grad_norm=policy_max_grad_norm,
        tau=tau,
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")