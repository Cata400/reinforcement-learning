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


class FCQSA(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCQSA, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim+output_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        
        self.device = device
        
    def forward(self, state, action):   
        x, u = state, action
            
        x = torch.cat((x, u), dim=1) 
        x = self.activation_fc(self.input_layer(x))
        
        for i, hidden_layer in enumerate(self.hidden_layers):
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
    

class FCGP(nn.Module):
    def __init__(self, input_dim, action_bounds, device, entropy_lr=0.001, log_std_min=-20, log_std_max=2, hidden_dims=(32, 32), 
                activation_fc=F.relu, out_activation_fc=F.tanh):
        super(FCGP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds
        self.device = device
        self.entropy_lr = entropy_lr
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer_mean = nn.Linear(in_features=hidden_dims[-1], out_features=len(self.env_max))
        self.output_layer_log_std = nn.Linear(in_features=hidden_dims[-1], out_features=len(self.env_max))
        
        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.logalpha], lr=self.entropy_lr)
        
        self.env_min = torch.tensor(self.env_min, dtype=torch.float32).to(self.device)
        self.env_max = torch.tensor(self.env_max, dtype=torch.float32).to(self.device)

        self.nn_min = self.out_activation_fc(torch.Tensor([float('-inf')]).to(self.device))
        self.nn_max = self.out_activation_fc(torch.Tensor([float('inf')]).to(self.device))
        
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min
        
    def forward(self, state):
        x = state
            
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
            
        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, self.log_std_min, self.log_std_max)
        
        return x_mean, x_log_std
    
    def full_pass(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        pi_s = torch.distributions.Normal(mean, std)

        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)
        
        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1 - tanh_action**2).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, self.rescale_fn(torch.tanh(mean))
    
    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken)/(env_max - env_min)))

    def _get_actions(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        action = self.rescale_fn(torch.tanh(torch.distributions.Normal(mean, std).sample()))
        greedy_action = self.rescale_fn(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(),
                                        high=self.env_max.cpu().numpy())

        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
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

    
class SAC:
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr):
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
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

        # 1) Compute targets using the target networks (no grad)
        with torch.no_grad():
            next_actions, logpi_sp, _ = self.policy_model.full_pass(next_states)
            q_spap_a = self.target_value_model_a(next_states, next_actions)
            q_spap_b = self.target_value_model_b(next_states, next_actions)
            q_spap = torch.min(q_spap_a, q_spap_b) - self.policy_model.logalpha.exp() * logpi_sp
            target_q_sa = (rewards + self.gamma * q_spap * (1 - is_terminals)).detach()

        # 2) Update value (critic) networks
        q_sa_a = self.online_value_model_a(states, actions)
        q_sa_b = self.online_value_model_b(states, actions)
        qa_loss = torch.mean((q_sa_a - target_q_sa) ** 2)
        qb_loss = torch.mean((q_sa_b - target_q_sa) ** 2)

        self.value_optimizer_a.zero_grad()
        qa_loss.backward()
        nn.utils.clip_grad_norm_(self.online_value_model_a.parameters(), max_norm=self.value_max_gradient_norm)
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        qb_loss.backward()
        nn.utils.clip_grad_norm_(self.online_value_model_b.parameters(), max_norm=self.value_max_gradient_norm)
        self.value_optimizer_b.step()

        # 3) Now update entropy alpha and the policy
        current_actions, logpi_s, _ = self.policy_model.full_pass(states)

        # alpha loss
        target_alpha = self.policy_model.target_entropy
        alpha_loss = -(self.policy_model.logalpha * (logpi_s + target_alpha).detach()).mean()
        self.policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_model.alpha_optimizer.step()
        alpha = self.policy_model.logalpha.exp()

        # compute policy loss using the updated value networks
        current_q_sa_a = self.online_value_model_a(states, current_actions)
        current_q_sa_b = self.online_value_model_b(states, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)

        policy_loss = (alpha * logpi_s - current_q_sa).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.policy_max_gradient_norm)
        self.policy_optimizer.step()
        
    def interaction_step(self, state, env):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        if len(self.replay_buffer) < self.min_buffer_size:
            action = self.policy_model.select_random_action(state_tensor)
        else:
            action = self.policy_model.select_action(state_tensor)

        new_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated and not truncated
        experience = (state, action, reward, new_state, float(is_failure))
        
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return new_state, terminated or truncated
    
    def update_value_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model_a.parameters(), self.online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
            
        for target, online in zip(self.target_value_model_b.parameters(), self.online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, replay_buffer, min_buffer_size,
            value_max_grad_norm, policy_max_grad_norm, tau, update_target_every_steps):
        env = gym.make(env_name, render_mode=env_render_mode)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        
        self.online_value_model_a = self.value_model_fn(nS, nA, self.device)
        self.target_value_model_a = self.value_model_fn(nS, nA, self.device)
        
        self.online_value_model_b = self.value_model_fn(nS, nA, self.device)
        self.target_value_model_b = self.value_model_fn(nS, nA, self.device)

        self.policy_model = self.policy_model_fn(nS, action_bounds, self.device)

        self.value_optimizer_a = self.value_optimizer_fn(self.online_value_model_a, lr=self.value_optimizer_lr)
        self.value_optimizer_b = self.value_optimizer_fn(self.online_value_model_b, lr=self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, lr=self.policy_optimizer_lr)
        self.update_value_networks(tau=1.0)
        
        self.online_value_model_a.to(self.device)
        self.target_value_model_a.to(self.device)
        self.online_value_model_b.to(self.device)
        self.target_value_model_b.to(self.device)
        self.policy_model.to(self.device)
        
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.value_max_gradient_norm = value_max_grad_norm
        self.policy_max_gradient_norm = policy_max_grad_norm
        self.tau = tau
        self.update_target_every_steps = update_target_every_steps
        
        for episode in tqdm(range(max_episodes)):
            (state, _), is_terminal = env.reset(), False
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                # print(f"Step {step}, state: {state.shape}, replay_buffer size: {len(self.replay_buffer)}", end='\r')
                            
                if len(self.replay_buffer) > self.min_buffer_size:
                    batch = self.replay_buffer.sample()  
                    batch = self.online_value_model_a.load(batch)    
                    self.optimize_model(batch)
                    
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_value_networks()
                    
                if is_terminal:
                    gc.collect()
                    break
                
            # stats
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            if (episode + 1) % (max_episodes // 10) == 0:
                print(f"Episode {episode + 1}, Timestep {np.sum(self.episode_timestep)}: "
                        f"mean_10_reward: {mean_10_reward:.2f}, "
                        f"std_10_reward: {std_10_reward:.2f}, "
                        f"mean_100_reward: {mean_100_reward:.2f}, "
                        f"std_100_reward: {std_100_reward:.2f})")
                
                
            if (episode + 1) % (max_episodes // 10) == 0:
                torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'sac_model_{env_name}_{episode + 1}.pth'))

        torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'sac_model_{env_name}_final.pth'))
        
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, f'sac_model_{env_name}_final.pth'))

        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.policy_model = self.policy_model_fn(nS, action_bounds, self.device)
        self.policy_model.load_state_dict(model_state_dict)
        self.policy_model.to(self.device)
        self.policy_model.eval()

        for e in range(n_episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            
            for t in count():
                action = self.policy_model.select_action(torch.tensor(state, device=self.device, dtype=torch.float32))
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    rs.append(episode_reward)
                    print(f"Episode {e+1}: reward: {episode_reward}")
                    break
                
        return np.mean(rs), np.std(rs)
                
                

if __name__ == '__main__':
    value_model_fn = lambda ns, na, device: FCQSA(ns, na, device, hidden_dims=(256, 256))
    value_optimizer_fn = lambda model, lr: torch.optim.Adam(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    policy_model_fn = lambda ns, bounds, device: FCGP(ns, bounds, device, hidden_dims=(256, 256))
    policy_optimizer_fn = lambda model, lr: torch.optim.Adam(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003
        
    batch_size = 256
    
    env_name = 'HalfCheetah-v5'
    seed = 42
    gamma = 0.99
    tau = 0.001
    update_target_every_steps = 1
    
    value_max_grad_norm = float('inf')
    policy_max_grad_norm = float('inf')

    agent = SAC(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
    )
    
    replay_buffer = ReplayBuffer(m_size=100_000, batch_size=batch_size)

    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=3_000,
        replay_buffer=replay_buffer,
        min_buffer_size=10*batch_size,
        value_max_grad_norm=value_max_grad_norm,
        policy_max_grad_norm=policy_max_grad_norm,
        tau=tau,
        update_target_every_steps=update_target_every_steps,
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")