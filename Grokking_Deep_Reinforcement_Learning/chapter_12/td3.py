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

class FCTQV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCTQV, self).__init__()
        self.activation_fc = activation_fc
        
        self.input_layer_a = nn.Linear(in_features=input_dim+output_dim, out_features=hidden_dims[0])
        self.input_layer_b = nn.Linear(in_features=input_dim+output_dim, out_features=hidden_dims[0])
        
        
        self.hidden_layers_a = nn.ModuleList()
        self.hidden_layers_b = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer_a = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers_a.append(hidden_layer_a)
            hidden_layer_b = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers_b.append(hidden_layer_b)
            
        self.output_layer_a = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        self.output_layer_b = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        
    def _format(self, x, u):
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
            
        return x, u
        
    def forward(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat((x, u), dim=1)
        
        xa = self.activation_fc(self.input_layer_a(x))
        xb = self.activation_fc(self.input_layer_b(x))
        
        for hidden_layer_a, hidden_layer_b in zip(self.hidden_layers_a, self.hidden_layers_b):
            xa = self.activation_fc(hidden_layer_a(xa))
            xb = self.activation_fc(hidden_layer_b(xb))
            
        xa = self.output_layer_a(xa)
        xb = self.output_layer_b(xb)
        
        return xa, xb
    
    def Qa(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat((x, u), dim=1)
        xa = self.activation_fc(self.input_layer_a(x))
        
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fc(hidden_layer_a(xa))
            
        xa = self.output_layer_a(xa)
        
        return xa
        
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
        
    
class NormalNoiseDecayStrategy():
    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
        self.t = 0
        self.low, self.high = bounds
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps
        self.ratio_noise_injected = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
        noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        return noise_ratio

    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.noise_ratio * self.high

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)

        self.noise_ratio = self._noise_ratio_update()
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
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

    
class TD3:
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
        
        with torch.no_grad():
            env_min = self.target_policy_model.env_min
            env_max = self.target_policy_model.env_max
            
            a_ran = env_max - env_min
            a_noise = torch.rand_like(actions) * self.policy_noise_ratio * a_ran
            
            n_min = env_min * self.policy_noise_clip_ratio
            n_max = env_max * self.policy_noise_clip_ratio
            
            a_noise = torch.clamp(a_noise, n_min, n_max)
            
            argmax_a_q_sp = self.target_policy_model(next_states)
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
            noisy_argmax_a_q_sp = torch.clamp(argmax_a_q_sp, env_min, env_max)
        
            max_a_q_sp_a, max_a_q_sp_b = self.target_value_model(next_states, noisy_argmax_a_q_sp)
            max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)
            target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        
        q_sa_a, q_sa_b = self.online_value_model(states, actions)
        td_errors_a = q_sa_a - target_q_sa.detach()
        td_errors_b = q_sa_b - target_q_sa.detach()
        value_loss = torch.mean(td_errors_a**2) + torch.mean(td_errors_b**2)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.online_value_model.parameters(), max_norm=self.value_max_gradient_norm) # Huber loss through gradient clipping
        self.value_optimizer.step()
        
        if np.sum(self.episode_timestep) % self.train_policy_every_steps == 0:
            argmax_a_q_s = self.online_policy_model(states)
            max_a_q_s = self.online_value_model.Qa(states, argmax_a_q_s)
            
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
    
    def update_value_network(self):
        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def update_policy_network(self):
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, replay_buffer, min_buffer_size,
            value_max_grad_norm, policy_max_grad_norm, tau, policy_noise_ratio, policy_noise_clip_ratio, train_policy_every_steps,
            update_value_target_every_steps, update_policy_target_every_steps):
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
        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.value_max_gradient_norm = value_max_grad_norm
        self.policy_max_gradient_norm = policy_max_grad_norm
        self.tau = tau
        self.policy_noise_ratio = policy_noise_ratio
        self.policy_noise_clip_ratio = policy_noise_clip_ratio
        self.train_policy_every_steps = train_policy_every_steps
        self.update_value_target_every_steps = update_value_target_every_steps
        self.update_policy_target_every_steps = update_policy_target_every_steps
        
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
                    
                if np.sum(self.episode_timestep) % self.update_value_target_every_steps == 0:
                    self.update_value_network()
                    
                if np.sum(self.episode_timestep) % self.update_policy_target_every_steps == 0:
                    self.update_policy_network()
                    
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
                torch.save(self.online_policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'td3_model_{env_name}_{episode + 1}.pth'))

        torch.save(self.online_policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'td3_model_{env_name}_final.pth'))
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, f'td3_model_{env_name}_final.pth'))

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
    value_model_fn = lambda ns, na: FCTQV(ns, na, hidden_dims=(256, 256))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0003

    policy_model_fn = lambda ns, bounds, device: FCDP(ns, bounds, device, hidden_dims=(256, 256))
    policy_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    training_strategy_fn = lambda bounds: NormalNoiseDecayStrategy(bounds, 
                                                                init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=200_000)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)
        
    batch_size = 256
    
    env_name = 'Hopper-v5'
    seed = 42
    gamma = 0.99
    tau = 0.005
    update_value_target_every_steps = 2
    update_policy_target_every_steps = 2
    
    policy_noise_ratio = 0.1
    policy_noise_clip_ratio = 0.5 
    train_policy_every_steps = 2
    
    value_max_grad_norm = float('inf')
    policy_max_grad_norm = float('inf')

    agent = TD3(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
    )
    
    replay_buffer = ReplayBuffer(m_size=1_000_000, batch_size=batch_size)

    # agent.train(
    #     env_name=env_name,
    #     env_render_mode=None,
    #     seed=seed,
    #     gamma=gamma,
    #     max_episodes=10_000,
    #     replay_buffer=replay_buffer,
    #     min_buffer_size=5*batch_size,
    #     value_max_grad_norm=value_max_grad_norm,
    #     policy_max_grad_norm=policy_max_grad_norm,
    #     tau=tau,
    #     policy_noise_ratio=policy_noise_ratio,
    #     policy_noise_clip_ratio=policy_noise_clip_ratio,
    #     train_policy_every_steps=train_policy_every_steps,
    #     update_value_target_every_steps=update_value_target_every_steps,
    #     update_policy_target_every_steps=update_policy_target_every_steps,
    # )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")