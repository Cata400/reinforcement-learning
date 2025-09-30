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
import torch.multiprocessing as mp

class FCDAP(nn.Module): # Fully Connected Discrete Action Policy
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCDAP, self).__init__()
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
    
    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action.item(), log_prob, entropy
    
    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return torch.argmax(logits).item()

class FCV(nn.Module): # Fully Connected Value Function
    def __init__(self, input_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        
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
    
class A3C:
    def __init__(self, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr, value_model_fn, value_optimizer_fn, value_optimizer_lr, device):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = device
    
    def work(self, rank, env_name):
        self.stats['n_active_workers'].add_(1)
        
        env = gym.make(env_name, render_mode=None)
        local_seed = self.seed + rank
        torch.manual_seed(local_seed); np.random.seed(local_seed); random.seed(local_seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())
        
        global_episode_idx = self.stats['episode'].add_(1).item() - 1
        
        while not self.get_out_signal:
            (state, _), terminated = env.reset(), False
            
            n_steps_start, total_episode_rewards = 0, 0
            log_probs, entropies, rewards, values = [], [], [], []
            
            for step in count(start=1):
                state, reward, terminated, truncated = self.interaction_step(
                    state, env, local_policy_model, local_value_model, 
                    log_probs, entropies, rewards, values, self.device)
                
                total_episode_rewards += reward
                
                if terminated or step - n_steps_start == self.max_n_steps:
                    is_failure = terminated and not truncated
                    next_value = 0 if is_failure else \
                        local_value_model(torch.tensor(state, dtype=torch.float32, device=self.device)).detach().item()
                    
                    rewards.append(next_value) # if is_failure next reward is 0, else we bootstrap
                    
                    self.optimize_model(log_probs, entropies, rewards, values,
                                        local_policy_model, local_value_model)
                    
                    log_probs, entropies, rewards, values = [], [], [], []
                    n_steps_start = step
                    
                if terminated:
                    gc.collect()
                    break  
                
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            # stats
            mean_10_reward = self.stats['episode_reward'][:global_episode_idx+1][-10:].mean().item()
            mean_100_reward = self.stats['episode_reward'][:global_episode_idx+1][-100:].mean().item()
            
            self.stats['result'][global_episode_idx][0].add_(mean_10_reward)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            
            if rank == 0:
                if (global_episode_idx + 1) % 100 == 0:
                    print(f"Episode {global_episode_idx + 1}: "
                            f"mean_10_reward: {mean_10_reward:.2f}, "
                            f"mean_100_reward: {mean_100_reward:.2f}")

                if (global_episode_idx + 1) % 100 == 0:
                    torch.save(self.shared_policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'a3c_model_{global_episode_idx + 1}.pth'))
                    
            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        if rank == 0:
            torch.save(self.shared_policy_model.state_dict(), os.path.join(self.checkpoint_dir, 'a3c_model_final.pth'))
            
        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            pass

        env.close() ; del env
        self.stats['n_active_workers'].sub_(1)
        
    def optimize_model(self, log_probs, entropies, rewards, values, local_policy_model, local_value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
        
        discounts = torch.tensor(discounts[:-1], dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = torch.tensor(returns[:-1], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        log_probs = torch.cat(log_probs).to(self.device)
        values = torch.cat(values).to(self.device)
        entropies = torch.cat(entropies).to(self.device)
        
        value_error = returns - values
        
        policy_loss = -(discounts * value_error.detach() * log_probs).mean()
        
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), self.policy_model_max_grad_norm)
        
        for param, shared_param in zip(local_policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        
        value_loss = torch.mean(value_error**2)
        
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), self.value_model_max_grad_norm)
        
        for param, shared_param in zip(local_value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())
    
    @staticmethod    
    def interaction_step(state, env, local_policy_model, local_value_model, 
                    log_probs, entropies, rewards, values, device):
        action, log_prob, entropy = local_policy_model.full_pass(torch.tensor(state, dtype=torch.float32, device=device))
        value = local_value_model(torch.tensor(state, dtype=torch.float32, device=device))
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        entropies.append(entropy)
        
        return next_state, reward, terminated, truncated
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, entropy_loss_weight, 
            policy_model_max_grad_norm, value_model_max_grad_norm, max_n_steps, n_workers):
        env = gym.make(env_name, render_mode=env_render_mode)
        self.seed = seed
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        
        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()
        
        self.gamma = gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        
        self.shared_policy_model = self.policy_model_fn(nS, nA)
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, lr=self.policy_optimizer_lr)
        self.shared_policy_model.to(self.device)
        
        self.shared_value_model = self.value_model_fn(nS)
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, lr=self.value_optimizer_lr)
        self.shared_value_model.to(self.device)
        
        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_() 
        workers = [mp.Process(target=self.work, args=(rank, env_name)) for rank in range(self.n_workers)]
        [w.start() for w in workers] ; [w.join() for w in workers]

        env.close() ; del env
        print("TRAIN DONE")

    def evaluate(self, env_name, env_render_mode, n_episodes=1, greedy=True):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'a3c_model_final.pth'))

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_model.load_state_dict(model_state_dict)
        self.policy_model.to(self.device)
        self.policy_model.eval()
        
        for e in range(n_episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            
            for t in count():
                if greedy:
                    action = self.policy_model.select_greedy_action(torch.tensor(state, dtype=torch.float32, device=self.device))
                else:
                    action = self.policy_model.select_action(torch.tensor(state, dtype=torch.float32, device=self.device))
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    rs.append(episode_reward)
                    print(f"Episode {e+1}: reward: {episode_reward}")
                    break
                
        return np.mean(rs), np.std(rs)
    
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
    
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['step'] = self.state[p]['shared_step']
                self.state[p]['shared_step'] += 1
                
        super().step(closure)
                
                
if __name__ == '__main__':
    policy_model_fn = lambda ns, na: FCDAP(ns, na, hidden_dims=(128, 64))
    policy_optimizer_fn = lambda model, lr: SharedAdam(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005
    
    value_model_fn = lambda ns: FCV(ns, hidden_dims=(256, 128))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0007
        
    env_name = 'CartPole-v1'
    seed = 42 
    gamma = 0.99
    entropy_loss_weight = 0.001
    policy_model_max_grad_norm = 1
    value_model_max_grad_norm = float('inf')
    max_n_steps = 50
    n_workers = 8
    device = 'cpu'
    
    agent = A3C(
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        device=device
    )
    
    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=1_000,
        entropy_loss_weight=entropy_loss_weight,
        policy_model_max_grad_norm=policy_model_max_grad_norm,
        value_model_max_grad_norm=value_model_max_grad_norm,
        max_n_steps=max_n_steps,
        n_workers=n_workers,
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
        greedy=True
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")