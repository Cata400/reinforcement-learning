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
    
    
class REINFORCE:
    def __init__(self, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])
        
        discounts = torch.tensor(discounts, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.log_probs = torch.cat(self.log_probs).to(self.device)
        
        policy_loss = -(discounts * returns * self.log_probs).mean()
        # policy_loss = -(returns * self.log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def interaction_step(self, state, env):
        action, log_prob, _ = self.policy_model.full_pass(torch.tensor(state, dtype=torch.float32, device=self.device))
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return next_state, terminated or truncated
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes):
        env = gym.make(env_name, render_mode=env_render_mode)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        
        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, lr=self.policy_optimizer_lr)
        self.gamma = gamma
        self.policy_model.to(self.device)
        
        for episode in tqdm(range(max_episodes)):
            state, _ = env.reset()
            self.log_probs = []
            self.rewards = []
            
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                    
                if is_terminal:
                    gc.collect()
                    break
                
            self.optimize_model()
                
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
                torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'reinforce_model_{episode + 1}.pth'))

        torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, 'reinforce_model_final.pth'))


    def evaluate(self, env_name, env_render_mode, n_episodes=1, greedy=True):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'reinforce_model_final.pth'))

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

if __name__ == '__main__':
    policy_model_fn = lambda ns, na: FCDAP(ns, na, hidden_dims=(128, 64))
    policy_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005
        
    env_name = 'CartPole-v1'
    seed = 42
    gamma = 0.99
    
    agent = REINFORCE(
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
    )
    
    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=10_000,
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
        greedy=True
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")