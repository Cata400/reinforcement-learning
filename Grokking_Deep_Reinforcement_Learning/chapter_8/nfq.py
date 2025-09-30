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
    
    
class EpsilonGreedyStrategy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, model, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = model(state).cpu().detach()
            q_values = q_values.data.numpy().squeeze()
            
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        
        return action
    
class GreedyStrategy:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def select_action(self, model, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)
    

class NFQ:
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn, evaluation_strategy_fn, batch_size, epochs):
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        is_terminals = is_terminals.to(self.device).unsqueeze(1)
        
        q_sp = self.online_model(next_states).detach()
        max_a_q_sp = q_sp.max(1)[0].unsqueeze(1) * (1 - is_terminals)
        target_q_s = rewards + self.gamma * max_a_q_sp
        
        q_sa = self.online_model(states).gather(1, actions.unsqueeze(1))
        td_errors = q_sa - target_q_s
        value_loss = torch.mean(td_errors**2)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated and not truncated
        experience = (state, action, reward, next_state, float(is_failure))
        
        self.experiences.append(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return next_state, terminated
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes):
        env = gym.make(env_name, render_mode=env_render_mode)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        
        self.online_model = self.value_model_fn(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(self.online_model, lr=self.value_optimizer_lr)
        self.training_strategy = self.training_strategy_fn()
        self.gamma = gamma
        self.online_model.to(self.device)
        self.experiences = []
        
        for episode in tqdm(range(max_episodes)):
            state, _ = env.reset()
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                
                if len(self.experiences) >= self.batch_size:
                    batches = list(zip(*self.experiences))
                    batches = [np.array(batch) for batch in batches]
                    experiences = self.online_model.load(batches)
                    
                    for _ in range(self.epochs):
                        self.optimize_model(experiences)
                    self.experiences.clear()
                    
                if is_terminal:
                    gc.collect()
                    break
                
            # stats
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}: "
                        f"mean_10_reward: {mean_10_reward:.2f}, "
                        f"std_10_reward: {std_10_reward:.2f}, "
                        f"mean_100_reward: {mean_100_reward:.2f}, "
                        f"std_100_reward: {std_100_reward:.2f})")

                torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, f'nfq_model_{episode + 1}.pth'))

        torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, 'nfq_model_final.pth'))
        
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'nfq_model_final.pth'))
        
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
    
    training_statagy_fn = lambda: EpsilonGreedyStrategy(epsilon=0.5)
    evaluation_strategy_fn = lambda: GreedyStrategy()
    
    batch_size = 1024
    epochs = 40
    
    env_name = 'CartPole-v1'
    seed = 42
    gamma = 1
    
    agent = NFQ(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        training_strategy_fn=training_statagy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
        batch_size=batch_size,
        epochs=epochs,
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
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")