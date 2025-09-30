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

class FCDuelingQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCDuelingQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.value_output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        self.advantage_output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
        
    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0)
            
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        
        x_advantage = self.advantage_output_layer(x)   
        x_value = self.value_output_layer(x)
        x_value = x_value.expand_as(x_advantage)
        
        x = x_value + x_advantage - torch.mean(x_advantage, dim=1, keepdim=True).expand_as(x_advantage)
        
        return x
    
    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        # ensure actions have shape (batch, 1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        rewards = torch.from_numpy(rewards).float()
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        new_states = torch.from_numpy(new_states).float()
        is_terminals = torch.from_numpy(is_terminals).float()
        if is_terminals.dim() == 1:
            is_terminals = is_terminals.unsqueeze(1)
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
    
class PrioritizedReplayBuffer:
    def __init__(self, max_samples=50_000, batch_size=64, rank_based=True, alpha=0.6, beta=0.1, beta_rate=0.9999):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.rank_based = rank_based
        self.alpha = alpha
        self.beta = beta
        self.beta_rate = beta_rate
        
        self.memory = np.empty(shape=(max_samples, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.next_index = 0 
        
        self.td_error_index = 0 
        self.sample_index = 1
        
        self.eps = ...
        self.beta_0 = beta    
        
    def store(self, sample):        
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[:self.n_entries, self.td_error_index].max()
            
        self.memory[self.next_index, self.td_error_index] = priority
        self.memory[self.next_index, self.sample_index] = np.array([s for s in sample], dtype=object)
        
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index %= self.max_samples
        
    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]
            
    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate ** (-1))
        return self.beta
        
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        self._update_beta()
        entries = self.memory[:self.n_entries]
        
        if self.rank_based:
            priorities = 1 / (np.arange(self.n_entries) + 1)
        else:
            priorities = entries[: self.td_error_index] + self.eps
            
        scaled_priorities = priorities ** self.alpha
        priority_sum = np.sum(scaled_priorities)
        probs = np.array(scaled_priorities / priority_sum, dtype=np.float64)
        
        weights = (self.n_entries * probs) ** (-self.beta)
        normalized_weights = weights / weights.max()
        
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.stack(batch_type) for batch_type in np.stack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        
        return idxs_stack, weights_stack, samples_stacks
    
    def __len__(self):
        return self.n_entries
        
class DuelingDDQNPER:
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
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experiences
        batch_size = len(is_terminals)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        is_terminals = is_terminals.to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        argmax_a_q_sp = self.online_model(next_states).detach().max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[torch.arange(batch_size), argmax_a_q_sp].unsqueeze(1) * (1 - is_terminals)
        target_q_sa = rewards + self.gamma * max_a_q_sp
        
        q_sa = self.online_model(states).gather(1, actions)
        td_errors = weights * (q_sa - target_q_sa)
        value_loss = torch.mean(td_errors**2)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=self.max_gradient_norm) # Huber loss through gradient clipping
        self.value_optimizer.step()
        
        priorities = np.abs(td_errors.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)
        
    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated and not truncated
        experience = (state, action, reward, next_state, float(is_failure))
        
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        
        return next_state, terminated or truncated
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            update = (1 - self.tau) * target.data + self.tau * online.data
            target.data.copy_(update)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, update_steps, replay_buffer, min_buffer_size, max_gradient_norm, tau):
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
        self.tau = tau
        
        for episode in tqdm(range(max_episodes)):
            state, _ = env.reset()
            self.episode_reward.append(0)
            self.episode_timestep.append(0)
            
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                            
                if len(self.replay_buffer) > self.min_buffer_size:
                    idxs, weights, samples = self.replay_buffer.sample()  
                    experiences = self.online_model.load(samples)
                    experiences = (idxs, weights) + (experiences, )    
                    self.optimize_model(experiences)
                    
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
                torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, f'dueling_ddqn_per_model_{episode + 1}.pth'))

        torch.save(self.online_model.state_dict(), os.path.join(self.checkpoint_dir, 'dueling_ddqn_per_model_final.pth'))
        
        
    def evaluate(self, env_name, env_render_mode, n_episodes=1):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'dueling_ddqn_per_model_final.pth'))

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
    value_model_fn = lambda ns, na: FCDuelingQ(ns, na, hidden_dims=(512, 128))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 2.5e-4

    training_strategy_fn = lambda: EpsilonGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.01, decay_steps=20_000)
    evaluation_strategy_fn = lambda: GreedyStrategy()
        
    batch_size = 64
    epochs = 40
    
    env_name = 'CartPole-v1'
    seed = 42
    gamma = 1
    update_steps = 1
    max_gradient_norm = float('inf')
    tau = 0.005
    
    agent = DuelingDDQNPER(
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
    )
    
    replay_buffer = PrioritizedReplayBuffer(max_samples=50_000, batch_size=batch_size)

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
        tau=tau
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")