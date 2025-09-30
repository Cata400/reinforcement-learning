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


class FCAC(nn.Module): # Fully Connected Discrete Action Policy
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.policy_output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
        self.value_output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        
    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
            
            x = x.unsqueeze(0)
            
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        
        return self.policy_output_layer(x), self.value_output_layer(x)
    
    def full_pass(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        action = action.item() if action.numel() == 1 else action.detach().cpu().numpy()

        return action, log_prob, entropy, value
    
    def select_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if action.numel() == 1 else action.detach().cpu().numpy()

        return action
    
    def select_greedy_action(self, state):
        logits, _ = self.forward(state)
        return torch.argmax(logits).item()
    
    def evaluate_state(self, state):
        _, value = self.forward(state)
        return value


class MultiprocessEnv:
    def __init__(self, env_name, env_render_mode, seed, n_workers):
        self.env_name = env_name
        self.env_render_mode = env_render_mode
        self.seed = seed
        self.n_workers = n_workers
        
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        
        self.workers = [
            mp.Process(target=self.work, args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)
        ]
        [w.start() for w in self.workers]
        
    def work(self, rank, worker_end):
        env = gym.make(self.env_name, render_mode=self.env_render_mode)
        local_seed = self.seed + rank
        torch.manual_seed(local_seed); np.random.seed(local_seed); random.seed(local_seed)
        
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            elif cmd == '_past_limit':
                worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                env.close(**kwargs)
                del env
                worker_end.close()
                break
            
    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(('step', {'action': actions[rank]}), rank) for rank in range(self.n_workers)]

        observations, rewards, terminateds, truncateds = [], [], [], []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            observation, reward, terminated, truncated, _ = parent_end.recv()
            if terminated or truncated:
                self.send_msg(('reset', {}), rank)
                observation, _ = parent_end.recv()
            observations.append(observation)
            rewards.append(np.array(reward))
            terminateds.append(np.array(terminated, dtype=np.float64))
            truncateds.append(np.array(truncated, dtype=np.float64))

        return np.vstack(observations), np.vstack(rewards), np.vstack(terminateds), np.vstack(truncateds), _

    def reset(self, rank=None, **kwargs):
        if rank is not None:
            parent_end, _ = self.pipes[rank]
            self.send_msg(('reset', {}), rank)
            observation, _ = parent_end.recv()
            return observation
        
        self.broadcast_msg(('reset', kwargs))
        observations = []
        for parent_end, _ in self.pipes:
            obs, _ = parent_end.recv()   # env.reset() returns (obs, info)
            observations.append(np.asarray(obs))

        return np.vstack(observations)
    
    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]
        
    def _past_limit(self, **kwargs):
        self.broadcast_msg(('_past_limit', kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])
    
    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):    
        [parent_end.send(msg) for parent_end, _ in self.pipes]
    
    def get_nS_nA(self):
        env = gym.make(self.env_name, render_mode=self.env_render_mode)
        
        nS, nA = env.observation_space.shape[0], env.action_space.n
        env.close()
        
        return nS, nA
        

class A2C:
    def __init__(self, ac_model_fn, ac_optimizer_fn, ac_optimizer_lr, make_envs_fn):
        self.ac_model_fn = ac_model_fn
        self.ac_optimizer_fn = ac_optimizer_fn
        self.ac_optimizer_lr = ac_optimizer_lr
        self.make_envs_fn = make_envs_fn
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_model(self):
        log_probs = torch.stack(self.log_probs).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()

        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T-t] * rewards[t:, w]) for t in range(T)] 
                            for w in range(self.n_workers)])

        np_values = values.data.detach().cpu().numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([[np.sum(tau_discounts[:T-1-t] * advs[t:, w]) for t in range(T-1)] 
                            for w in range(self.n_workers)])
        discounted_gaes = discounts[:-1] * gaes
        
        values = values[:-1,...].view(-1).unsqueeze(1)
        log_probs = log_probs.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).reshape(-1, 1).to(self.device)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1, 1).to(self.device)

        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert log_probs.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + \
                self.value_loss_weight * value_loss + \
                self.entropy_loss_weight * entropy_loss

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 
                                    self.ac_model_max_grad_norm)
        self.ac_optimizer.step()
        
    def interaction_step(self, state, envs):
        actions, log_probs, entropies, values = self.ac_model.full_pass(torch.tensor(state, dtype=torch.float32, device=self.device))
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.entropies.append(entropies)

        self.running_rewards += rewards
        self.running_timestep += 1
        
        return next_states, np.logical_or(terminateds, truncateds)
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, policy_loss_weight, value_loss_weight, 
            entropy_loss_weight, ac_model_max_grad_norm, n_workers, max_n_steps, tau):
        envs = self.make_envs_fn(env_name, env_render_mode, seed, n_workers)
        
        nS, nA = envs.get_nS_nA()
        
        self.gamma = gamma
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.ac_model_max_grad_norm = ac_model_max_grad_norm
        self.n_workers = n_workers
        self.max_n_steps = max_n_steps
        self.tau = tau
        
        self.running_timestep = np.array([[0.],] * self.n_workers)
        self.running_rewards = np.array([[0.],] * self.n_workers)
        
        self.ac_model = self.ac_model_fn(nS, nA)
        self.ac_optimizer = self.ac_optimizer_fn(self.ac_model, lr=self.ac_optimizer_lr)
        self.ac_model.to(self.device)
        
        states = envs.reset()
        self.episode_timestep = []
        self.episode_reward = []

        episode, n_steps_start = 0, 0
        self.log_probs, self.entropies = [], []
        self.rewards, self.values = [], []
        
        for step in count():
            states, is_terminals = self.interaction_step(states, envs)
                
            if is_terminals.sum() or step - n_steps_start == self.max_n_steps:
                past_limits_enforced = envs._past_limit()
                failure = np.logical_and(is_terminals, np.logical_not(past_limits_enforced))
                next_values = self.ac_model.evaluate_state(torch.tensor(states, device=self.device)).detach().cpu().numpy() * (1 - failure)
                self.rewards.append(next_values)
                self.values.append(torch.tensor(next_values, device=self.device))
                self.optimize_model()
                self.log_probs, self.entropies = [], []
                self.rewards, self.values = [], []
                n_steps_start = step
                
            if is_terminals.sum():
                for i in range(self.n_workers):
                    if is_terminals[i]:
                        states[i] = envs.reset(rank=i)
                        self.episode_timestep.append(self.running_timestep[i][0])
                        self.episode_reward.append(self.running_rewards[i][0])
                        episode += 1
                        
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
                    torch.save(self.ac_model.state_dict(), os.path.join(self.checkpoint_dir, f'a2c_model_{episode + 1}.pth'))
                    
                if episode >= max_episodes:
                    break
                
                self.running_timestep *= 1 - is_terminals
                self.running_rewards *= 1 - is_terminals

        torch.save(self.ac_model.state_dict(), os.path.join(self.checkpoint_dir, 'a2c_model_final.pth'))


    def evaluate(self, env_name, env_render_mode, n_episodes=1, greedy=True):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, 'a2c_model_final.pth'))

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.ac_model = self.ac_model_fn(nS, nA)
        self.ac_model.load_state_dict(model_state_dict)
        self.ac_model.to(self.device)
        self.ac_model.eval()
        
        for e in range(n_episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            
            for t in count():
                if greedy:
                    action = self.ac_model.select_greedy_action(torch.tensor(state, dtype=torch.float32, device=self.device))
                else:
                    action = self.ac_model.select_action(torch.tensor(state, dtype=torch.float32, device=self.device))
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    rs.append(episode_reward)
                    print(f"Episode {e+1}: reward: {episode_reward}")
                    break
                
        return np.mean(rs), np.std(rs)

if __name__ == '__main__':
    ac_model_fn = lambda ns, na: FCAC(ns, na, hidden_dims=(128, 64))
    ac_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    ac_optimizer_lr = 0.0005
    
    make_envs_fn = lambda env_name, env_render_mode, seed, n_workers: MultiprocessEnv(env_name, env_render_mode, seed, n_workers)
        
    env_name = 'CartPole-v1'
    seed = 42 
    gamma = 0.99
    policy_loss_weight = 1.0
    value_loss_weight = 0.6
    entropy_loss_weight = 0.001
    ac_model_max_grad_norm = 1
    max_n_steps = 50
    n_workers = 8
    tau = 0.95
    
    agent = A2C(
        ac_model_fn=ac_model_fn,
        ac_optimizer_fn=ac_optimizer_fn,
        ac_optimizer_lr=ac_optimizer_lr,
        make_envs_fn=make_envs_fn
    )
    
    # agent.train(
    #     env_name=env_name,
    #     env_render_mode=None,
    #     seed=seed,
    #     gamma=gamma,
    #     max_episodes=10_000,
    #     policy_loss_weight=policy_loss_weight,
    #     value_loss_weight=value_loss_weight,
    #     entropy_loss_weight=entropy_loss_weight,
    #     ac_model_max_grad_norm=ac_model_max_grad_norm,
    #     max_n_steps=max_n_steps,
    #     n_workers=n_workers,
    #     tau=tau
    # )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
        greedy=False
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")