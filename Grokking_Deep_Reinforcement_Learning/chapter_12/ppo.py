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


class FCCA(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dims=(32, 32), activation_fc=F.relu):
        super(FCCA, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(in_features=input_dim[0], out_features=hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
        
        self.device = device
        
    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        return self.output_layer(x)

    def np_pass(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        log_probs = dist.log_prob(actions)
        np_log_probs = log_probs.detach().cpu().numpy()
        return np_actions, np_log_probs
    
    def select_action(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu().item()
    
    def get_predictions(self, states, actions):
        states, actions = self._format(states), self._format(actions)
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, entropies
    
    def select_greedy_action(self, states):
        logits = self.forward(states)
        return np.argmax(logits.detach().squeeze().cpu().numpy())
    
    
class FCV(nn.Module):
    def __init__(self, input_dim, device, hidden_dims=(32,32), activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim[0], hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.device = device

    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, states):
        x = self._format(states)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x).squeeze()


class EpisodeBuffer:
    def __init__(self, state_dim, gamma, tau, n_workers, max_episodes, max_episode_steps, device):
        assert max_episodes >= n_workers

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps

        self.discounts = np.logspace(
            0, max_episode_steps+1, num=max_episode_steps+1, base=gamma, endpoint=False, dtype=np.float128)
        self.tau_discounts = np.logspace(
            0, max_episode_steps+1, num=max_episode_steps+1, base=gamma*tau, endpoint=False, dtype=np.float128)

        self.device = device

        self.clear()

    def clear(self):
        self.states_mem = np.empty(
            shape=np.concatenate(((self.max_episodes, self.max_episode_steps), self.state_dim)), dtype=np.float64)
        self.states_mem[:] = np.nan

        self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.actions_mem[:] = np.nan

        self.returns_mem = np.empty(shape=(self.max_episodes,self.max_episode_steps), dtype=np.float32)
        self.returns_mem[:] = np.nan

        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem[:] = np.nan

        self.log_probs_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.log_probs_mem[:] = np.nan

        self.episode_steps = np.zeros(shape=(self.max_episodes), dtype=np.uint16)
        self.episode_reward = np.zeros(shape=(self.max_episodes), dtype=np.float32)

        self.current_ep_idxs = np.arange(n_workers, dtype=np.uint16)
        gc.collect()
    
    def fill(self, envs, policy_model, value_model):
        states = envs.reset()
        we_shape = (n_workers, self.max_episode_steps)
        
        worker_rewards = np.zeros(shape=we_shape, dtype=np.float32)
        worker_steps = np.zeros(shape=(n_workers), dtype=np.uint16)

        buffer_full = False
        while not buffer_full and len(self.episode_steps[self.max_episode_steps > 0]) < self.max_episodes / 2:
            with torch.no_grad():
                actions, log_probs = policy_model.np_pass(torch.tensor(states, dtype=torch.float32, device=self.device))
                values = value_model(torch.tensor(states, dtype=torch.float32, device=self.device))
                
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.log_probs_mem[self.current_ep_idxs, worker_steps] = log_probs
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards.squeeze()
            
            terminals = np.logical_or(terminateds, truncateds)
            
            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
                next_values = np.zeros(shape=(n_workers))
                
                if truncateds.sum():
                    idx_truncateds = np.flatnonzero(truncateds)
                    with torch.no_grad():
                        next_values[idx_truncateds] = value_model(
                            torch.tensor(next_states[idx_truncateds], dtype=torch.float32, device=self.device)).cpu().numpy()
                        
            states = next_states
            worker_steps += 1
            
            if terminals.sum():
                new_states = envs.reset(ranks=idx_terminals)
                states[idx_terminals] = new_states
                
                for w_idx in range(self.n_workers):
                    if w_idx not in idx_terminals:
                        continue
                    
                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    
                    ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_discounts = self.discounts[:T+1]
                    ep_returns = np.array([np.sum(ep_discounts[:T+1-t] * ep_rewards[t:]) for t in range(T)])
                    self.returns_mem[e_idx, :T] = ep_returns
                    
                    ep_states = self.states_mem[e_idx, :T]
                    with torch.no_grad():
                        ep_values = torch.cat((
                            value_model(torch.tensor(ep_states, dtype=torch.float32, device=self.device)),
                            torch.tensor([next_values[w_idx]], device=self.device, dtype=torch.float32)))

                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    ep_tau_discounts = self.tau_discounts[:T]
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    gaes = np.array([np.sum(ep_tau_discounts[:T-t] * deltas[t:]) for t in range(T)])
                    self.gaes_mem[e_idx, :T] = gaes
                    
                    worker_rewards[w_idx] = 0
                    worker_steps[w_idx] = 0
                    
                    new_ep_id = max(self.current_ep_idxs) + 1
                    if new_ep_id >= self.max_episodes:
                        buffer_full = True
                        break
                    self.current_ep_idxs[w_idx] = new_ep_id
                    
        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]
        
        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)

        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = np.concatenate(self.returns_mem)
        
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = np.concatenate(self.gaes_mem)

        self.log_probs_mem = [row[:ep_t[i]] for i, row in enumerate(self.log_probs_mem[ep_idxs])]
        self.log_probs_mem = np.concatenate(self.log_probs_mem)
        
        ep_r = self.episode_reward[ep_idxs]

        return ep_r
    
    def get_stacks(self):
        return (self.states_mem, self.actions_mem,
                self.returns_mem, self.gaes_mem, self.log_probs_mem)

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()


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
        self.dones = {rank: False for rank in range(self.n_workers)}
        
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

    def reset(self, ranks=None, **kwargs):
        if not (ranks is None):
            [self.send_msg(('reset', {}), rank) for rank in ranks]  
            observations = []
            for rank, (parent_end, _) in enumerate(self.pipes):
                if rank in ranks:
                    obs, _ = parent_end.recv()
                    observations.append(np.asarray(obs))
            return np.stack(observations)

        self.broadcast_msg(('reset', kwargs))
        observations = [parent_end.recv()[0] for parent_end, _ in self.pipes]
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
        
        nS, nA = env.observation_space.shape, env.action_space.n
        env.close()
        
        return nS, nA
        

class PPO:
    def __init__(self, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr, 
                value_model_fn, value_optimizer_fn, value_optimizer_lr, make_envs_fn):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.make_envs_fn = make_envs_fn
        
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_model(self):
        states, actions, returns, gaes, log_probs = self.episode_buffer.get_stacks()
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        gaes = torch.tensor(gaes, dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        
        values = self.value_model(states).detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        n_samples = len(actions)
        
        for i in range(self.policy_optimization_epochs):
            batch_size = int(self.policy_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, size=batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            log_probs_batch = log_probs[batch_idxs]
            
            states_batch = torch.tensor(states_batch, dtype=torch.float32, device=self.device)
            actions_batch = torch.tensor(actions_batch, dtype=torch.int64, device=self.device)
            gaes_batch = torch.tensor(gaes_batch, dtype=torch.float32, device=self.device)
            log_probs_batch = torch.tensor(log_probs_batch, dtype=torch.float32, device=self.device)

            log_probs_pred, entropies_pred = self.policy_model.get_predictions(states_batch, actions_batch)
            ratios = (log_probs_pred - log_probs_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * torch.clamp(ratios, 1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range)
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight
            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_grad_norm)
            self.policy_optimizer.step()
            
            with torch.no_grad():
                log_probs_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (log_probs_all - log_probs).mean().item()
                if kl > self.policy_stopping_kl:
                    break
                
        for i in range(self.value_optimization_epochs):
            batch_size = int(self.value_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, size=batch_size, replace=False)
            states_batch = states[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]
            
            values_pred = self.value_model(states_batch)
            v_loss = (values_pred - returns_batch)**2
            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range, self.value_clip_range)
            v_loss_clipped = (values_pred_clipped - returns_batch)**2
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)
            self.value_optimizer.step()
            
            with torch.no_grad():
                values_pred_all = self.value_model(states)
                mse = (values_pred_all - values).pow(2).mean().item()
                if mse > self.value_stopping_mse:
                    break
    
    def train(self, env_name, env_render_mode, seed, gamma, max_episodes, n_workers, tau, episode_buffer_fn, max_buffer_episodes, 
            max_buffer_episode_steps, policy_optimization_epochs, policy_sample_ratio, policy_clip_range, policy_stopping_kl,
            value_optimization_epochs, value_sample_ratio, value_clip_range, value_stopping_mse,
            policy_model_max_grad_norm, value_model_max_grad_norm, entropy_loss_weight):
        envs = self.make_envs_fn(env_name, env_render_mode, seed, n_workers)
        
        nS, nA = envs.get_nS_nA()
        
        self.gamma = gamma
        self.n_workers = n_workers
        self.tau = tau
        self.episode_buffer_fn = episode_buffer_fn
        self.max_buffer_episodes = max_buffer_episodes
        self.max_buffer_episode_steps = max_buffer_episode_steps
        self.policy_optimization_epochs = policy_optimization_epochs
        self.policy_sample_ratio = policy_sample_ratio
        self.policy_clip_range = policy_clip_range
        self.policy_stopping_kl = policy_stopping_kl
        self.value_optimization_epochs = value_optimization_epochs
        self.value_sample_ratio = value_sample_ratio
        self.value_clip_range = value_clip_range
        self.value_stopping_mse = value_stopping_mse
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.entropy_loss_weight = entropy_loss_weight

        self.policy_model = self.policy_model_fn(nS, nA, self.device)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)
        self.policy_model.to(self.device)

        self.value_model = self.value_model_fn(nS, self.device)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)
        self.value_model.to(self.device)
        
        self.episode_buffer = self.episode_buffer_fn(nS, self.gamma, self.tau,
                                                    self.n_workers, 
                                                    self.max_buffer_episodes,
                                                    self.max_buffer_episode_steps,
                                                    self.device)

        self.episode_reward = []
        episode = 0
        
        while True:
            episode_reward = self.episode_buffer.fill(envs, self.policy_model, self.value_model)
            n_ep_batch = len(episode_reward)
            self.episode_reward.extend(episode_reward.tolist())
            self.optimize_model()
            self.episode_buffer.clear()
                        
            # stats
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            episode += n_ep_batch

            if (episode + 1) % (max_episodes // 10) == 0:
                print(f"Episode {episode + 1}: "
                        f"mean_10_reward: {mean_10_reward:.2f}, "
                        f"std_10_reward: {std_10_reward:.2f}, "
                        f"mean_100_reward: {mean_100_reward:.2f}, "
                        f"std_100_reward: {std_100_reward:.2f})")
        
                torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'ppo_model_{env_name}_{episode + 1}.pth'))
                
            if episode >= max_episodes:
                break

        torch.save(self.policy_model.state_dict(), os.path.join(self.checkpoint_dir, f'ppo_model_{env_name}_final.pth'))


    def evaluate(self, env_name, env_render_mode, n_episodes=1, greedy=True):
        rs = []
        env = gym.make(env_name, render_mode=env_render_mode)
        model_state_dict = torch.load(os.path.join(self.checkpoint_dir, f'ppo_model_{env_name}_final.pth'))

        nS, nA = env.observation_space.shape, env.action_space.n
        self.policy_model = self.policy_model_fn(nS, nA, self.device)
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
    policy_model_fn = lambda ns, na, device: FCCA(ns, na, device, hidden_dims=(128, 64))
    policy_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    value_model_fn = lambda ns, device: FCV(ns, device, hidden_dims=(128, 64))
    value_optimizer_fn = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    make_envs_fn = lambda env_name, env_render_mode, seed, n_workers: MultiprocessEnv(env_name, env_render_mode, seed, n_workers)

    episode_buffer_fn = lambda state_dim, gamma, tau, n_workers, max_episodes, max_episode_steps, device: EpisodeBuffer(
        state_dim=state_dim,
        gamma=gamma,
        tau=tau,
        n_workers=n_workers,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        device=device
    )

    env_name = 'LunarLander-v3'
    seed = 42 
    gamma = 0.99
    n_workers = 8
    tau = 0.97
    max_buffer_episodes = 16
    max_buffer_episode_steps = 1000
    entropy_loss_weight = 0.01
    
    policy_optimization_epochs = 80
    policy_sample_ratio = 0.8
    policy_clip_range = 0.1
    policy_stopping_kl = 0.02
    policy_model_max_grad_norm = float('inf')
    
    value_optimization_epochs = 80
    value_sample_ratio = 0.8
    value_clip_range = float('inf')
    value_stopping_mse = 25
    value_model_max_grad_norm = float('inf')
    
    agent = PPO(
        policy_model_fn=policy_model_fn,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
        value_model_fn=value_model_fn,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        make_envs_fn=make_envs_fn
    )
    
    agent.train(
        env_name=env_name,
        env_render_mode=None,
        seed=seed,
        gamma=gamma,
        max_episodes=5_000,
        n_workers=n_workers,
        tau=tau,
        episode_buffer_fn=episode_buffer_fn,
        max_buffer_episodes=max_buffer_episodes,
        max_buffer_episode_steps=max_buffer_episode_steps,
        policy_optimization_epochs=policy_optimization_epochs,
        policy_sample_ratio=policy_sample_ratio,
        policy_clip_range=policy_clip_range,
        policy_stopping_kl=policy_stopping_kl,
        value_optimization_epochs=value_optimization_epochs,
        value_sample_ratio=value_sample_ratio,
        value_clip_range=value_clip_range,
        value_stopping_mse=value_stopping_mse,
        policy_model_max_grad_norm=policy_model_max_grad_norm,
        value_model_max_grad_norm=value_model_max_grad_norm,
        entropy_loss_weight=entropy_loss_weight
    )
    
    mean_score, std_score = agent.evaluate(
        env_name=env_name,
        env_render_mode="human",
        n_episodes=10,
        greedy=False
    )
    print(f"mean_score: {mean_score}, std_score: {std_score}")