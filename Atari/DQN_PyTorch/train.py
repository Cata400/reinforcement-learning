import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count
import torch
import utils

plt.ion()


if __name__ == '__main__':
    env_name = 'SpaceInvaders-v4'
    render_mode = None #"human"
    difficulty = 0
    env = gym.make(env_name, render_mode=render_mode, difficulty=difficulty, obs_type='ram')
    # utils.test_env(env)

    batch_size = 128
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    tau = 0.005
    lr = 1e-4
    num_episodes = 10_000
    
    model_name = env_name + '-policy-fc-1-10K'
        
    hyperparameters_dict = {
        'batch_size': batch_size,
        'gamma': gamma,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'tau': tau,
        'lr': lr,
        'num_episodes': num_episodes
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state, info = env.reset()
    
    policy_net = utils.DQN(input_shape=len(state), n_actions=n_actions).to(device)
    target_net = utils.DQN(input_shape=len(state), n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = utils.ReplayMemory(10000)
    
    steps_done = 0
    episode_scores = []
    episode_means = []
    max_score = 0
    max_mean = 0
    total_timesteps = 0
    
    for i_episode in range(num_episodes):
        print(i_episode)
        print(total_timesteps)
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        
        for t in count():
            total_timesteps += 1
            action, steps_done = utils.select_action(env, state, policy_net, eps_start, eps_end, eps_decay, device, steps_done)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            episode_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)
            
            # Move to next state
            state = next_state
            
            # Perform one step of optimization on the policy net
            utils.model_optimization_step(policy_net, target_net, memory, optimizer, batch_size, gamma, device)
            
            # Soft update of the target network weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
                
            target_net.load_state_dict(target_net_state_dict)
            
            
            if done:
                if episode_reward >= max_score:
                    max_score = episode_reward
                    hyperparameters_dict["max_score"] = max_score 
                    
                    torch.save({'model_state_dict': policy_net.state_dict(), 'hyperparameters': hyperparameters_dict}, f'./Models/{model_name}_best_score.pt')                    
                
                episode_scores.append(episode_reward)
                
                if i_episode < 100:
                    episode_means.append(0)
                else:
                    episode_means.append(sum(episode_scores[-100:]) / 100)
                    
                if episode_means[-1] >= max_mean:
                    max_mean = episode_means[-1]
                    hyperparameters_dict["max_mean"] = max_mean 

                    torch.save({'model_state_dict': policy_net.state_dict(), 'hyperparameters': hyperparameters_dict}, f'./Models/{model_name}_best_mean.pt')
                
                utils.plot(episode_scores, episode_means)
                torch.save({'model_state_dict': policy_net.state_dict(), 'hyperparameters': hyperparameters_dict}, f'./Models/{model_name}_last.pt')

                
                break
            
    
    print('Complete')
    plt.ioff()
    plt.savefig(f"./Figures/{model_name}.png")
            
            
            
    
    
    