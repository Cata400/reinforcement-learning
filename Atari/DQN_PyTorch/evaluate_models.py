import gymnasium as gym
from itertools import count
import torch
import utils
import json
import os
import numpy as np


if __name__ == '__main__':
    env_name = 'SpaceInvaders-v4'
    render_mode = None
    difficulty = 0
    env = gym.make(env_name, render_mode=render_mode, difficulty=difficulty, obs_type='ram')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    state, info = env.reset()
        
    model = utils.DQN(input_shape=len(state), n_actions=n_actions).to(device)
    models_path = './Models'
    model_type = 5
    
    if not os.path.exists(os.path.join('Evaluations', f'evaluations.json')):
        all_evaluations = []
    else:
        f = open(os.path.join('Evaluations', f'evaluations.json'))
        all_evaluations = json.load(f)
        
    for file in os.listdir(models_path):
        if file.endswith('.pt') and f'-{model_type}' in file and env_name in file:
            print(file)
            checkpoint = torch.load(f'./Models/{file}')
            model.load_state_dict(checkpoint['model_state_dict'])  
            hyperparameters = checkpoint['hyperparameters']
            
            evaluations = {
                'model': file,
                'train_max_score': hyperparameters["max_score"],
                'train_max_mean': hyperparameters["max_mean"],
                }
            
            num_episodes = 1000
            episode_rewards = []    
            
            for i_episode in range(num_episodes):
                state, info = env.reset()
                episode_reward = 0
                
                for t in count():
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = torch.argmax(model(state), dim=1).view(1, 1)
                    
                    state, reward, terminated, truncated, _ = env.step(action.item())
                    episode_reward += reward
                    done = terminated or truncated
                    
                    if done:
                        episode_rewards.append(episode_reward)
                        break
                    
            evaluations['test_max_score'] = np.max(episode_rewards)
            evaluations['test_mean_score'] = np.mean(episode_rewards)
            evaluations['test_std_score'] = np.std(episode_rewards)
            
            all_evaluations.append(evaluations)
            
        
    json_obj = json.dumps(all_evaluations, indent=4)
    
    with open(os.path.join('Evaluations', f'evaluations.json'), 'w') as f:
        f.write(json_obj)    
    
    print("Completed")