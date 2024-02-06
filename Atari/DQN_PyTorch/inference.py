import gymnasium as gym
from itertools import count
import torch
import utils


if __name__ == '__main__':
    env_name = 'MsPacman-v4'
    render_mode = "human"
    difficulty = 0
    env = gym.make(env_name, render_mode=render_mode, difficulty=difficulty, obs_type='ram')
    # utils.test_env(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    state, info = env.reset()
        
    model = utils.DQN(input_shape=len(state), n_actions=n_actions).to(device)
        
    checkpoint = torch.load('./Models/MsPacman-v4-policy-fc-5.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    hyperparameters = checkpoint['hyperparameters']
    print(hyperparameters)
    
    num_episodes = 3    
    
    for i_episode in range(num_episodes):
        print(i_episode)
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
                print(f"Score: {episode_reward}")
                break
            
    env.close()

    print("Completed")