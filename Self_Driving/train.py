import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def test_env(env):
    print(env.reset())
    print(env.action_space)
    
    episodes = 5 
    for episode in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            env.render() 
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
        
        print(f'{episode = }; {score = }')


if __name__ == '__main__':
    env_name = 'CarRacing-v2'
    render_mode = None # "human"
    
    log_path = './Logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    model_path = './Models'
    model_name = 'self_driving'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Checking environment
    env = gym.make(env_name, render_mode=render_mode)
    # test_env(env)
    env = DummyVecEnv([lambda: env])
    
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=100_000)
    model.save(os.path.join(model_path, model_name))    