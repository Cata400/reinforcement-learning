from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import os
import gymnasium as gym
import time


def test_env(env):     
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
    env_name = 'SpaceInvaders-v4'
    difficulty = 0
        
    model_path = './Models'
    model_name = f'{env_name}-policy-cnn-1-10M'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    log_path = './Logs'
    log_name = f'{model_name}'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Checking environment
    # env = gym.make(env_name, render_mode="human", difficulty=difficulty)
    # test_env(env)
    
    env = make_atari_env(env_name, n_envs=1, seed=42, env_kwargs={"difficulty": difficulty, "obs_type": 'rgb'})
    env = VecFrameStack(env, n_stack=1)
    
    # Model training
    start = time.time()  
    model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_path, device='cuda', tau=0.005, batch_size=128)
    model.learn(total_timesteps=10_000_000, tb_log_name=log_name)
    model.save(os.path.join(model_path, model_name))
    print(f'Training time: {time.time() - start} s')