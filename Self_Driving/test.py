import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


if __name__ == '__main__':
    model_path = os.path.join('Models', 'self_driving')
    env_name = 'CarRacing-v2'
    render_mode = "human"
    
    env = gym.make(env_name, render_mode=render_mode)
    env = DummyVecEnv([lambda: env])
    
    model = PPO.load(model_path, env)
    
    # Model evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    print(f'{mean_reward = }; {std_reward = }')