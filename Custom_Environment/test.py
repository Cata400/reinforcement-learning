import os
from stable_baselines3 import PPO
from environment import ShowerEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


if __name__ == '__main__':
    model_path = os.path.join('Models', 'custom_env')
    
    env = ShowerEnv()
    env = DummyVecEnv([lambda: env])
    
    model = PPO.load(model_path, env)
    
    # Model evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f'{mean_reward = }; {std_reward = }')