import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


if __name__ == '__main__':
    env_name = 'MsPacman-v4'
    model_name = 'MsPacman-v4-policy-cnn-1_10M'
    difficulty = 0
    model_path = os.path.join('Models', model_name)

    env = make_atari_env(env_name, n_envs=1, seed=42, env_kwargs={"render_mode": "human", "difficulty": difficulty})
    env = VecFrameStack(env, n_stack=1)
    
    model = DQN.load(model_path, env)
    
    # Model evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
    print(f'{mean_reward = }; {std_reward = }')