import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import json
import numpy as np



if __name__ == '__main__':
    env_names = ['MsPacman-v4', 'Breakout-v4', 'SpaceInvaders-v4']
    difficulty = 0
    
    if not os.path.exists(os.path.join('Evaluations', f'evaluations.json')):
        all_evaluations = []
    else:
        f = open(os.path.join('Evaluations', f'evaluations.json'))
        all_evaluations = json.load(f)
    
    for env_name in env_names:
        model_path = './Models'
                
        for model_name in os.listdir(model_path):
            if env_name in model_name:
                env = make_atari_env(env_name, n_envs=1, seed=42, env_kwargs={"difficulty": difficulty, "obs_type": 'rgb'})
                env = VecFrameStack(env, n_stack=1)
                print(model_name)
                evaluations = {
                    "model": model_name,
                }

                model = DQN.load(os.path.join(model_path, model_name), env)
                
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000, render=False)
                
                evaluations["test_mean_score"] = mean_reward
                evaluations["test_std_score"] = std_reward
                
                all_evaluations.append(evaluations)
                
                env.close()                
                
    print(all_evaluations)
    json_obj = json.dumps(all_evaluations, indent=4)

    with open(os.path.join('Evaluations', f'evaluations.json'), 'w') as f:
        f.write(json_obj)    

    print("Completed")