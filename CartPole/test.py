import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from train import log_path, save_path


if __name__ == '__main__':
    environment_name = 'CartPole-v1'
    render_mode = None # "human"
    env = gym.make(environment_name, render_mode=render_mode)
    
    env = DummyVecEnv([lambda: env])
    model = PPO.load(save_path, env)
    # model = DQN.load(save_path, env)
    
    # Model evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True if render_mode == 'human' else False)
    print(f'{mean_reward = }; {std_reward = }')
    
    # Core metrics: average reward, average episode length

    # Training strategies: longer training, hyperparameter tuning, other algorithms

    # "Manual" model evaluation
    episodes = 5
    mean_score = 0
    for episode in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            if render_mode == "human": env.render() 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        
        mean_score += score
        print(f'{episode = }; {score[0] = }')
    
    mean_score /= episodes
    print(f'mean_score = {mean_score[0] / episodes:.2f}')
    
    
    env.close()