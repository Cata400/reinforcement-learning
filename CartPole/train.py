import os
from environment import env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


log_path = os.path.join('Training', 'Logs')
save_path = os.path.join('Training', 'Models', 'model_cartpole')


if __name__ == '__main__':
    if not os.path.exists(log_path):
        os.makedirs(log_path) 
        
    env = DummyVecEnv([lambda: env])
    
    new_architecture = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]   # first is for the custom actor, second is for the value function
    
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': new_architecture})
    # model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(env,
                                callback_on_new_best=stop_callback,
                                eval_freq=10_000,
                                best_model_save_path=save_path + '_best',
                                verbose=1)
    
    model.learn(total_timesteps=20000, callback=eval_callback)
    model.save(save_path + '_last') 
    
    env.close()
    
    