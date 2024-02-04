import gym

# Loading environment
environment_name = 'CartPole-v1'
env = gym.make(environment_name)

if __name__ == '__main__':
    episodes = 5 
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            env.render() 
            action = env.action_space.sample()
            n_state, reward, done, truncated, info = env.step(action)
            score += reward
        
        print(f'{episode = }; {score = }')
    # env.close()

    # Understanding the environment
    print(env.action_space) # Move left, move right
    print(env.observation_space) # Cart position, cart velocity, pole angle, pole velocity