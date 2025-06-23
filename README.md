# reinforcement-learning
This repository contains code for various reinforcement learning algorithms and experiments. It is designed to be modular and extensible, allowing for easy addition of new algorithms and environments. 

Projects:
- Atari: This directory explores the usage of Deep Q-Networks (DQNs) to train agents on Atari games: Ms. Pacman, Breakout, and Space Invaders. The code compares fully connected (FC) models using RAM input with convolutional neural networks (CNNs) using raw image input. Experience replay, ϵ-greedy policies, and target network updates were used to enhance learning stability. FC models often performed better early on, but CNNs improved with longer training, especially in Ms. Pacman. Code is implemented in Pytorch and Stable Baselines3 for the neural netorks, and OpenAI Gym for the environments.

- CartPole: This directory implements a simple reinforcement learning agent using Q-learning to balance a pole on a cart. The agent learns to apply forces to the cart to keep the pole upright. The code is implemented in Stable Baselines3 and uses OpenAI Gym for the environment.

- Custom_Environment: This directory contains a custom reinforcement learning environment implemented in OpenAI Gym. 

- Self_Driving_Car: This directory implements a reinforcement learning agent to control a self-driving car in a simulated environment. The agent learns to navigate the car through various scenarios using Q-learning. The code is implemented in Stable Baselines3 and uses OpenAI Gym for the environment.

- Snake: This directory implements a reinforcement learning agent to play the classic Snake game. The agent learns to navigate the snake to collect food while avoiding collisions with itself and the walls. The model is implemented in Pytorch while the environment is built using Pygame. The agent uses Q-learning to learn optimal actions based on the current state of the game.
