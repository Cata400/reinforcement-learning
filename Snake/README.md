This repo  serves as an introductory project to Reinforcement Learning and represents a personal rendition of the classic Snake game, for which I implemented an intelligent agent which has learned to play the game, obtaining a high score of 71 during training. The agent is based on a Deep Q-Learning algorithm, and it is implemented in Python using the PyTorch library. The game itself is implemented using the PyGame library.

The agent can be retrained using the `agent.py` script, which will save the trained model in the `./model/model.pth` file. The agent can be tested using the `ai_play.py` script, which will load the model from the saved file and will play the game using the learnt policy. In the `model` directory you will find the already trained model previously mentioned.