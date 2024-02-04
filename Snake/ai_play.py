from agent import Agent
from snake_game_ai import SnakeGameAI
import torch
import time


if __name__ == '__main__':
    record = 0
    
    agent = Agent()    
    agent.model.load_state_dict(torch.load('./model/model_71.pth'))
    agent.no_games = 100    # In order to make the epsilon value 0, so that no exploration is being made, the model is already trained
    
    game = SnakeGameAI()

    while True:
        # get current state
        state = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state)
        
        # perform the move and get new state
        _, done, score = game.play_step(action=final_move)
        
        if done:
            game.reset()
            agent.no_games += 1
            
            if score > record:
                record = score
                
            print(f"Game: {agent.no_games - 100}, Score: {score}, Record: {record}")
            time.sleep(1)
    