from collections import namedtuple, deque
import random
import torch
import math
import matplotlib.pyplot as plt
from IPython import display


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=input_shape, out_features=512)
        self.layer2 = torch.nn.Linear(in_features=512, out_features=128)
        self.layer3 = torch.nn.Linear(in_features=128, out_features=64)
        self.layer4 = torch.nn.Linear(in_features=64, out_features=16)
        self.layer5 = torch.nn.Linear(in_features=16, out_features=n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        x = torch.nn.functional.relu(self.layer4(x))
        return self.layer5(x)
        

def test_env(env):     
    episodes = 5 
    for episode in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            env.render() 
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
        
        print(f'{episode = }; {score = }')

        
def select_action(env, state, policy_net, eps_start, eps_end, eps_decay, device, steps_done):    
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            actions = policy_net(state)
            return torch.argmax(actions, dim=1).view(1, 1), steps_done # return most probable action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done
    

def plot(scores, means):
    scores_t = torch.tensor(scores)

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(means)
    plt.ylim(ymin=0)
    plt.legend(["Scores", "Mean scores"])
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(means)-1, means[-1], str(means[-1]))
    plt.show(block=False)
    plt.pause(.1)
    
        
def model_optimization_step(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return
    
    # Covnert batch-array of transitions to transition of batch-arrays
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) #TODO: check how this looks like
    
    # Compute mask of non-final states and concatenate batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) using the policy net and get the scores for the respective actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all the next states
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(dim=1).values
        
    # Compute the expected Q values
    expected_state_action_values = reward_batch + gamma * next_state_values
    
    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
