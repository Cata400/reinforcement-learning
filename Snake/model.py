import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
            
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):     
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # reshape to (1, x)
            state = state.view((1, -1))
            next_state = next_state.view((1, -1))
            action = action.view((1, -1))
            reward = reward.view((1, -1))
            done = (done, )
        
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        # 2: Q_new = R + y * max(next predicted Q value) -> only do this if not done
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new += self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx, torch.argmax(action).item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()