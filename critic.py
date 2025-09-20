import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Input -> state
# Output -> one scalar value
class Actor(nn.Module):
    def __init__(self, input_dim , hidden_dim , output_dim,alpha=0.001):
        super(Actor,self).__init__()
        self.w1=nn.Linear(input_dim,hidden_dim)
        self.w2=nn.Linear(hidden_dim,hidden_dim)
        self.w3=nn.Linear(hidden_dim,output_dim)

        self.learning_rate=alpha
        self.optimizer=optim.Adam(self.parameters(),lr=alpha)

    def forward(self,state):
        x=F.relu(self.w1(state))
        x=F.relu(self.w2(x))
        x=F.softmax(self.w3(x),dim=-1)
        return x
    
    def backward(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
