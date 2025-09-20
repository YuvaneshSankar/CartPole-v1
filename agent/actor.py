import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        #For a long time I wanted to mention about these things for a backward pass
        # So the zero grad ..... so when we do a backward pass the gradients we calculate are added to the new grads for the next backward pass to remove that we use this zero_grad function
        self.optimizer.zero_grad()
        # This calculates the gradients for the loss w.r.t the parameters
        loss.backward()
        # This updates the parameters using the gradients calculated by the backward function
        self.optimizer.step()
