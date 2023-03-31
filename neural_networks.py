# For Neural Networks we use pre made classes made by PyTorch to run the model (instead of constructing everything by oneself, it is much easier)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

# Simple Neural Network model
class Neural(nn.Module):
    def __init__(self):
        super(Neural, self).__init__()
        self.linear1= nn.Linear(2,2, bias=None)
        self.linear1.weight = torch.nn.Parameter(torch.tensor([[0.11, 0.21], [0.12, 0.08]]))

    def forward(self,x):
        x=self.linear1(x)
        return x

# Defining models
input= torch.randn(100,2)
lin_model= nn.Linear(2,2)
nn_model= Neural()
lin_model.weight= nn.Parameter(torch.tensor([[1.0,2.0],[2.0,2.0]]))

y= lin_model(input)+torch.randn(100,2)*0.5
y_pred=nn_model.forward(input)
print(y_pred)

optimizer= torch.optim.Adam(nn_model.parameters(), lr=0.01)
criterion= torch.nn.MSELoss()
loss_arr= []

for epoch in range(500):

    y_pred = nn_model.forward(input)
    loss= criterion(y,y_pred)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss}")
    loss_np= loss.detach().numpy()
    loss_arr.append(loss_np)


#print(nn_model.forward(input))
#print(nn_model.state_dict())
print()
print(nn_model.state_dict())
print(lin_model.weight)

plt.figure()
plt.plot(np.arange(len(loss_arr)),loss_arr,'.',label=f"{nn_model.state_dict()}")
plt.legend()
plt.title("Loss function w/ iterations",weight='bold')
plt.show()