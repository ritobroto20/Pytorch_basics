# Here we construct a simple linear model using in-built pytorch library
# link- https://pytorch.org/docs/stable/nn.html
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

# super().__init() ---> makes the child class inherit the attributes of the parent class
# nn.Module: https://www.youtube.com/watch?v=YJ1wSxbqqo8&ab_channel=Dr.DataScience
# nn.sequential: https://www.youtube.com/watch?v=bH9Nkg7G8S0&ab_channel=deeplizard
"""
class Neural(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear1= nn.Linear(2,2, bias=None)
        self.linear1.weight = torch.nn.Parameter(torch.tensor([[0.11, 0.21], [0.12, 0.08]]))
"""

input=torch.randn(128,1)
lin_test=nn.Linear(1,1)
weights=torch.tensor([2.0],requires_grad=True)

# Manually assigning the value of weights to 2
print(weights.view(weights.size(0), -1))
weights=weights.view(weights.size(0), -1)  # Coverting shape of the weight
lin_test.weight=nn.parameter.Parameter(data=weights)
output=lin_test(input)

#lin_test.weight=nn.Parameter([1],requires_grad=True)
plt.figure()
plt.plot(input.detach().numpy(),output.detach().numpy(),'k.',label=f'Slope={lin_test.weight}')
plt.legend()
plt.show()




