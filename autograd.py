# https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
# https://pytorch.org/blog/overview-of-pytorch-autograd-engine/#:~:text=PyTorch%20computes%20the%20gradient%20of,ways%3B%20forward%20and%20reverse%20mode.

import numpy as np
import torch

x=torch.tensor([2.0],requires_grad= True)
y=torch.tensor([4.0],requires_grad= True)
z=torch.tensor([5.0],requires_grad= True)

w=x*y**2
a=torch.log(w)
t=a+x
t.backward()
print(y.grad)
print(x.grad)

# The basic implementation of Neural Networks using Pytorch framework is stored in this folder. Instead of numpy what are the advantages of using pytorch library??