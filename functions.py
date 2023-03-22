# http://d2l.ai/chapter_multilayer-perceptrons/mlp.html

import torch
import numpy as np
import matplotlib.pyplot as plt

# ReLU function
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)


plt.figure()
plt.grid()
plt.title("ReLU function")
plt.plot(x.detach(), y.detach())
plt.show()
print(x.requires_grad,x.detach().requires_grad)

# Sigmoid Function
y= torch.sigmoid(x)
plt.figure()
plt.grid()
plt.title("Sigmoid function")
plt.plot(x.detach(), y.detach())
plt.show()

# Gradient of Sigmoid function
y.backward(torch.ones_like(x))
plt.grid()
plt.title("derivative of sigmoid function")
plt.plot(x.detach(), x.grad)
x.grad.zero_()
plt.show()