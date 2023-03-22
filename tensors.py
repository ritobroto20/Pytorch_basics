import torch
import numpy as np

x= torch.empty(3,3, dtype=torch.float64)
y= torch.empty(3,3)
print(torch.add(x,y))
print(x+y,"\n")

a1= torch.rand(2,2)
b1= torch.rand(2,2)
print(torch.add(a1,b1))
print(a1.add_(b1),"\n")              # Does in place sddition, replaces the original variable with the sum

np_arr= a1.numpy()
print(np_arr)
