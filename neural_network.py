import torch
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)



x=torch.randn((3, 100),requires_grad=True)
bta=torch.tensor([3.0,1.0,5.0],requires_grad=True)
y = torch.matmul(bta,x)+3
# +2*np.random.normal(size=100)

whgt=torch.tensor([0.0001,0.0001,0.001],requires_grad=True)
y_pred=torch.matmul(whgt,x)+3
for epoch in range(1000):
    y_pred[0].backward(retain_graph=True)
    for i in range(len(bta)):
        print("Gradient: ",x.grad[i][0])
    x.grad.zero_()