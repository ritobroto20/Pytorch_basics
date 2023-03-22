import numpy as np
import torch

x= torch.tensor([1,2,3,4,5],dtype=torch.float)
y= torch.tensor([2,5,6,8,11],dtype=torch.float)
w= torch.tensor(0.0,dtype=torch.float,requires_grad=True)

def forward(w,x):
    return w*x

def loss_func(y,y_pred):
    return ((y-y_pred)**2).mean()

learning_rate= 0.01
n_iterations= 50

for epach in range(n_iterations):
    y_pred= forward(w,x)
    loss= loss_func(y,y_pred)

    # Gradient = Backward pass
    loss.backward()
    with torch.no_grad():
        w-=w.grad*learning_rate

    w.grad.zero_()       # initializes w_grad to zero

    if epach%4==0:
        print(f"Loss (iterations {epach})= {loss:.4f}")

x_pred=7
print(f"\nw= {w:.4f}")
print(f"Prediction: x={x_pred}, y={w*x_pred:.4f}")