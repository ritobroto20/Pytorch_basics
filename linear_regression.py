# We make a linear regression model using tensors of PyTorch

import torch
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)


# matmul(tutorial): https://www.geeksforgeeks.org/python-matrix-multiplication-using-pytorch/
x=torch.randn((3, 300),requires_grad=True)
bta=torch.tensor([3.0,1.0,5.0],requires_grad=True)
y = torch.matmul(bta,x)+3+2*torch.from_numpy(np.random.normal(size=300))



# no_grad:  https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
whgt=torch.tensor([0.0001,0.0001,0.001],requires_grad=True)
alpha=0.002

plt.figure()
plt.hist(y.detach().numpy(),bins=30,edgecolor='k',alpha=0.7,label='Y')
plt.hist((torch.matmul(bta,x)).detach().numpy()+3,bins=30,edgecolor='k',alpha=0.7,label='f(X)= y_predicted: where f is linear')
plt.text(x=-14,y=20,s=f'mean f(X)= {torch.matmul(bta,x).detach().numpy().mean()}\nstd_dev= {torch.matmul(bta,x).detach().numpy().std()}')
plt.legend()
plt.show()

for epoch in range(1000):
    y_pred = torch.matmul(whgt, x) + 3
    loss = torch.mean(torch.square((y - y_pred)))

    loss.backward(retain_graph=True)
    if epoch%100==0:
        print(f"epoch:{epoch}--\t{whgt}")
    with torch.no_grad():
        whgt-= alpha*whgt.grad
    whgt.grad.zero_()


print("\n",whgt)
print(bta)

y_pred=torch.matmul(whgt, x) + 3
error_numpy= (y_pred-y).detach().numpy()
plt.figure()
plt.hist(error_numpy,bins=30,edgecolor='k',label='Y',color='salmon')
plt.title("Expected: Normal distribution with std_dev=2, mean=0")
plt.text(x=-6,y=19,s=f'mean={error_numpy.mean()};\nstd_dev={error_numpy.std()}')
plt.show()

print("!!From this model the standard deviation of the error has reduced from 6 to 2!!!")