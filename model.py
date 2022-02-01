import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, W, Q, b):
        a1 = torch.mm(W,x)
        h = self.relu(a1)
        a2 = torch.mm(torch.t(Q),h)
        a3 = a2 + b
        y = self.sigmoid(a3)
        return h,y

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = Model().to(device)
    print(model)

    x = torch.tensor([[1.0],
                    [0.],
                    [1.0]], requires_grad=True)
    W = torch.tensor([[1,-1,0.5],
                [-1,1,0.5]], requires_grad=True)
    Q = torch.tensor([[1.0],
                    [1.0]], requires_grad=True)
    b = torch.tensor([[-1.5]], requires_grad=True)
    
    h_out, y_out = model(x,W,Q,b)
    print(h_out, y_out)

    lossfunc = nn.BCELoss()
    loss = lossfunc(y_out, torch.tensor([[0.]], requires_grad=False))
    print(f'loss: {loss.item()}')
    loss.backward()
    print(f'grad W:{W.grad}')
    print(f'grad Q:{Q.grad}')
    print(f'grad b:{b.grad}')

    torch.onnx.export(
    model,
    (x,W,Q,b),
    "answer.onnx",
    verbose=True,
    input_names=['x','W','Q','b'],
    output_names=['h', 'y_out']
    )