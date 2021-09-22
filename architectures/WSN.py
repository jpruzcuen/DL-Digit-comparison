import torch
from torch import nn
from torch.nn import functional as F

# ==== Joint Network with Weight Sharing (WSN) ==== #

class WSN(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super().__init__()

        # Digit Classification
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 49, kernel_size=2)
        self.fc1 = nn.Linear(196, 100)
        self.fc2 = nn.Linear(100, 10)

        # Digit comparison  
        self.fc3 = nn.Linear(20,1)

        # Extra features: batch normalization and dropout
        self.batch_norm = batch_norm
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.batch_norm2 = nn.BatchNorm1d(10) 
         
        self.dropout = dropout
        self.drop = nn.Dropout(0.5)


    def forward(self, inp):

        N = inp.shape[0]
        out = torch.zeros(N,20)

        # Digit classification 
        for ii in range(inp.shape[1]): 
            x = inp[:,ii,:,:].unsqueeze_(1)  # Extract digit, size Nx1x14x14
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
            x = self.fc1(x.view(-1, 196))
            if self.dropout: x = self.drop(x)
            if self.batch_norm: x = self.batch_norm1(x)
            x = F.relu(x)                 
            x = self.fc2(x) #Nx10
            if self.dropout: x = self.drop(x)
            if self.batch_norm: x = self.batch_norm2(x)

            # out is Nx20, first 10 cols have digit 1 
            # and last 10 cols have digit 2
            out[:,torch.arange(10)+ii*10] = x    

        # Comparison 
        x = self.fc3(out.view(-1,20))
        if self.dropout: x = self.drop(x)
        x = F.relu(x) #Nx1
        x = torch.sigmoid(x) # Bound output between [0,1] for BCELoss
        
        return out,x