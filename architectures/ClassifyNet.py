
import torch
from torch import nn
from torch.nn import functional as F

# ==== Split Network: ClassifyNet ==== #

class ClassifyNet(nn.Module):
    def __init__(self, dropout = False, batch_normalization = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 49, kernel_size=2)

        self.fc1 = nn.Linear(196, 100)
        self.fc2 = nn.Linear(100, 10) #end of classification

        self.bn1 = nn.BatchNorm2d(32) # C from an expected input of size (N, C, H, W)
        self.bn2 = nn.BatchNorm2d(49)

        self.drop1 = nn.Dropout(0.5) # Have dropout ratio p = 0.5

        self.dropout = dropout 
        self.batch_normalization = batch_normalization


    def forward(self, input):

        N = input.shape[0]

        out = torch.zeros(2*N,10)
        reshaped_out = torch.zeros(N,20)

        for ii in range(input.shape[1]): # Loop over the channels
            x = input[:,ii,:,:].unsqueeze_(1) # Nx1x14x14
            x = self.conv1(x) 

            if self.batch_normalization: x = self.bn1(x)
            x = F.relu(F.max_pool2d(x, kernel_size=2))
            x = self.conv2(x)

            if self.batch_normalization: x = self.bn2(x)
            x = F.relu(F.max_pool2d(x, kernel_size=2))
            x = x.view(-1, 196) # Get to correct size

            if self.dropout: x = self.drop1(x)
            x = F.relu(self.fc1(x)) #Nx10
            
            if self.dropout: x = self.drop1(x)
            x = F.relu(self.fc2(x))

            out[N*ii:N*(ii+1),:] = x #output is 2Nx10
        
        reshaped_out[:, 0:10] = out[0:N,:]
        reshaped_out[:,10:20] = out[N:2*N,:] # Nx20
        
        return (out,reshaped_out)

