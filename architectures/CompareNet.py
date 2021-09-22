
import torch
from torch import nn
from torch.nn import functional as F

# ==== Split Network: CompareNet ==== #

class CompareNet(nn.Module):
    def __init__(self, dropout = False, batch_normalization = False): 
        super().__init__()
        self.fc3 = nn.Linear(20,2)    #operation for comparison

        self.bn1 = nn.BatchNorm1d(2) # C from an expected input of size (N, C)
        

        self.drop1 = nn.Dropout(0.5) # Have dropout ratio p = 0.5

        self.dropout = dropout 
        self.batch_normalization = batch_normalization


    def forward(self, x):

        x = self.fc3(x) # Nx2 
        if self.batch_normalization: x = self.bn1(x) #norm layer
        if self.dropout: x = self.drop1(x)
        x = F.relu(x)

        return x