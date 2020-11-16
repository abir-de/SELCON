import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        #self.feature_dim = input_dim
    
    def forward(self, x, last=False):
        scores = self.linear(x)
        return scores.view(-1)
        '''if last:
            return scores, x
        else:
            return scores'''

    #def get_feature_dim(self):
    #    return self.feature_dim

class DualNet(nn.Module):
    def __init__(self, input_dim):
        super(DualNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1,bias=False)
    
    def forward(self, x):
        scores = self.linear(x)
        return scores.view(-1)