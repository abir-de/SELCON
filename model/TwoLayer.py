import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, 1)
        self.feature_dim = hidden_units
    
    def forward(self, x, last=False):
        l1scores = F.relu(self.linear1(x))
        scores = self.linear2(l1scores)
        if last:
            return scores.view(-1), l1scores
        else:
            return scores.view(-1)

    def get_embedding_dim(self):
        return self.feature_dim

"""
class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)#,bias=False)
        self.feature_dim = input_dim
    
    def forward(self, x, last=False):
        scores = self.linear(x)
        #scores = torch.sigmoid(self.linear(x))
        #return scores.view(-1)
        if last:
            return scores.view(-1), x
        else:
            return scores.view(-1)

    def get_embedding_dim(self):
        return self.feature_dim
"""