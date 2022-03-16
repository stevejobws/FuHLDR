import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()  
        self.gc1 = GraphConvolution(nfeat, nhid)   
        self.gc2 = GraphConvolution(nhid, nclass)  
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))  
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))    
        x1 = F.dropout(x, self.dropout, training = self.training)  
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training = self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x2, dim = 1), x1   