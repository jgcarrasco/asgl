import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import parametrize

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class AdjacencyProperties(nn.Module):
    def forward(self, X):
        # Symmetric
        X = X.triu() + X.triu(1).transpose(-1, -2)
        # No self-loops (at least for now)
        X = X.fill_diagonal_(0)
        # Clamp values 
        X = torch.clamp(X, 0, 1)
        return X
    

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x
    

class GraphLearning(nn.Module):
    def __init__(self, num_nodes, A=None):
        super().__init__()
        self.A = nn.Parameter(torch.rand(num_nodes, num_nodes))
        if A is not None:
            self.A = nn.Parameter(A)
    
    def forward(self):
        edge_index, edge_weight = dense_to_sparse(self.A)
        return edge_index, edge_weight


class ASGL(nn.Module):
    def __init__(self, num_features, num_nodes, num_classes, A=None):
        super().__init__()
        self.gcn = GCN(num_features, num_classes)
        self.gl = GraphLearning(num_nodes, A)
        parametrize.register_parametrization(self.gl, "A", AdjacencyProperties())

    def forward(self, data):
        x = data.x
        edge_index, edge_weight = self.gl()
        x = self.gcn(x, edge_index, edge_weight)
        return x