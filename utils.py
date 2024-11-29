import matplotlib.pyplot as plt 
import numpy as np

import torch

from torch_geometric.data import Data


def normalize_A(A):
    """
    Normalize Adjacency Matrix
    """
     # Compute degree matrix
    D = torch.zeros_like(A)
    D[range(D.shape[0]), range(D.shape[1])] = A.sum(dim=1, keepdim=False)
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt = D_inv_sqrt.masked_fill(D_inv_sqrt == float("inf"), 0.)
    # Compute A_norm
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def FeatureSmoothnessLoss():
    def feature_smoothness_loss(x, A):
        """
        Compute feature smoothness loss
        """
        A_norm = normalize_A(A)
        # Compute normalized laplacian
        L = torch.eye(A_norm.shape[0]).to(A_norm.device) - A_norm
        # Feature smoothness loss
        loss = torch.trace(x.T @ L @ x)
        return loss
    return feature_smoothness_loss


def SparseLoss(a):
    """
    Regularization term, similar to the one presented in 
    https://arxiv.org/pdf/2106.05303.pdf
    
    - a: float value indicating the percentage of sparseness, i.e. a=0.2 will
    induce to learn an adjacency matrix A with 20% of values close to one while
    the rest is pulled to zero.
    """
    def sparse_reg_fn(A):
        # A has shape (N, N), where N is the number of nodes
        A = A.view(-1) # flatten
        A, _ = torch.sort(A) # sort in ascending order
        # Build a vector with the first (1-p)% values set to zero and the rest
        # set to one
        r = torch.zeros_like(A)
        r[-int(r.numel()*a):] = 1.
        # Compute the L2 norm between flattened A and r
        return torch.norm(A-r)
    return sparse_reg_fn


def plot_adjacency_matrix(A):
    threshold = 0.1
    A = A.cpu()
    fig, ax = plt.subplots()
    ax.imshow(A, cmap="Blues")
    p = ((A < threshold).sum() / A.nelement()).item() * 100
    ax.set_title(f"{p:.2f}% of values lower than threshold({threshold})")
    return fig


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def TerroristDataset():
    labels_to_i = {}

    with open("data/Terrorist/terrorist_attack.labels") as f:
        for i, line in enumerate(f.readlines()):
            labels_to_i[line.rstrip('\n')] = i

    name_to_i = {}
    x = []
    y = []

    with open("data/Terrorist/terrorist_attack.nodes") as f:
        for i, line in enumerate(f):
            line = line.split("\t")
            name_to_i[line[0]] = i
            x.append(torch.tensor([float(x) for x in line[1:-1]], dtype=torch.float32))
            y.append(labels_to_i[line[-1].rstrip("\n")])
    x = torch.stack(x, dim=0)
    y = torch.tensor(y)

    edge_index = []

    with open("data/Terrorist/terrorist_attack_loc.edges") as f:
        for line in f:
            a, b = line.rstrip("\n").split(" ")
            a = name_to_i[a]
            b = name_to_i[b]
            edge_index.append(torch.tensor([a, b]))
            edge_index.append(torch.tensor([b, a]))
    edge_index = torch.stack(edge_index, dim=1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data