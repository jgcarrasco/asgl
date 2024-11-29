import datetime

import comet_ml
from comet_ml import Experiment

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from models import ASGL
from utils import FeatureSmoothnessLoss, SparseLoss, plot_adjacency_matrix, EarlyStopper


#########################
#    HYPERPARAMETERS    #
#########################

n_repeats = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
log_comet = True
dataset = "CiteSeer" # ["Cora", "CiteSeer", "PubMed"]
gsl_method = "ASGL" # ["GLNN", "ASGL"]
fix_seed = False
initialize_A = False
learn_A = True
lambda_CE = 1.
lambda_L2 = 0.
lambda_fs = 0.1
lambda_sparse = 0.1
a = 0.5

if fix_seed:
    torch.manual_seed(42)

dataset = Planetoid(root="data", name=dataset, split="full")
data = dataset[0].to(device)
# transform = T.Compose([T.NormalizeFeatures()])
# data = transform(data)
A = to_dense_adj(data.edge_index).squeeze(dim=0)
num_features = dataset.num_node_features
num_nodes = data.num_nodes
num_classes = dataset.num_classes


########################
#    MODEL & LOSSES    #
########################

A_init = None

if not learn_A:
    A_init = torch.cdist(data.x, data.x)
    A_init = A_init.fill_diagonal_(1e6)
    i = int(a * A_init.nelement())
    values, _ = A_init.view(-1).sort()
    threshold = values[i]
    A_init[A_init < threshold] = 1.
    A_init[A_init >= threshold] = 0.
    if initialize_A:
        A_mask = (data.train_mask.float().unsqueeze(dim=1) @ data.train_mask.float().unsqueeze(dim=0)).bool()
        A_init[A_mask] = A[A_mask]

# plt.imshow(A_init.detach().cpu().numpy())
# plt.show()

loss_fn = nn.CrossEntropyLoss()
loss_A_fn = lambda A_pred: torch.norm(A_pred[data.train_mask] - A[data.train_mask])
loss_fs_fn = FeatureSmoothnessLoss()
loss_sparse_fn = SparseLoss(a=a) if gsl_method == "ASGL" else lambda A_pred: torch.norm(A_pred, 1)
acc_fn = lambda logits, y: ((logits.argmax(dim=-1) == y).sum() / y.shape[0]).item()

date = datetime.datetime.now()

for i in range(n_repeats):
    
    exp_name = f"{date:%Y_%m_%d_%H:%M:%S}_{i}"

    model = ASGL(num_features, num_nodes, num_classes, A=A_init).to(device)
    optimizer_gcn = torch.optim.Adam(model.gcn.parameters(), lr=1e-1, weight_decay=5e-4)
    optimizer_A = torch.optim.Adam(model.gl.parameters(), lr=1e-1)

    ##################
    #   CALLBACKS    #
    ##################
    early_stopping = EarlyStopper(patience=500, min_delta=0)
    max_acc = -1.

    if log_comet:
        experiment = Experiment(
            api_key="HV1APQhJ9aHguCCiLfvw3nY74",
            project_name="ASGL",
        )
        experiment.set_name(exp_name)
        experiment.log_parameters({
            "lambda_L2": lambda_L2,
            "lambda_fs": lambda_fs,
            "lambda_sparse": lambda_sparse,
            "lambda_CE": lambda_CE,
            "a": a
        })


    model.train()
    for epoch in range(1000):
        logits = model(data)

        # Optimize the GCN
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        optimizer_gcn.zero_grad()
        loss.backward()
        optimizer_gcn.step()

        acc = acc_fn(logits[data.train_mask], data.y[data.train_mask])

        # Optimize A
        if learn_A:
            logits = model(data)

            loss_CE = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            loss_L2 = loss_A_fn(model.gl.A)
            loss_fs = loss_fs_fn(data.x, model.gl.A)
            loss_sparse = loss_sparse_fn(model.gl.A)
            loss_A = lambda_CE*loss_CE + lambda_L2*loss_L2 + lambda_fs*loss_fs + lambda_sparse*loss_sparse

            optimizer_A.zero_grad()
            loss_A.backward()
            optimizer_A.step()
        
        #num_edges = model.gl.A.count_nonzero()
        num_edges = (model.gl.A == 1.).sum()
        num_zeros = (model.gl.A == 0.).sum()
        
        if log_comet:
            experiment.log_metric("num_edges", num_edges, epoch=epoch)
        
        if (epoch+1) % 10 == 0:
            print("Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}"\
                .format(epoch+1, loss.item(), acc))
            
        if log_comet:
            experiment.log_metrics({"loss_CE": loss_CE, "loss_fs": loss_fs, 
                                    "loss_sparse": loss_sparse}, step=epoch)
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(A[100:200,100:200].cpu().numpy(), vmin=0, vmax=1)
            ax1.set_title(f"{data.num_edges}, {data.num_edges / data.num_nodes**2*100:.2f}%")
            ax2.imshow(model.gl.A[100:200,100:200].detach().cpu().numpy(), vmin=0, vmax=1)
            ax2.set_title(f"{num_edges}, {num_edges / model.gl.A.nelement() * 100:.2f}%, {(1 - num_zeros/model.gl.A.nelement()) * 100:.2f}%")
            experiment.log_figure("A", fig, step=epoch)
            plt.close()
            # fig, ax = plt.subplots()
            # ax.hist(model.gl.A.view(-1).detach().cpu().numpy(), bins=100, edgecolor="black", range=(0, 1))
            # experiment.log_figure("A_hist", fig, step=epoch)
            # plt.close()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(data)
            val_loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_acc = acc_fn(logits[data.val_mask], data.y[data.val_mask])
        model.train()

        if log_comet:
            experiment.log_metrics({"loss": loss, "acc": acc, 
                                    "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
        
        if val_acc > max_acc:
            torch.save(model.state_dict(), f"checkpoints/{exp_name}.pth")
        # Callbacks
        if early_stopping.early_stop(val_loss):
            break


    print("Final Evaluation")
    model = ASGL(num_features, num_nodes, num_classes, A=A_init).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{exp_name}.pth"))
    model.eval()
    logits = model(data)
    test_loss = loss_fn(logits[data.test_mask], data.y[data.test_mask])
    test_acc = acc_fn(logits[data.test_mask], data.y[data.test_mask])
    print("Test loss: {:.4f} Test Acc: {:.4f}".format(test_loss, test_acc))
    if log_comet:
            experiment.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            experiment.end()

