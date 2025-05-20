import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg

# ----------------------
# 1. Compute normalized Laplacian eigen-decomposition
def compute_laplacian_eigendecomp(edge_index, num_nodes, k=20):
    ew = torch.ones(edge_index.size(1))
    L_ei, L_ew = get_laplacian(edge_index, ew, normalization='sym', num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(L_ei, L_ew, num_nodes=num_nodes).tocsc()
    Lambda_k, Vk = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    return torch.from_numpy(Vk).float(), torch.from_numpy(Lambda_k).float()

# ----------------------
# 2a. Simple GCN
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, ei)
        return F.log_softmax(x, dim=1)

# ----------------------
# 2b. S2GNN Layer & Model
class S2GNNLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, Vk, Lambda_k):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.register_buffer('Vk', Vk)
        self.register_buffer('Lambda_k', Lambda_k)
        self.gain = torch.nn.Parameter(torch.ones(Lambda_k.size(0)))
        self.lin = torch.nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x, ei):
        x_sp = self.conv(x, ei)
        x_hat = self.Vk.T @ x                        # (k, in_ch)
        x_hat = x_hat * self.gain[:, None]
        x_spc = self.lin(self.Vk @ x_hat)            # (n, out_ch)
        return F.relu(x_sp + x_spc)

class S2GNN(torch.nn.Module):
    def __init__(self, in_ch, num_classes, Vk, Lambda_k):
        super().__init__()
        self.l1 = S2GNNLayer(in_ch, 16, Vk, Lambda_k)
        self.l2 = S2GNNLayer(16, num_classes, Vk, Lambda_k)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.l1(x, ei)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x, ei)
        return F.log_softmax(x, dim=1)

# ----------------------
# 3. Training & testing
def train(model, data, opt):
    model.train()
    opt.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()

@torch.no_grad()
def test_acc(model, data):
    model.eval()
    out = model(data).argmax(dim=1)
    correct = (out[data.test_mask] == data.y[data.test_mask]).sum().item()
    return correct / int(data.test_mask.sum())

# ----------------------
# 4. Run experiments for 3 seeds and both models
def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    Vk, Lambda_k = compute_laplacian_eigendecomp(data.edge_index, data.num_nodes, k=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GCN
    gcn = SimpleGCN(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(100):
        train(gcn, data, opt_gcn)
    acc_gcn = test_acc(gcn, data)

    # S2GNN
    s2 = S2GNN(dataset.num_features, dataset.num_classes, Vk.to(device), Lambda_k.to(device)).to(device)
    opt_s2 = torch.optim.Adam(s2.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(100):
        train(s2, data, opt_s2)
    acc_s2 = test_acc(s2, data)

    return acc_gcn, acc_s2

if __name__ == "__main__":
    seeds = [42, 52, 62]
    results_gcn = []
    results_s2 = []
    for s in seeds:
        gcn_acc, s2_acc = run(s)
        print(f"Seed {s}: GCN Acc = {gcn_acc:.4f}, S2GNN Acc = {s2_acc:.4f}")
        results_gcn.append(gcn_acc)
        results_s2.append(s2_acc)

    mean_gcn, std_gcn = np.mean(results_gcn), np.std(results_gcn)
    mean_s2, std_s2 = np.mean(results_s2), np.std(results_s2)
    print(f"\nGCN: {mean_gcn:.4f} ± {std_gcn:.4f}")
    print(f"S2GNN: {mean_s2:.4f} ± {std_s2:.4f}")
