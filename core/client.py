import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation else nn.Identity()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        support = torch.sparse.mm(adj, x)
        
        out = self.linear(support)
        
        out = self.bn(out)
        out = self.activation(out)
        return out

class Client(torch.nn.Module):
    def __init__(self, args, device="cpu", input_dim=None, full_data=None):
        super().__init__()
        self.device = device
        self.topk = getattr(args, 'topk', 15)

        self.gcn1 = GraphConv(input_dim, 256)
        
        rep_dim = 256

        self.instance_projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, args.feature_dim),
        )

        self.full_x = full_data.to(device).detach()
        
        self.global_adj = self.get_knn_graph(self.full_x)

        self.to(device)
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        self.cache = None

    def get_knn_graph(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        dist = torch.cdist(x_norm, x_norm, p=2)
        k = min(self.topk, x.shape[0])
        _, indices = torch.topk(dist, k=k, dim=1, largest=False)
        
        B = x.shape[0]
        adj = torch.zeros(B, B, device=x.device)
        
        adj.scatter_(1, indices, 1.0)
        
        adj = adj.cpu()
        adj = torch.max(adj, adj.t())
        
        adj.fill_diagonal_(1.0)
        
        D = adj.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.

        adj.mul_(D_inv_sqrt.unsqueeze(1))
        adj.mul_(D_inv_sqrt.unsqueeze(0))
        adj = adj.to_sparse()
        adj = adj.to(self.device)
        
        return adj

    def forward(self, full_idx):
        h = self.gcn1(self.full_x, self.global_adj)
        
        full_z = torch.nn.functional.normalize(self.instance_projector(h), dim=1)
        
        z = full_z[full_idx]
        
        z_out = z.detach().clone()
        if self.training:
            z_out.requires_grad_(True)
        
        self.cache = z
        return z_out

    def backward(self, grad_z):
        if self.cache is None: return
        z = self.cache
        if grad_z is not None: z.backward(grad_z)
        self.cache = None
