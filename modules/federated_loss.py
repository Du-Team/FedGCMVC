import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewSwAVLoss(nn.Module):
    def __init__(self, feature_dim, class_num, temperature=0.1, epsilon=0.05, sinkhorn_iterations=3):
        super(MultiViewSwAVLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        self.prototypes = nn.Parameter(torch.randn(class_num, feature_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def forward(self, z_list):
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

        qs = []
        scores_list = []

        for z in z_list:
            z = F.normalize(z, dim=1)
            scores = torch.mm(z, self.prototypes.t())
            scores_list.append(scores)
            with torch.no_grad():
                q = self.sinkhorn(scores)
                qs.append(q)

        total_loss = 0
        num_views = len(z_list)

        for i in range(num_views):
            for j in range(num_views):
                if i == j:
                    continue
                log_p_i = F.log_softmax(scores_list[i] / self.temperature, dim=1)
                loss_ij = -torch.mean(torch.sum(qs[j] * log_p_i, dim=1))
                total_loss += loss_ij

        return total_loss / (num_views * (num_views - 1))

    def sinkhorn(self, out):
        Q = out / self.epsilon
        Q = Q - Q.max()
        Q = torch.exp(Q).t()
        B = Q.shape[1]
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for _ in range(self.sinkhorn_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()