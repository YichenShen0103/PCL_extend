import torch
import torch.nn as nn
import math

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.w_pos = 3.0
        self.w_truth = 5.0
        self.w_neg = 2.0

    def forward(self, out_1, out_2, q_truth, queries, proportion):
        n = self.batch_size
        device = self.device

        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        # ---------------------------
        # 强化版权重 mask（核心修改）
        # ---------------------------
        pos_mask = (
            (queries == 1).float() * self.w_pos
            + torch.eye(n, device=device)
        ).repeat(2, 2)
        pos_mask -= torch.eye(2 * n, device=device)

        weight_mask = (
            (queries == 1).float() * self.w_pos
            + (q_truth == 1).float() * proportion * self.w_truth
            + torch.eye(n, device=device)
        ).repeat(2, 2)
        weight_mask -= torch.eye(2 * n, device=device)

        # 不能-link 的负样本权重
        neg_weight = (
            1.0
            + (queries == -1).float() * self.w_neg
        ).repeat(2, 2)
        neg_weight -= torch.eye(2 * n, device=device)

        # ---------------------------

        neg = torch.div((sim_matrix * neg_weight).sum(dim=-1),
                        neg_weight.sum(dim=-1)) * (2 * n)

        pos_neg = torch.div(sim_matrix * pos_mask, neg.reshape(-1, 1))

        loss_i = (
            - torch.log(torch.where(pos_neg == 0,
                                    torch.ones_like(pos_neg),
                                    pos_neg)) * weight_mask
        ).sum(dim=-1)

        return loss_i.sum() / (weight_mask.sum() + 1e-10)

class ClusterLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.w_pos = 3.0
        self.w_truth = 5.0
        self.w_neg = 2.0

        self.mask = self.mask_correlated_clusters(class_num).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, q_truth, queries, proportion):
        eps = 1e-8

        # ---------- entropy loss ----------
        p_i = c_i.sum(0) / c_i.sum()
        p_j = c_j.sum(0) / c_j.sum()

        ne_loss = (
            math.log(p_i.size(0)) + (p_i * torch.log(p_i + eps)).sum() +
            math.log(p_j.size(0)) + (p_j * torch.log(p_j + eps)).sum()
        )

        # ---------- contrastive ----------
        n = self.batch_size
        device = self.device
        out = torch.cat([c_i, c_j], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        # ---------------------------
        # 强化版权重 mask（核心修改）
        # ---------------------------
        pos_mask = (
            (queries == 1).float() * self.w_pos
            + torch.eye(n, device=device)
        ).repeat(2, 2)
        pos_mask -= torch.eye(2 * n, device=device)

        weight_mask = (
            (queries == 1).float() * self.w_pos
            + (q_truth == 1).float() * proportion * self.w_truth
            + torch.eye(n, device=device)
        ).repeat(2, 2)
        weight_mask -= torch.eye(2 * n, device=device)

        neg_weight = (
            1.0
            + (queries == -1).float() * self.w_neg
        ).repeat(2, 2)
        neg_weight -= torch.eye(2 * n, device=device)
        # ---------------------------

        neg = torch.div((sim_matrix * neg_weight).sum(dim=-1),
                        neg_weight.sum(dim=-1)) * (2 * n)
        pos_neg = torch.div(sim_matrix * pos_mask, neg.reshape(-1, 1))
        acc_loss = (
            - torch.log(torch.where(pos_neg == 0,
                                    torch.ones_like(pos_neg),
                                    pos_neg)) * weight_mask
        ).sum(dim=-1).sum() / (weight_mask.sum() + 1e-10)

        # ---------- cluster-level NCE ----------
        N = 2 * self.class_num
        c = torch.cat((c_i.t(), c_j.t()), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        positive_clusters = torch.cat(
            (torch.diag(sim, self.class_num), torch.diag(sim, -self.class_num)), dim=0
        ).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        labels = torch.zeros(N).long().to(device)
        loss = self.criterion(logits, labels) / N

        return loss + acc_loss + ne_loss, loss, acc_loss, ne_loss

# class InstanceLoss(nn.Module):
#     def __init__(self, batch_size, temperature, device):
#         super(InstanceLoss, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.device = device
#         self.mask = (torch.ones(2 * self.batch_size, device=self.device) - torch.eye(2 * self.batch_size, device=self.device)).bool()

#     def forward(self, out_1, out_2, q_truth, queries, proportion): #q_truth:查xun，que：推理
#         n = self.batch_size
#         out = torch.cat([out_1, out_2], dim=0)
#         sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
#         device = self.device
#         pos_mask = (((queries == 1).to(device) + torch.eye(n, device=device)).repeat(2, 2) - torch.eye(2 * n, device=device)).to(device)
#         weight_mask = (((queries == 1).to(device) + (q_truth == 1).to(device) * proportion + torch.eye(n, device=device)).repeat(2, 2) - torch.eye(2 * n, device=device)).to(device)
#         neg_weight = (torch.ones(2 * n, device=device) - torch.eye(2 * n, device=device) + (queries == -1).to(device).repeat(2, 2).float()).to(device)
#         neg = torch.div((sim_matrix * neg_weight).sum(dim=-1), neg_weight.sum(dim=-1)) * (2 * n)
#         pos_neg = torch.div(sim_matrix*pos_mask, neg.reshape(-1,1))
#         loss_i = (- torch.log(torch.where(pos_neg == 0, torch.ones_like(pos_neg), pos_neg)) * weight_mask).sum(dim=-1)
#         return loss_i.sum() / (weight_mask.sum() + torch.tensor(1e-10, device=device))

# class ClusterLoss(nn.Module):
#     def __init__(self, batch_size, class_num, temperature, device):
#         super(ClusterLoss, self).__init__()
#         self.batch_size = batch_size
#         self.class_num = class_num
#         self.temperature = temperature
#         self.device = device
#         self.mask = self.mask_correlated_clusters(class_num).to(self.device)
#         self.amask = (torch.ones(2 * self.batch_size, device=self.device) - torch.eye(2 * self.batch_size, device=self.device)).bool()
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)

#     def mask_correlated_clusters(self, class_num):
#         N = 2 * class_num
#         mask = torch.ones((N, N), device=self.device)
#         mask = mask.fill_diagonal_(0)
#         for i in range(class_num):
#             mask[i, class_num + i] = 0
#             mask[class_num + i, i] = 0
#         mask = mask.bool()
#         return mask
    
#     def forward(self, c_i, c_j, q_truth, queries, proportion):
#         eps=1e-8
#         p_i = (c_i).sum(0).view(-1)
#         p_i /= p_i.sum()
#         ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i+eps)).sum()
#         # ne_i =  (p_i * torch.log(p_i)).sum()
#         p_j = (c_j).sum(0).view(-1)
#         p_j /= p_j.sum()
#         ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j+eps)).sum()
#         # ne_j=(p_j * torch.log(p_j)).sum()
#         ne_loss = ne_i + ne_j
        
#         n = self.batch_size
#         out = torch.cat([c_i, c_j], dim=0)
#         device = self.device
#         sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
#         pos_mask = (((queries == 1).to(device) + torch.eye(n, device=device)).repeat(2, 2) - torch.eye(2 * n, device=device)).to(device)
#         weight_mask = (((queries == 1).to(device) + (q_truth == 1).to(device) * proportion + torch.eye(n, device=device)).repeat(2, 2) - torch.eye(2 * n, device=device)).to(device)
#         neg_weight = (torch.ones(2 * n, device=device) - torch.eye(2 * n, device=device) + (queries == -1).to(device).repeat(2, 2).float()).to(device)
#         neg = torch.div((sim_matrix * neg_weight).sum(dim=-1), neg_weight.sum(dim=-1)) * (2 * n)
#         pos_neg = torch.div(sim_matrix*pos_mask, neg.reshape(-1,1))
#         loss_i = (- torch.log(torch.where(pos_neg == 0, torch.ones_like(pos_neg), pos_neg)) * weight_mask).sum(dim=-1)
#         acc_loss = loss_i.sum() / (weight_mask.sum() + torch.tensor(1e-10, device=device))

#         N = 2 * self.class_num
#         c = torch.cat((c_i.t(), c_j.t()), dim=0)
        
#         sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
#         positive_clusters = torch.cat((torch.diag(sim, self.class_num), torch.diag(sim, -self.class_num)), dim=0).reshape(N, 1)
#         negative_clusters = sim[self.mask].reshape(N, -1)

#         labels = torch.zeros(N).to(positive_clusters.device).long()
#         logits = torch.cat((positive_clusters, negative_clusters), dim=1)
#         loss = self.criterion(logits, labels) / N

#         return loss + acc_loss + ne_loss, loss , acc_loss, ne_loss

