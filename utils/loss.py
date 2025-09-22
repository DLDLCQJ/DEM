import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# CORAL Loss
def Coral_loss(source, target, **kwargs):
    # Note that target_labels is unseen in this scenario
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss


# # Contrastive Loss Function
# def Contrastive_loss(features1, features2, labels1, labels2, margin=1.0):
#     euclidean_distance = nn.functional.pairwise_distance(features1, features2)
#     labels_match = (labels1 == labels2).float()
#     loss_contrastive = (labels_match * euclidean_distance.pow(2) +
#                         (1 - labels_match) * nn.functional.relu(margin - euclidean_distance).pow(2)).mean()
#     return loss_contrastive

def Self_Contrastive_loss(z1, z2, labels1, labels2, margin=1.0):
    B, C = z1.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    distance_matrix = torch.cdist(z1, z2)  # 2B x 2B
    #euclidean_distance = nn.functional.pairwise_distance(z1, z2)
    # row_ind, col_ind = linear_sum_assignment(distance_matrix.cpu().detach().numpy())
    # labels2 = torch.zeros(len(z2), dtype=torch.float32, device=z1.device)
    # for i in range(len(col_ind)):
    #     labels2[col_ind[i]] = labels1[row_ind[i]]
    # Create a mask to exclude diagonal elements and lower triangular part
    #mask = torch.triu(torch.ones_like(euclidean_distance,dtype=torch.bool), diagonal=1).to(z.device)
    #mask = torch.eye(labels1.shape[0], dtype=torch.bool).to(z1.device)
    # Calculate the contrastive loss
    labels_expanded = (labels1.unsqueeze(0) == labels2.unsqueeze(1)).float()
    positive_pairs = labels_expanded  * distance_matrix.pow(2)
    negative_pairs = (1 - labels_expanded) * F.relu(margin - distance_matrix).pow(2)
    # Combine the positive and negative losses
    loss = (positive_pairs + negative_pairs).sum() / len(labels1)

    return loss

def SelfAttn_Contrastive_loss(z1, z2, labels, margin=1.0):
    B, C = z1.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Compute pairwise Euclidean distances
    scales = torch.matmul(z1, z2.T)/ np.sqrt(z1.size(-1))
    weights = F.softmax(scales ,dim=-1)
    attns = torch.matmul(weights, z2)
    euclidean_distance = torch.cdist(attns, attns)
    #euclidean_distance = torch.cdist(z, z)  # 2B x 2B
    # Create a mask to exclude diagonal elements and lower triangular part
    #mask = torch.triu(torch.ones_like(euclidean_distance,dtype=torch.bool), diagonal=1).to(z.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
    # Calculate the contrastive loss
    labels_expanded = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Positive pairs (similar)
    positive_pairs = labels_expanded * mask * euclidean_distance.pow(2)
    # Negative pairs (dissimilar)
    negative_pairs = (1 - labels_expanded) * mask * F.relu(margin - euclidean_distance).pow(2)
    # Combine the positive and negative losses
    loss = (positive_pairs + negative_pairs).sum() / mask.sum()

    return loss

def Cross_Contrastive_loss(z1, z2, label1, label2, margin=1.0):
    B, C = z1.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    # Concatenate the embeddings and labels
    z = torch.cat([z1, z2], dim=0)  # 2B x C
    labels = torch.cat([label1, label2], dim=0)  # 2B

    # Compute pairwise Euclidean distances
    euclidean_distance = torch.matmul(z, z.T)
    #euclidean_distance = torch.cdist(z, z)  # 2B x 2B
    # Create a mask to exclude diagonal elements and lower triangular part
    #mask = torch.triu(torch.ones_like(euclidean_distance,dtype=torch.bool), diagonal=1).to(z.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device) # diagonal value is larger than other positions
    # Calculate the contrastive loss
    labels_expanded = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Positive pairs (similar)
    positive_pairs = labels_expanded * mask * euclidean_distance.pow(2)
    # Negative pairs (dissimilar)
    negative_pairs = (1 - labels_expanded) * mask * F.relu(margin - euclidean_distance).pow(2)
    # Combine the positive and negative losses
    loss = (positive_pairs + negative_pairs).sum() / mask.sum()

    return loss

def ContrPseudo_Contrastive_loss(z1, z2, label1, label2, margin=1.0):
    B, C = z1.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    # Concatenate the embeddings and labels
    z = torch.cat([z1, z2], dim=0)  # 2B x C
    labels = torch.cat([label1, label2], dim=0)  # 2B

    # Compute pairwise Euclidean distances
    euclidean_distance = torch.matmul(z, z.T)
    #euclidean_distance = torch.cdist(z, z)  # 2B x 2B
    # Create a mask to exclude diagonal elements and lower triangular part
    #mask = torch.triu(torch.ones_like(euclidean_distance,dtype=torch.bool), diagonal=1).to(z.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    # Calculate the contrastive loss
    labels_expanded = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Positive pairs (similar)
    positive_pairs = labels_expanded * mask * euclidean_distance.pow(2)
    # Negative pairs (dissimilar)
    negative_pairs = (1 - labels_expanded) * mask * F.relu(margin - euclidean_distance).pow(2)
    # Combine the positive and negative losses
    loss = (positive_pairs + negative_pairs).sum() / mask.sum()

    return loss


class MaximumSquareLoss(nn.Module):
    def __init__(self):
        super(MaximumSquareLoss, self).__init__()
    def forward(self, x):
        p = F.softmax(x, dim=1)
        b = (torch.mul(p, p))
        b = -1.0 *  b.sum(dim=1).mean() / 2
        return b
    
class CLSEntropyLoss(nn.Module):
    def __init__(self):
        super(CLSEntropyLoss, self).__init__()

    def forward(self, x):
        # Ensure x has a batch dimension if it's a single example
        if x.dim() == 1:
            x = x.unsqueeze(0)
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))  # Adding a small epsilon to avoid log(0)
        return entropy
    
class ClustersEntropyLoss(nn.Module):
    def __init__(self, entropy_loss_weight=1.0):
        super(ClustersEntropyLoss, self).__init__()
        self.entropy_loss_weight = entropy_loss_weight

    def forward(self, cluster_probabilities):
        num_clusters = float(cluster_probabilities.shape[1])
        target_entropy = torch.log(torch.tensor(num_clusters))
        cluster_probabilities = torch.mean(cluster_probabilities, dim=0)
        cluster_probabilities = torch.clamp(cluster_probabilities, min=1e-8, max=1.0)
        entropy = -torch.sum(cluster_probabilities * torch.log(cluster_probabilities))
        loss = target_entropy - entropy
        return self.entropy_loss_weight * loss

class ClustersConsistencyLoss(nn.Module):
    def __init__(self):
        super(ClustersConsistencyLoss, self).__init__()

    def forward(self, assigned_labels):
        target = torch.ones_like(assigned_labels)
        loss = F.binary_cross_entropy_with_logits(assigned_labels, target)
        return loss.mean()
    
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(TemporalConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, assigned_labels_current, assigned_labels_previous):
        # Normalize the assigned labels using softmax to get probability distributions
        assigned_labels_current = torch.sigmoid(assigned_labels_current / self.temperature, dim=-1)
        assigned_labels_previous = torch.sigmoid(assigned_labels_previous / self.temperature, dim=-1)

        # Compute the cosine similarity between current and previous cluster assignments
        cosine_similarity = F.cosine_similarity(assigned_labels_current, assigned_labels_previous, dim=-1)

        # The consistency loss encourages high similarity (close to 1)
        loss = 1 - cosine_similarity.mean()

        return loss


def discrepancy(out1, out2):
    # Ensure both outputs have the same batch size by truncating the larger batch
    if out1.size(0) != out2.size(0):
        smaller_size = min(out1.size(0), out2.size(0))
        out1 = out1[:smaller_size]
        out2 = out2[:smaller_size]
    
    # Use sigmoid to compute probabilities for binary classification
    p1 = torch.sigmoid(out1)
    p2 = torch.sigmoid(out2)

    # Calculate the mean absolute difference between the probabilities
    return torch.mean(torch.abs(p1 - p2))
