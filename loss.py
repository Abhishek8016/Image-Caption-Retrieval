import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletRankingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, img_emb, txt_emb):
        # Pairwise Cosine Similarity Matrix (batch_size x batch_size)
        scores = torch.matmul(img_emb, txt_emb.T)
        diagonal = scores.diag().view(-1, 1)
        
        # Pairwise Ranking Terms
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.T.expand_as(scores)
        
        cost_s = F.relu(self.margin - d1 + scores)   # Image-to-Text Negatives
        cost_im = F.relu(self.margin - d2 + scores)  # Text-to-Image Negatives
        
        # Zero out diagonal elements (Positive matched pairs)
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)
        
        # Average over batch size for mean reduction
        loss = (cost_s.sum() + cost_im.sum()) / img_emb.size(0)
        return loss