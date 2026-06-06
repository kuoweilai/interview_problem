import torch
import torch.nn as nn
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp 
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z1, z2, device):

        batch_size = z1.shape[0]

        z = torch.concatenate([z1, z2], dim=1).reshape(-1, z1.shape[1])

        m = z @ z.T

        diag_mask = ~np.eye(batch_size * 2, dtype=bool)
        logits = m[diag_mask].reshape(batch_size * 2, -1)

        logits = logits / self.temp

        target = torch.zeros((batch_size * 2, batch_size * 2))
        target[np.arange(batch_size * 2), 2 * (np.arange(batch_size * 2) // 2)] = 1
        target = target[:, :-1]
        target = target.to(device)

        loss = self.criterion(logits, target)

        return loss
        
        