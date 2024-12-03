import torch
import torch.nn as nn
import numpy as np

def get_loss_func(config):
    if config.loss_type == 'cross_entropy':
        return Cross_Entropy_Loss(**config.cross_entropy, path2weights=config.path2weights, use_weights=config.use_weights)
    elif config.loss_type == 'focal_loss':
        return FocalLoss(**config.focal, path2weights=config.path2weights, use_weights=config.use_weights)
    else:
        raise ValueError(f"Unknown loss function: {config.loss_type}")
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0, path2weights=None, use_weights=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if use_weights:
            weight = np.load(path2weights) 
            weight = torch.from_numpy(weight).to(torch.float32)
        else:
            weight = None
        self.label_smoothing = label_smoothing
        self.use_weights = use_weights
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    
    def __str__(self):
        return super().__str__() + f" with label smoothing {self.label_smoothing}, weights {self.use_weights}, alpha {self.alpha}, gamma {self.gamma}"
    
    
class Cross_Entropy_Loss(nn.Module):
    def __init__(self, label_smoothing=0.1, path2weights=None, use_weights=True):
        super(Cross_Entropy_Loss, self).__init__()
        if use_weights:
            weight = np.load(path2weights) 
            weight = torch.from_numpy(weight).to(torch.float32)
        else:
            weight = None
        self.label_smoothing = label_smoothing
        self.use_weights = use_weights
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        return ce_loss
    
    def __str__(self):
        return super().__str__() + f" with label smoothing {self.label_smoothing} and weights {self.use_weights}"