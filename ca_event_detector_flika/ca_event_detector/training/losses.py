"""
Lovász-Softmax loss for semantic segmentation.
Based on the paper by Berman et al. (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import filterfalse


def lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovász extension w.r.t sorted errors.
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present', ignore_index=-100):
    """
    Multi-class Lovász-Softmax loss.

    Args:
        probas: (P, C) class probabilities at each prediction
        labels: (P,) ground truth labels
        classes: 'all' for all classes, 'present' for classes present in labels
        ignore_index: void class labels
    """
    if probas.numel() == 0:
        return probas * 0.

    C = probas.size(1)
    losses = []

    # Filter void pixels
    valid_mask = labels != ignore_index
    if not valid_mask.all():
        labels = labels[valid_mask]
        probas = probas[valid_mask]

    class_to_sum = list(range(C)) if classes == 'all' else None
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.).to(probas.device)


def flatten_probas(probas, labels, ignore_index=-100):
    """
    Flatten predictions and labels.
    """
    if probas.dim() == 5:  # 3D case: (B, C, D, H, W)
        B, C, D, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        labels = labels.reshape(-1)
    elif probas.dim() == 4:  # 2D case: (B, C, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.reshape(-1)
    else:
        raise ValueError(f'Unexpected probas dimension: {probas.dim()}')

    return probas, labels


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for multi-class segmentation.

    Args:
        classes: 'all' for all classes, 'present' for classes present in labels
        per_image: compute loss per image instead of per batch
        ignore_index: void class labels
    """

    def __init__(self, classes='present', per_image=False, ignore_index=-100):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, *) unnormalized predictions
            labels: (B, *) ground truth labels
        """
        probas = F.softmax(logits, dim=1)

        if self.per_image:
            loss = torch.mean(torch.stack([
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0),
                                                    self.ignore_index),
                                   classes=self.classes, ignore_index=self.ignore_index)
                for prob, lab in zip(probas, labels)
            ]))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self.ignore_index),
                                      classes=self.classes, ignore_index=self.ignore_index)

        return loss


class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Lovász-Softmax loss.
    """

    def __init__(self, ce_weight=0.5, lovasz_weight=0.5, ignore_index=-100):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        ce = self.ce_loss(logits, labels)
        lovasz = self.lovasz_loss(logits, labels)
        return self.ce_weight * ce + self.lovasz_weight * lovasz


def get_loss_function(loss_type='lovasz', ignore_index=-100):
    """
    Factory function to get loss function.

    Args:
        loss_type: 'lovasz', 'cross_entropy', or 'combined'
        ignore_index: index to ignore in loss calculation

    Returns:
        Loss function instance
    """
    if loss_type == 'lovasz':
        return LovaszSoftmaxLoss(ignore_index=ignore_index)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'combined':
        return CombinedLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test loss functions
    import numpy as np

    B, C, D, H, W = 2, 4, 16, 64, 64
    logits = torch.randn(B, C, D, H, W)
    labels = torch.randint(0, C, (B, D, H, W))

    # Test Lovász-Softmax
    loss_fn = LovaszSoftmaxLoss()
    loss = loss_fn(logits, labels)
    print(f"Lovász-Softmax loss: {loss.item():.4f}")

    # Test Combined loss
    loss_fn = CombinedLoss()
    loss = loss_fn(logits, labels)
    print(f"Combined loss: {loss.item():.4f}")
