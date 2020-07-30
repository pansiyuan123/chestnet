import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is None:
            loss = - (targets * ((1 - inputs) ** self.gamma) * inputs.log()
                        + (1 - targets) * (inputs ** self.gamma) * (1 - inputs).log())
        else:
            loss = - (self.alpha * targets * (1 - inputs) ** self.gamma * inputs.log()
                      + (1 - self.alpha) * (1 - targets) * inputs ** self.gamma * (1 - inputs).log())

        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MultiLabelBCELoss(nn.Module):
    def __init__(self, alpha=None, reduction='mean'):
        super(MultiLabelBCELoss, self).__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is None:
            loss = - (targets * inputs.log() + (1 - targets) * (1 - inputs).log())
        else:
            loss = - (self.alpha * targets * inputs.log()
                        + self.alpha * (1 - targets) * (1 - inputs).log())

        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class MultiweightBCELoss(nn.Module):
    def __init__(self, AUROC=0.5, reduction='mean'):
        super(MultiweightBCELoss, self).__init__()
        self.AUROC = AUROC
        self.reduction = reduction

    def forward(self, inputs, targets):
        self.alpha = F.softmax(2 * (1 - self.AUROC), dim=0)

        loss = - (self.alpha * targets * inputs.log()
                    + self.alpha * (1 - targets) * (1 - inputs).log())
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class GHMC_Loss(nn.Module):
    def __init__(self, bins=10, momentum=0):
        super(GHMC_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = [0.] * bins

    def forward(self, inputs, targets):

        bs, cla = inputs.shape
        weights = torch.zeros

        #gradient length
        g = torch.abs(inputs.detach() - targets)

        tot = bs * cla
        n = 0
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                weights[inds] = tot / self.acc_sum[i]
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy(inputs, targets, weights, reduction='sum') / tot
        return loss

class GHMS_Loss(nn.Module):
    def __init__(self, device, bins=10, momentum=0):
        super(GHMS_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.device = device
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = [0.] * bins
        self.smooth = 1 - 25 * (torch.arange(0, 1, 1/bins)) ** 2
        self.smooth[int(bins*0.2):] = 0
        self.smooth[self.smooth < 0] = 0

    def forward(self, inputs, targets):

        bs, cla = inputs.shape

        #gradient length
        g = torch.abs(inputs.detach() - targets).mean(1)
        weights = torch.zeros(bs).to(self.device)

        tot = bs * cla
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                weights[inds] = self.smooth[i] * tot / (self.acc_sum[i])
        #
        weights = torch.pow(weights, 2)
        weights /= weights.sum()
        # weights = F.softmax(weights)
        weights[(weights < 1e-6)] = 0

        loss = - (targets * inputs.log() + (1 - targets) * (1 - inputs).log())
        loss = loss.mean(1)
        loss = (loss * weights).sum()

        return loss