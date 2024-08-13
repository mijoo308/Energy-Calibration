import torch
import torch.nn as nn

from source.metric import _ECELoss, ClasswiseECELoss, _MCELoss, ece_kde_binary, ece_kde_binary_from_conf_acc
import numpy as np


class Evaluation():
    def __init__(self, probs, labels, n_bins=15):

        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(probs)

        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)

        self.y = labels.type(torch.LongTensor)
        self.probs = probs
        self.y_hat = torch.argmax(self.probs, axis=1)

        self.bin = n_bins

    def ACC(self):
        acc = sum(self.y_hat == self.y) / len(self.y)
        return acc.item()

    def CE(self):
        nll = nn.NLLLoss()
        return nll(torch.log(self.probs + 1e-34), self.y).item()

    def ECE(self):
        ece = _ECELoss(self.bin)
        return round(ece(self.probs + 1e-34, self.y)*100,4)

    def MCE(self):
        mce = _MCELoss(self.bin)
        return mce(self.probs + 1e-34, self.y)
    
    def conf(self):
        conf = torch.max(self.probs, axis=1).values
        return conf.numpy()

    def BRIER(self):
        conf = torch.max(self.probs, axis=1).values
        return (np.abs(conf - self.y)**2).mean()

    def classwise_ECE(self):
        classwise_ece = ClasswiseECELoss(self.bin)
        return classwise_ece(self.probs + 1e-34, self.y)*100


    def KDE_ECE(self, tacc=None):
        if tacc is not None:
            kde_ece = ece_kde_binary_from_conf_acc(self.conf() + 1e-34, tacc)
        else:
            kde_ece = ece_kde_binary(self.probs + 1e-34, self.y)
        return kde_ece
    




