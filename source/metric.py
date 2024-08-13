import torch
import torch.nn as nn
import sys


class _ECELoss(nn.Module):
    """
    from
    [1] https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    [2] https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes_probs, labels):
        confidences, predictions = torch.max(softmaxes_probs, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()



class ClasswiseECELoss(nn.Module):
    '''
    from
    [1] https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    def forward(self, softmaxes_probs, labels):
        num_classes = softmaxes_probs.shape[-1]
        per_class_sce = None
        for i in range(num_classes):
            class_confidences = softmaxes_probs[:, i]
            class_sce = torch.zeros(1, device=softmaxes_probs.device)
            labels_in_class = labels.eq(i)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin)
            if i == 0:
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)
        sce = torch.mean(per_class_sce)
        return sce.item()



class _MCELoss(nn.Module):
    """
    from
    [1] https://github.com/kartikgupta-at-anu/spline-calibration/blob/master/cal_metrics/MCE.py
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_MCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes_probs, labels):
        confidences, predictions = torch.max(softmaxes_probs, 1)
        accuracies = predictions.eq(labels)
        mce = torch.zeros(1)
        max_ce = 0.
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
            
                if max_ce < torch.abs(avg_confidence_in_bin - accuracy_in_bin):
                    max_ce = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
        
        mce[0] = max_ce
        return mce


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:58:58 2019

@author: zhang64
"""

'''
from :
https://github.com/futakw/DensityAwareCalibration
'''

### KDEpy==1.1.0

import torch
import numpy as np
import torch.nn.parallel

from KDEpy import FFTKDE


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return d


def ece_kde_binary(p,label,p_int=None,order=1):

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)
    
    
    x_int = np.linspace(-0.6, 1.6, num=2**14)
    
    
    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    # label_index = np.array([np.where(r==1)[0][0] for r in label])
    label_index = label
    with torch.no_grad():
        if p.shape[1] !=2:
            # p_new = torch.from_numpy(p)
            p_new = p
            p_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
        else:
            p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index
                
    method = 'triweight'
    
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
    kbw = np.std(p_b.numpy())*(N*2)**-0.2
    # kbw = np.std(dconf_1)*(N*2)**-0.2
    kbw = (np.std(dconf_1)*(N*2)**-0.2) + 1e-34
    # Mirror the data about the domain boundary
    # low_bound = 0.0
    low_bound = 1e-9
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    
    
    p_int = p_int/np.sum(p_int,1)[:,None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1,1))
        if p_int.shape[1]!=2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i,pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i,1]

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)
            
    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu)==False:
                integral[i] = np.abs(conf-accu)**order*pp2[i]  
                reliability[i] = accu
        else:
            if i>1:
                integral[i] = integral[i-1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])


def ece_hist_binary(p, label, n_bins = 15, order=1):
    
    p = np.clip(p,1e-256,1-1e-256)
    
    N = p.shape[0]
    label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
    with torch.no_grad():
        if p.shape[1] !=2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:])  
        else:
            preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)


        x = confidences.numpy()
        x = np.sort(x,axis=0)
        binCount = int(len(x)/n_bins) #number of data points in each bin
        bins = np.zeros(n_bins) #initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
            #print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins)+1,1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        
        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
    return ece_avg




def ece_eval_binary(p, label):
    mse = np.mean(np.sum((p-label)**2,1)) # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label*np.log(p))/N # log_likelihood
    accu = (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]) # Accuracy
    ece = ece_hist_binary(p,label).cpu().numpy() # ECE
    # or if KDE is used
    ece = ece_kde_binary(p,label)   

    return ece, nll, mse, accu



def ece_kde_binary_from_conf_acc(confidences, accuracies, p_int=None, order=1):
    # # points from numerical integration
    # if p_int is None:
    #     p_int = np.copy(p)
    N = confidences.shape[0]

    x_int = np.linspace(-0.6, 1.6, num=2 ** 14)  # x points to use after KDE estimated.

    # conf to tensor
    # p_b = torch.from_numpy(confidences)
    p_b = confidences
    label_binary = accuracies

    # points from numerical integration
    if p_int is None:
        pred_b_int = np.copy(p_b).reshape(-1, 1)

    method = "triweight"

    # PP1: p(z) ... estimated density of confidence being z
    dconf_1 = (
        p_b[np.where(label_binary == 1)].reshape(-1, 1)
    ) # Confidences of correct preds. Incorrect ones are useless to predict p(z).

    kbw = np.std(p_b) * (N * 2) ** -0.2  # <= should be deleted??
    kbw = (np.std(dconf_1) * (N * 2) ** -0.2) + 1e-34
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1, low_bound, up_bound)  # ???????????
    # print(dconf_1, dconf_1m)
    # Compute KDE using the bandwidth found, and twice as many grid points
    if kbw > 0:
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    else:
        print(
            "wrong kbw:",
            kbw,
        )
        print(
            "dconf_1:",
            dconf_1,
        )
        sys.exit()
    pp1[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1

    # PP2: p(z) ... Estimated density of conf being z, only using confs of integration points
    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int, low_bound, up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    # print(len(pp2), (pp1 == pp2).sum())

    # if p.shape[1] != 2:  # top label (confidence)
    #     perc = np.mean(label_binary)
    # else:  # or joint calibration for binary cases
    #     perc = np.mean(label_index)
    perc = np.mean(label_binary)

    integral = np.zeros(x_int.shape)
    reliability = np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]  # x point
        conf_i = np.abs(x_int - conf).argmin()  # idx of the x point
        if np.max([pp1[conf_i], pp2[conf_i]]) > 1e-6:
            accu = np.min([perc * pp1[conf_i] / pp2[conf_i], 1.0])
            # if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            #     accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu) == False:
                integral[i] = np.abs(conf - accu) ** order * pp2[i]
                reliability[i] = accu
        else:
            if i > 1:
                integral[i] = integral[i - 1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    # print(np.trapz(pp2[ind],x_int[ind]))
    return np.trapz(integral[ind], x_int[ind]) / np.trapz(pp2[ind], x_int[ind])