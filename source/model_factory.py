""" helper function

author baiyu
"""

"""  #####
extension of 
https://github.com/weiaicunzai/pytorch-cifar100
"""

import os
import sys

import torch
from torch.optim.lr_scheduler import _LRScheduler
# import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn as nn


def get_network(network_name, dataset, path, ddp_trained=False):
    """ return given network
    """
    if dataset == 'cifar10' or dataset =='cifar100':
        if dataset=='cifar10':
            class_num = 10
        elif dataset == 'cifar100':
            class_num = 100

        if network_name == 'vgg16':
            from models.vgg import vgg16_bn
            net = vgg16_bn(class_num=class_num)
        elif network_name == 'vgg13':
            from models.vgg import vgg13_bn
            net = vgg13_bn(class_num=class_num)
        elif network_name == 'vgg11':
            from models.vgg import vgg11_bn
            net = vgg11_bn(class_num=class_num)
        elif network_name == 'vgg19':
            from models.vgg import vgg19_bn
            net = vgg19_bn(class_num=class_num)
        elif network_name == 'densenet121':
            from models.densenet import densenet121
            net = densenet121(class_num=class_num)
        elif network_name == 'densenet161':
            from models.densenet import densenet161
            net = densenet161(class_num=class_num)
        elif network_name == 'densenet169':
            from models.densenet import densenet169
            net = densenet169(class_num=class_num)
        elif network_name == 'densenet201':
            from models.densenet import densenet201
            net = densenet201(class_num=class_num)
        elif network_name == 'googlenet':
            from models.googlenet import googlenet
            net = googlenet(class_num=class_num)
        elif network_name == 'inceptionv3':
            from models.inceptionv3 import inceptionv3
            net = inceptionv3(class_num=class_num)
        elif network_name == 'inceptionv4':
            from models.inceptionv4 import inceptionv4
            net = inceptionv4(class_num=class_num)
        elif network_name == 'inceptionresnetv2':
            from models.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2(class_num=class_num)
        elif network_name == 'xception':
            from models.xception import xception
            net = xception(class_num=class_num)
        elif network_name == 'resnet18':
            from models.resnet import resnet18
            net = resnet18(class_num=class_num)
        elif network_name == 'resnet34':
            from models.resnet import resnet34
            net = resnet34(class_num=class_num)
        elif network_name == 'resnet50':
            from models.resnet import resnet50
            net = resnet50(class_num=class_num)
        elif network_name == 'resnet101':
            from models.resnet import resnet101
            net = resnet101(class_num=class_num)
        elif network_name == 'resnet152':
            from models.resnet import resnet152
            net = resnet152(class_num=class_num)
        elif network_name == 'preactresnet18':
            from models.preactresnet import preactresnet18
            net = preactresnet18(class_num=class_num)
        elif network_name == 'preactresnet34':
            from models.preactresnet import preactresnet34
            net = preactresnet34(class_num=class_num)
        elif network_name == 'preactresnet50':
            from models.preactresnet import preactresnet50
            net = preactresnet50(class_num=class_num)
        elif network_name == 'preactresnet101':
            from models.preactresnet import preactresnet101
            net = preactresnet101(class_num=class_num)
        elif network_name == 'preactresnet152':
            from models.preactresnet import preactresnet152
            net = preactresnet152(class_num=class_num)
        elif network_name == 'resnext50':
            from models.resnext import resnext50
            net = resnext50(class_num=class_num)
        elif network_name == 'resnext101':
            from models.resnext import resnext101
            net = resnext101(class_num=class_num)
        elif network_name == 'resnext152':
            from models.resnext import resnext152
            net = resnext152(class_num=class_num)
        elif network_name == 'shufflenet':
            from models.shufflenet import shufflenet
            net = shufflenet(class_num=class_num)
        elif network_name == 'shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            net = shufflenetv2(class_num=class_num)
        elif network_name == 'squeezenet':
            from models.squeezenet import squeezenet
            net = squeezenet(class_num=class_num)
        elif network_name == 'mobilenet':
            from models.mobilenet import mobilenet
            net = mobilenet(class_num=class_num)
        elif network_name == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            net = mobilenetv2(class_num=class_num)
        elif network_name == 'nasnet':
            from models.nasnet import nasnet
            net = nasnet(class_num=class_num)
        elif network_name == 'attention56':
            from models.attention import attention56
            net = attention56(class_num=class_num)
        elif network_name == 'attention92':
            from models.attention import attention92
            net = attention92(class_num=class_num)
        elif network_name == 'seresnet18':
            from models.senet import seresnet18
            net = seresnet18(class_num=class_num)
        elif network_name == 'seresnet34':
            from models.senet import seresnet34
            net = seresnet34(class_num=class_num)
        elif network_name == 'seresnet50':
            from models.senet import seresnet50
            net = seresnet50(class_num=class_num)
        elif network_name == 'seresnet101':
            from models.senet import seresnet101
            net = seresnet101(class_num=class_num)
        elif network_name == 'seresnet152':
            from models.senet import seresnet152
            net = seresnet152(class_num=class_num)
        elif network_name == 'wideresnet':
            from models.wideresidual import wideresnet
            net = wideresnet(class_num=class_num)
        elif network_name == 'stochasticdepth18':
            from models.stochasticdepth import stochastic_depth_resnet18
            net = stochastic_depth_resnet18(class_num=class_num)
        elif network_name == 'stochasticdepth34':
            from models.stochasticdepth import stochastic_depth_resnet34
            net = stochastic_depth_resnet34(class_num=class_num)
        elif network_name == 'stochasticdepth50':
            from models.stochasticdepth import stochastic_depth_resnet50
            net = stochastic_depth_resnet50(class_num=class_num)
        elif network_name == 'stochasticdepth101':
            from models.stochasticdepth import stochastic_depth_resnet101
            net = stochastic_depth_resnet101(class_num=class_num)

        else:
            print('the network name you have entered is not supported yet')
            sys.exit()

        checkpoint = torch.load(path, map_location="cuda:0")
        if ddp_trained:
            for key in list(checkpoint.keys()):
                if 'module.' in key:
                    checkpoint[key.replace('module.','')] = checkpoint[key]
                    del checkpoint[key]
                    
        net.load_state_dict(checkpoint)

    else: # for imagenet
        net = torch.hub.load("pytorch/vision:v0.13.1", network_name, weights="IMAGENET1K_V1")

    return net


def test(net, dataloader, gpu=False): # should return logit
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloader)))

            if gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    top1_error = 1 - correct_1 / len(dataloader.dataset)
    top5_error = 1 - correct_5 / len(dataloader.dataset)
    accuracy = 1 - top1_error

    print()
    print("Top 1 err: ", top1_error)
    print("Top 5 err: ", top5_error)
    print("Accuracy : ", accuracy)

    return accuracy

def inference(net, dataloader, gpu=False):
    all_logits, all_labels = None, None
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloader)))
            if gpu:
                image = image.cuda()
                label = label.cuda()
            logits = net(image)

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = label
            else:
                all_labels = torch.cat((all_labels, label))
    
    return all_logits, all_labels
