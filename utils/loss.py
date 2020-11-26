import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

class Myloss(nn.Module):
    def __init__(self,epsilon=1e-8):
        super(Myloss,self).__init__()
        self.epsilon = epsilon
        return
    def forward(self,input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) -(1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight)/2 
    
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ *torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def HDA_UDA(input_list, ad_net, coeff=None, myloss=Myloss()):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out = ad_net(softmax_output)
    ad_out = nn.Sigmoid()(ad_out)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    x = softmax_output
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    heuristic = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[softmax_output.size(0)//2:] = 0
    source_weight = entropy*source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:softmax_output.size(0)//2] = 0
    target_weight = entropy*target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return myloss(ad_out,dc_target,weight.view(-1, 1)), mean_entropy, heuristic

def HDA_SSDA(input_list, ad_net, coeff=None, myloss=Myloss()):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out = ad_net(softmax_output)
    batch_size = softmax_output.size(0) // 3
    dc_target = torch.from_numpy(np.array([2]*batch_size + [1] * batch_size  + [0] * batch_size)).long().cuda()

    x = softmax_output
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    heuristic = torch.mean(torch.abs(focals))*0.5

    source_mask = torch.ones_like(entropy)
    source_mask[softmax_output.size(0)//3:] = 0
    source_weight = entropy*source_mask

    label_mask = torch.ones_like(entropy)
    label_mask[softmax_output.size(0)*2//3:] = 0
    label_mask[0:softmax_output.size(0)//3] = 0
    label_weight = entropy*label_mask

    target_mask = torch.ones_like(entropy)
    target_mask[0:softmax_output.size(0)*2//3] = 0
    target_weight = entropy*target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             label_weight / torch.sum(label_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return torch.sum(weight * nn.CrossEntropyLoss(reduction='none')(ad_out, dc_target))/3*2, mean_entropy, heuristic 

def HDA_MSDA(input_list, ad_net, coeff=None, myloss=Myloss()):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    labels = input_list[2]
    ad_out = ad_net(softmax_output)

    batch_size = softmax_output.size(0) - labels.size(0)
    dc_target = torch.cat((labels+1, torch.from_numpy(np.array([0]*batch_size)).long().cuda()),0)

    x = softmax_output
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    heuristic = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[labels.size(0):] = 0
    source_weight = entropy*source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:labels.size(0)] = 0
    target_weight = entropy*target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return torch.sum(weight * nn.CrossEntropyLoss(reduction='none')(ad_out, dc_target)), mean_entropy, heuristic

