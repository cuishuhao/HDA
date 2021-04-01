import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import random


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def one_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=1)
        nn.init.zeros_(m.bias)

def two_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=2)
        nn.init.zeros_(m.bias)

def three_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=3)
        nn.init.zeros_(m.bias)

def four_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=4)
        nn.init.zeros_(m.bias)

def hun_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=100)
        nn.init.zeros_(m.bias)

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, bottleneck_dim=256, new_cls=False, class_num=1000, heuristic_num=1, heuristic_initial=False):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

    self.sigmoid = nn.Sigmoid()
    self.new_cls = new_cls
    self.heuristic_num = heuristic_num
    if new_cls:
        self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
        if heuristic_initial:
            self.fc.apply(hun_weights)
        else:
            self.fc.apply(init_weights)
        self.heuristic = nn.Linear(model_resnet.fc.in_features, class_num)
        self.heuristic.apply(init_weights)
        self.heuristic1 = nn.Linear(model_resnet.fc.in_features, class_num)
        self.heuristic1.apply(one_weights)
        self.heuristic2 = nn.Linear(model_resnet.fc.in_features, class_num)
        self.heuristic2.apply(two_weights)
        self.heuristic3 = nn.Linear(model_resnet.fc.in_features, class_num)
        self.heuristic3.apply(three_weights)
        self.heuristic4 = nn.Linear(model_resnet.fc.in_features, class_num)
        self.heuristic4.apply(four_weights)
        self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x, heuristic=True):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.heuristic_num==1:
        geuristic = self.heuristic(x)
    elif self.heuristic_num==2:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now_all = torch.cat((now1,now2),0).reshape(self.heuristic_num,-1,now1.shape[1])
        geuristic = now1+now2 
    elif self.heuristic_num==3:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        now_all = torch.cat((now1,now2,now3),0).reshape(self.heuristic_num,-1,now1.shape[1])
        geuristic = (now1+now2+now3)
    elif self.heuristic_num==4:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        now4 = self.heuristic3(x)
        geuristic = (now1+now2+now3+now4)
    elif self.heuristic_num==5:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        now4 = self.heuristic3(x)
        now5 = self.heuristic4(x)
        geuristic = (now1+now2+now3+now4+now5)
    y = self.fc(x)
    if heuristic:
        y = y - geuristic
    return x, y, geuristic

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
		    {"params":self.heuristic1.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic2.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic3.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic4.parameters(), "lr_mult":10, 'decay_mult':2},
                    {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                    {"params":self.heuristic.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, multi=1):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, multi)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
