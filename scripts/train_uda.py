import argparse
import os
import os.path as osp
import sys
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import pdb
import math

import network.network as network
import utils.loss as loss
import utils.lr_schedule as lr_schedule
import dataset.preprocess as prep
from dataset.dataloader import ImageList
from distutils.version import LooseVersion


##test the model
def image_classification_test(loader, model, heuristic=False):
    start_test = True
    with torch.no_grad():
       iter_test = iter(loader["test"])
       for i in range(len(loader['test'])):
           data = iter_test.next()
           inputs = data[0]
           labels = data[1]
           inputs = inputs.cuda()
           labels = labels.cuda()
           _, outputs ,_  = model(inputs,heuristic=heuristic) 
           if start_test:
               all_output = outputs.float()
               all_label = labels.float()
               start_test = False
           else:
               all_output = torch.cat((all_output, outputs.float()), 0)
               all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

##calculate the gaussianity
def nogauss(a):
    num = a.shape[1]
    std = torch.std(a, dim=1, keepdim=True).repeat(1,num)
    mean = torch.mean(a, dim=1, keepdim=True).repeat(1,num)
    cal = (a-mean)/std
    y = torch.mean(torch.pow(cal,4),1)-3*torch.pow(torch.mean(torch.pow(cal,2),1),2)
    return torch.mean(torch.abs(y))

def train_uda(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_target(**config["prep"]['params'])
    prep_dict["target"] = prep.image_target(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    ad_net = network.AdversarialNetwork( class_num, 1024)
    ad_net = ad_net.cuda()
 
    ## set optimizer
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    #multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i,k in enumerate(gpus)])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i,k in enumerate(gpus)])
    
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    for i in range(config["num_iterations"]):
        #test
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, heuristic=config["heuristic"])
            temp_model = nn.Sequential(base_network)
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

        #save model
        if i % config["snapshot_interval"] == 0 and i:
            torch.save(base_network.state_dict(), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        loss_params = config["loss"]                  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        #dataloader
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        #data    
        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        #network
        features_source, outputs_source, focal_source = base_network(inputs_source,heuristic=config["heuristic"])
        features_target, outputs_target, focal_target = base_network(inputs_target,heuristic=config["heuristic"])
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        focals = torch.cat((focal_source,focal_target),dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        #similarity
        sim_source = torch.sum(outputs_source *focal_source,1)/torch.sqrt(torch.sum(torch.pow(outputs_source,2),1))/torch.sqrt(torch.sum(torch.pow(focal_source,2),1))
        sim_target = torch.sum(outputs_target *focal_target,1)/torch.sqrt(torch.sum(torch.pow(outputs_target,2),1))/torch.sqrt(torch.sum(torch.pow(focal_target,2),1))
        relate_source = torch.mean(torch.abs(sim_source))
        relate_target = torch.mean(torch.abs(sim_target))
        relate_all = torch.mean(torch.abs(torch.cat((sim_source,sim_target),0)))

        # calculate the theta value
        #theta = torch.acos(torch.cat((sim_source,sim_target)))
        #m_theta = torch.mean(theta)
        #s_theta = torch.std(theta)

        #calculate the gaussian
        gaussian = torch.abs(nogauss(outputs) - nogauss(outputs+focals))
        #loss calculation
        transfer_loss, mean_entropy, heuristic = loss.HDA_UDA([softmax_out,focals], ad_net, network.calc_coeff(i))
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + config["heuristic"] * heuristic #+ config["gauss"] *gaussian

        if i % config["print_num"] == 0 :
            log_str = "iter:{:05d},transfer:{:.5f},classifier:{:.5f},heuristic:{:.5f},relate:{:.5f},gaussian:{:.5f}".format(i, transfer_loss, classifier_loss, heuristic, relate_all, gaussian)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

        total_loss.backward()
        optimizer.step()

