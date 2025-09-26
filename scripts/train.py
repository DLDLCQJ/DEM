
import os
import datetime
import time
import sys
from os.path import join as opj
import copy
import numpy as np
import math
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from transformers import get_scheduler 

from utils.loss import Coral_loss, Cross_Contrastive_loss,SelfAttn_Contrastive_loss,CLSEntropyLoss,discrepancy,Self_Contrastive_loss
from utils.others import pad_dataset,custom_iter,pad_batch,consistency_check, thresholding,EWC,reinitialize_retrain,initialize_retrain_alone
from scripts.test import test_epoch
from data.data_processing import MyDataset, Filter_Dataset
from modules.GnT import FFGnT
from utils.AdamGnT import AdamGnT


def run_pretrain_epoch(args, model, src_loader, tgt_dataset, fold, epoch=None, criterion=None, optimizer=None, lr_scheduler=None, mode='train'):
    model.train() if mode=='train' else model.eval()
    tgt_loader = DataLoader(tgt_dataset, batch_size = args.batch_size)
    len_dataloader = min(len(src_loader), len(tgt_loader))
    epoch_loss, acc, total, epoch_correct= 0, 0, 0, 0
    predictions, truths = [], []
    for i, ((src_imgs, src_labels),(tgt_imgs, tgt_labels)) in enumerate(zip(src_loader,tgt_loader)):
        src_imgs,src_labels = src_imgs.to(args.device), src_labels.to(args.device)
        tgt_imgs,tgt_labels = tgt_imgs.to(args.device), tgt_labels.to(args.device)
        src_dlabels = torch.zeros(len(src_imgs)).to(args.device)
        tgt_dlabels = torch.ones(len(tgt_imgs)).to(args.device)
         ## calculating alpha
        p = float(i + epoch * len_dataloader) / (args.num_epochs_pretrain * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        src_preds,tgt_preds,domains,x1_feats_adv,x2_feats_adv,_,_ = model(src_imgs, tgt_imgs,alpha)
        if mode=='train':
            labels = src_labels
            combined_dlabels = torch.cat((src_dlabels, tgt_dlabels),dim=0)
            dlabels = combined_dlabels.to(torch.float32)
            outputs = src_preds.squeeze(1)
            doutputs = domains.squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            loss_clf = criterion(outputs, labels)
            if args.adaptive:
                loss_domain = criterion(doutputs, dlabels)
                loss_coral = Coral_loss(x1_feats_adv, x2_feats_adv)
                loss_disc = discrepancy(src_preds, tgt_preds)
                loss = loss_clf +loss_coral +loss_disc + loss_domain
            else:
                loss = loss_clf
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()*labels.size(0)
            loss.backward(retain_graph=True)
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()        # Update network parameters
                lr_scheduler.step()     # Update learning rate
                optimizer.zero_grad()   # Reset gradient
        else:
            labels = src_labels.to(torch.float32)
            outputs = src_preds.squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            cls_loss = criterion(outputs, labels)
            epoch_loss += cls_loss.item()*labels.size(0)
        # Print statistics
        total += labels.size(0)
        epoch_correct += (predicts == labels).sum().item()
        acc = epoch_correct / total
        predictions.extend(outputs.detach().cpu().numpy())
        truths.extend(labels.cpu().numpy())
    if mode == 'train':
        return epoch_loss, predictions, truths, acc
    else:
        return epoch_loss, predictions, truths, acc

def run_alone_pretrain_epoch(args, model, data, criterion=None, optimizer=None, lr_scheduler=None, mode='train'):
    model.train() if mode=='train' else model.eval()
    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size = args.batch_size)
    else:
        loader = data
    epoch_loss, total, epoch_correct= 0, 0, 0
    predictions, truths = [], []
    for i, (imgs, labels) in enumerate(loader):
        imgs,labels = imgs.to(args.device), labels.to(args.device)
        preds,_,_ = model(imgs)
        if mode=='train':
            outputs = preds.squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            loss_clf = criterion(outputs, labels)
            loss = loss_clf
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()*labels.size(0)
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()    
                lr_scheduler.step()     
                optimizer.zero_grad()   
        else:
            labels = labels.to(torch.float32)
            outputs = preds.squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            cls_loss = criterion(outputs, labels)
            epoch_loss += cls_loss.item()*labels.size(0)
        # Print statistics
        total += labels.size(0)
        epoch_correct += (predicts == labels).sum().item()
        predictions.extend(outputs.detach().cpu().numpy())
        truths.extend(labels.cpu().numpy())
    acc = epoch_correct / total
    return epoch_loss, predictions, truths, acc

def reassigned_labels(args, model, src_loader, tgt_dataset,model_state=None):
    if model_state:
        model.load_state_dict(model_state)
    model.eval()
    if isinstance(tgt_dataset, Dataset):
        tgt_loader = DataLoader(tgt_dataset, batch_size = args.batch_size)
    else:
        tgt_loader = tgt_dataset
    tgt_total,tgt_epoch_correct = 0, 0
    predictions = []
    for i, ((src_imgs, _), (tgt_imgs, tgt_labels)) in enumerate(zip(src_loader, tgt_loader)):
        src_imgs = src_imgs.to(args.device)
        tgt_imgs = tgt_imgs.to(args.device)
        tgt_labels = tgt_labels.to(args.device)
        src_preds,tgt_preds,domains,src_feats_adv,tgt_feats_adv,features1,features2 = model(src_imgs,tgt_imgs,alpha=0)
        tgt_outputs = tgt_preds.squeeze(1)
        # tgt_probs = torch.sigmoid(tgt_outputs)
        # all_cons.append(cons.detach().cpu().numpy())
        tgt_predicts = (torch.sigmoid(tgt_outputs) > 0.5).float()
        tgt_total += tgt_labels.size(0)
        tgt_epoch_correct += (tgt_predicts == tgt_labels).sum().item()
        predictions.append(tgt_predicts)
    predictions = torch.cat(predictions,dim=0)
    # confidences = np.concatenate([con for con in all_cons])
    tgt_val_acc = tgt_epoch_correct / tgt_total
    return predictions,tgt_val_acc

def reassigned_labels_alone(args, model, data):
    model.eval()
    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size = args.batch_size)
    else:
        loader = data
    total,epoch_correct = 0, 0
    predictions,all_cons = [],[]
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)
        preds,_,_ = model(imgs) ##
        outputs = preds.squeeze(1) 
        probs = torch.sigmoid(outputs)
        cons = torch.max(probs, 1 - probs)
        all_cons.append(cons.detach().cpu().numpy())
        predicts = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        epoch_correct += (predicts == labels).sum().item()
        predictions.append(predicts)
    predictions = torch.cat(predictions,dim=0)
    confidences = np.concatenate([con for con in all_cons])
    val_acc = epoch_correct / total
    return predictions,confidences,val_acc


"Updating ...."

