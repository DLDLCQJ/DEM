
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


def run_retrain_epoch(args, model_C,model_R,optimizer,lr_scheduler,tgt_dataset,src_loader=None,fold=None,sub_epoch=None,criterion=None,mode='train',gnt_extract=None,gnt_clf=None):
  
    model_C.train() if mode=='train' else model_C.eval()
    tgt_loader = DataLoader(tgt_dataset, batch_size = args.batch_size)
    len_dataloader = min(len(src_loader), len(tgt_loader))
    predictions, truths, accs =[],[],[]
    probs_, rewards_ =[], []
    epoch_loss, src_acc, tgt_acc, src_total,tgt_total,src_epoch_correct,tgt_epoch_correct = 0, 0, 0, 0, 0, 0, 0
    best_loss = -float('inf')
    previous_features1, previous_features2 = None, None
    for i, ((src_imgs, src_labels),(tgt_imgs, tgt_labels)) in enumerate(zip(src_loader,tgt_loader)):
        src_imgs,src_labels = src_imgs.to(args.device), src_labels.to(args.device)
        tgt_imgs,tgt_labels = tgt_imgs.to(args.device), tgt_labels.to(args.device)
        src_dlabels = torch.zeros(len(src_imgs)).to(args.device)
        tgt_dlabels = torch.ones(len(tgt_imgs)).to(args.device)
        if mode=='train':
             ## calculating alpha
            alpha = 2. / (1. + np.exp(-10 * (sub_epoch / args.num_epochs_train))) - 1
            src_preds,tgt_preds, domains,x1_feats_adv,x2_feats_adv,features1,features2 = model_C(src_imgs,tgt_imgs,alpha)
            combined_dlabels = torch.cat((src_dlabels, tgt_dlabels),dim=0)
            dlabels = combined_dlabels.to(torch.float32)
            tgt_outputs = tgt_preds.squeeze(1)
            doutputs = domains.squeeze(1)
            tgt_predicts = (torch.sigmoid(tgt_outputs) > 0.5).float()
            loss_clf = criterion(tgt_outputs, tgt_labels)
            tgt_total += tgt_labels.size(0)
            tgt_epoch_correct += (tgt_predicts == tgt_labels).sum().item()
            tgt_acc = tgt_epoch_correct / tgt_total
            if args.adaptive:
                loss_domain = criterion(doutputs, dlabels)
                loss_disc = discrepancy(src_preds, tgt_preds)
                loss_coral = Coral_loss(x1_feats_adv, x2_feats_adv)
                loss = loss_clf + loss_coral + loss_disc + loss_domain 
            else:
                loss = loss_clf
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()*tgt_labels.size(0)
            previous_features1 = features1
            previous_features2 = features2
            loss.backward(retain_graph=True)
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()        
                lr_scheduler.step()    
                optimizer.zero_grad()   
                if args.reinitial=='cbp':
                    gnt_extract.gen_and_test(features=previous_features1)
                    gnt_clf.gen_and_test(features=previous_features2)
            ##
            probs = model_R(x2_feats_adv.detach())
            probs = probs.squeeze(1)
            tgt_probs = torch.sigmoid(probs)
            probs_.extend(tgt_probs)
        else:
            src_preds,tgt_preds, probs,x1_feats_adv,x2_feats_adv,_,_ = model_C(src_imgs,tgt_imgs,alpha=1.0)
            ## statistic
            tgt_outputs = tgt_preds.squeeze(1)
            cls_loss = criterion(tgt_outputs, tgt_labels)
            epoch_loss += cls_loss.item()*tgt_labels.size(0)
            tgt_predicts = (torch.sigmoid(tgt_outputs) > 0.5).float()
            tgt_total += tgt_labels.size(0)
            tgt_epoch_correct += (tgt_predicts == tgt_labels).sum().item()
            tgt_acc = tgt_epoch_correct / tgt_total
            ##
            src_outputs = src_preds.squeeze(1)
            src_predicts = (torch.sigmoid(src_outputs) > 0.5).float()
            src_total += src_labels.size(0)
            src_epoch_correct += (src_predicts == src_labels).sum().item()
            src_acc = src_epoch_correct / src_total
            accs.append(src_acc)
            ##
            # probs = model_R(x2_feats_adv)
            # probs = probs.squeeze(1)
            # tgt_probs = torch.sigmoid(probs)
        # Print statistics
        predictions.extend(tgt_outputs.detach().cpu().numpy())
        truths.extend(tgt_labels.cpu().numpy())
        with open(opj(args.save_dir, f"model_fold{fold+1}_{args.continual_type}_catastrophic_forgetting_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}__{args.tgt_img_file}.pkl"), "wb") as f:
            pickle.dump(accs, f)
    if mode == 'train':    
        return epoch_loss, predictions,probs_, truths, tgt_acc
    else:
        return epoch_loss, predictions,tgt_outputs, truths, tgt_acc, src_acc

def collate_to_device(batch, device):
    inputs, labels = zip(*batch)

    # Move tensors to the correct device before stacking
    inputs = [input.to(device) for input in inputs]
    labels = [label.to(device) for label in labels]

    # Stack and return
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels


def run_alone_retrain_epoch(args, model, optimizer, lr_scheduler, next_tgt_dataset,src_loader, fold=None,criterion=None, mode='train',gnt_extract=None,gnt_clf=None):
    model.train() if mode=='train' else model.eval()
    tgt_loader = DataLoader(next_tgt_dataset, batch_size = args.batch_size,drop_last= True)
    predictions, truths, accs = [], [], []
    epoch_loss,tgt_acc, src_acc, tgt_total, src_total,tgt_epoch_correct,src_epoch_correct = 0, 0, 0, 0, 0, 0, 0
    previous_features1, previous_features2 = None, None
    for i, ((src_imgs, src_labels),(tgt_imgs, tgt_labels)) in enumerate(zip(src_loader,tgt_loader)):
        src_imgs, src_labels = src_imgs.to(args.device), src_labels.to(args.device)
        tgt_imgs,tgt_labels = tgt_imgs.to(args.device), tgt_labels.to(args.device)
        tgt_preds,features1,features2 = model(tgt_imgs)
        if mode=='train':
            tgt_outputs = tgt_preds.squeeze(1)
            tgt_predicts = (torch.sigmoid(tgt_outputs) > 0.5).float()
            loss_clf = criterion(tgt_outputs, tgt_labels)
            tgt_total += tgt_labels.size(0)
            tgt_epoch_correct += (tgt_predicts == tgt_labels).sum().item()
            tgt_acc = tgt_epoch_correct / tgt_total
            loss = loss_clf
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()*tgt_labels.size(0)
            loss.backward()
            previous_features1 = features1
            previous_features2 = features2
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()        
                lr_scheduler.step()    
                optimizer.zero_grad()
                if  args.reinitial=='cbp':
                    gnt_extract.gen_and_test(features=previous_features1)
                    gnt_clf.gen_and_test(features=previous_features2)
        else:
            tgt_outputs = tgt_preds.squeeze(1)
            cls_loss = criterion(tgt_outputs, tgt_labels)
            epoch_loss += cls_loss.item()*tgt_labels.size(0)
            tgt_predicts = (torch.sigmoid(tgt_outputs) > 0.5).float()
            tgt_total += tgt_labels.size(0)
            tgt_epoch_correct += (tgt_predicts == tgt_labels).sum().item()
            tgt_acc = tgt_epoch_correct / tgt_total
            ##
            src_preds,_,_ = model(src_imgs) 
            src_outputs = src_preds.squeeze(1)
            src_predicts = (torch.sigmoid(src_outputs) > 0.5).float()
            src_total += src_labels.size(0)
            src_epoch_correct += (src_predicts == src_labels).sum().item()
            src_acc = src_epoch_correct / src_total
            accs.append(src_acc)

        # Print statistics
        predictions.extend(tgt_outputs.detach().cpu().numpy())
        truths.extend(tgt_labels.cpu().numpy())
        with open(opj(args.save_dir, f"model_fold{fold+1}_{args.continual_type}_catastrophic_forgetting_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}__{args.tgt_img_file}.pkl"), "wb") as f:
            pickle.dump(accs, f)
     
    if mode == 'train':    
        return epoch_loss, predictions, truths, tgt_acc
    else:
        return epoch_loss, predictions, truths, tgt_acc, src_acc
 
def pretrain_epoch(args, model, src_tr_loader, src_val_loader, tgt_tr_dataset, tgt_val_dataset, fold, criterion, optimizer, lr_scheduler):
    best_loss = float('inf')
    total_time = 0
    start_time = time.time()
    
    # Tensor to handle early stopping synchronization
    stop_signal = torch.zeros(1).to(args.device)

    for pre_epoch in range(args.num_epochs_pretrain):
        if args.distributed:
            src_tr_loader.sampler.set_epoch(pre_epoch)
            src_val_loader.sampler.set_epoch(pre_epoch)

        if args.rank == 0:
            print(f"Starting Pretraining epoch {pre_epoch+1}")
        
        if args.continual_type == 'CRL':
            run_pretrain_epoch(args, model, src_tr_loader, tgt_tr_dataset, fold, epoch=pre_epoch, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='train')
        else:
            run_alone_pretrain_epoch(args, model, src_tr_loader, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='train')
            
        epoch_time = time.time() - start_time
        total_time += epoch_time

        with torch.no_grad():
            if args.continual_type == 'CRL':
                # Assuming run_pretrain_epoch returns tensors on device. If scalars/lists, convert them.
                val_loss, val_predicts, val_truths, _ = run_pretrain_epoch(args, model, src_val_loader, tgt_val_dataset, fold, epoch=pre_epoch, criterion=criterion, mode='eval')
            else:
                val_loss, val_predicts, val_truths, _ = run_alone_pretrain_epoch(args, model, src_val_loader, criterion=criterion, mode='eval')

        if args.distributed:
            # Aggregate Loss
            val_loss = torch.tensor(val_loss).to(args.device) if not torch.is_tensor(val_loss) else val_loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= args.world_size
            
            # Gather Predictions & Labels
            # Ensure they are tensors on the GPU
            if not torch.is_tensor(val_predicts): val_predicts = torch.tensor(val_predicts).to(args.device)
            if not torch.is_tensor(val_truths): val_truths = torch.tensor(val_truths).to(args.device)
            
            preds_list = [torch.zeros_like(val_predicts) for _ in range(args.world_size)]
            labels_list = [torch.zeros_like(val_truths) for _ in range(args.world_size)]
            
            dist.all_gather(preds_list, val_predicts)
            dist.all_gather(labels_list, val_truths)
            
            all_preds = torch.cat(preds_list)
            all_labels = torch.cat(labels_list)
        else:
            all_preds, all_labels = val_predicts, val_truths

        if args.rank == 0:
            # Calculate accuracy on gathered data
            # Assuming binary classification logic based on your code context
            probs = torch.sigmoid(all_preds)
            preds_binary = (probs > 0.5).float()
            val_acc = accuracy_score(all_labels.cpu().numpy(), preds_binary.cpu().numpy())
            
            print(50 * "--")
            print(f"Fold {fold + 1}, Pre_Epoch [{pre_epoch + 1}/{args.num_epochs_pretrain}], Cumulative_time:{total_time:.4f}, Epoch_time:{epoch_time:.4f}, Pre_val_loss: {val_loss.item():.4f}, Pre_val_acc: {val_acc:.4f}")

            # Early Stopping Check
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                counter = 0
            else:
                counter += 1
                if counter >= args.pre_patience:
                    print(f"Early stopping pretrain triggered on fold {fold + 1}, pre_epoch {pre_epoch + 1}")
                    stop_signal.fill_(1.0)
    
        if args.distributed:
            dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1.0:
            break


def retrain_epoch(args, model_C, model_R, initial_model_state, src_tr_loader, src_val_loader, thre_tgt_tr_dataset, tgt_val_dataset, fold, epoch, iterate,
                  criterion, optimizer=None, lr_scheduler=None, gnt_extract=None, gnt_clf=None):
    
    # Load initial state (ensure this is done on all ranks)
    model_C.load_state_dict(initial_model_state)
    
    counter = 0    
    best_model_state, best_results_tr, best_results_val, best_val_acc = None, None, None, None
    best_loss = float('inf')
    best_probs = None # Initialize to avoid UnboundLocalError
    results_tr, results_val = [], []
    total_time = 0
    start_time = time.time()
    
    stop_signal = torch.zeros(1).to(args.device)

    for re_epoch in range(args.num_epochs_train):
        if args.distributed:
            src_tr_loader.sampler.set_epoch(re_epoch)
            src_val_loader.sampler.set_epoch(re_epoch)

        if args.continual_type == 'CRL':
            tr_loss, tr_predicts, tr_probs, tr_truths, _ = run_retrain_epoch(args, model_C, model_R, optimizer, lr_scheduler, thre_tgt_tr_dataset, src_loader=src_tr_loader, fold=fold, sub_epoch=re_epoch, criterion=criterion, mode='train', gnt_extract=gnt_extract, gnt_clf=gnt_clf)
        elif args.continual_type == 'RL':
            tr_loss, tr_predicts, tr_truths, _ = run_alone_retrain_epoch(args, model_C, model_R, optimizer, lr_scheduler, thre_tgt_tr_dataset, src_tr_loader, fold=fold, criterion=criterion, mode='train', gnt_extract=gnt_extract, gnt_clf=gnt_clf)
            tr_probs = torch.sigmoid(tr_predicts) # Derive probs if not returned

        epoch_time = time.time() - start_time
        total_time += epoch_time

        with torch.no_grad():
            if args.continual_type == 'CRL':
                val_loss, val_predicts, val_probs, val_truths, _, _ = run_retrain_epoch(args, model_C, model_R, optimizer, lr_scheduler, tgt_val_dataset, src_loader=src_val_loader, fold=fold, sub_epoch=re_epoch, criterion=criterion, mode='eval', gnt_extract=gnt_extract, gnt_clf=gnt_clf)
            elif args.continual_type == 'RL':
                val_loss, val_predicts, val_truths, _, _ = run_alone_retrain_epoch(args, model_C, model_R, optimizer, lr_scheduler, tgt_val_dataset, src_val_loader, fold=fold, criterion=criterion, mode='eval', gnt_extract=gnt_extract, gnt_clf=gnt_clf)
                val_probs = torch.sigmoid(val_predicts)

        if args.distributed:
            # Reduce Losses
            tr_loss = torch.tensor(tr_loss).to(args.device) if not torch.is_tensor(tr_loss) else tr_loss
            val_loss = torch.tensor(val_loss).to(args.device) if not torch.is_tensor(val_loss) else val_loss
            
            dist.all_reduce(tr_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            tr_loss /= args.world_size
            val_loss /= args.world_size
            
            # Gather Predictions/Labels (Validation) for accurate metric calculation
            if not torch.is_tensor(val_predicts): val_predicts = torch.tensor(val_predicts).to(args.device)
            if not torch.is_tensor(val_truths): val_truths = torch.tensor(val_truths).to(args.device)
            
            val_preds_list = [torch.zeros_like(val_predicts) for _ in range(args.world_size)]
            val_labels_list = [torch.zeros_like(val_truths) for _ in range(args.world_size)]
            
            dist.all_gather(val_preds_list, val_predicts)
            dist.all_gather(val_labels_list, val_truths)
            
            all_val_preds = torch.cat(val_preds_list)
            all_val_labels = torch.cat(val_labels_list)

            # Note: We usually don't need to gather Train preds unless debugging, 
            # calculating Train Acc roughly on local rank is often acceptable, 
            # but for consistency we can gather them or just calc local acc. 
            # Below I calc local acc for train to save overhead, but full acc for val.
        else:
            all_val_preds, all_val_labels = val_predicts, val_truths

        if args.rank == 0:
            # Calc Val metrics
            val_probs_all = torch.sigmoid(all_val_preds)
            val_preds_binary = (val_probs_all > 0.5).float()
            tgt_val_acc = accuracy_score(all_val_labels.cpu().numpy(), val_preds_binary.cpu().numpy())
            
            # Calc Train metrics (Approximate or Gather if needed. Here using local for display)
            # If you strictly need exact global train acc, you must gather tr_predicts too.
            tgt_tr_acc = accuracy_score(tr_truths.cpu().numpy(), (torch.sigmoid(tr_predicts) > 0.5).cpu().float().numpy())

            print(50 * "--")
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Re_Epoch [{re_epoch + 1}/{args.num_epochs_train}], Cumulative_time:{total_time:.4f}, Epoch_time:{epoch_time:.4f}, Tr_Loss: {tr_loss.item():.4f}, Tr_Acc: {tgt_tr_acc:.4f}, Val_loss:{val_loss.item():.4f}, Val_tgt_acc:{tgt_val_acc:.4f}")

            # Early Stopping Check
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_epoch = re_epoch
                # Save best state (unwrap DDP if present)
                if isinstance(model_C, torch.nn.parallel.DistributedDataParallel):
                     best_model_state = copy.deepcopy(model_C.module.state_dict())
                else:
                     best_model_state = copy.deepcopy(model_C.state_dict())
                
                best_probs = tr_probs # Note: this is local probs. For full saving you might want to gather.
                best_val_acc = tgt_val_acc
                
                # Construct result dictionaries
                best_results_tr = {
                    "fold": fold + 1,
                    "epoch": best_epoch + 1,
                    "predictions": tr_predicts.cpu().tolist(), # Local preds
                    "labels": tr_truths.cpu().tolist()
                }
                best_results_val = {
                    "fold": fold + 1,
                    "epoch": best_epoch + 1,
                    "predictions": all_val_preds.cpu().tolist(), # Global preds
                    "labels": all_val_labels.cpu().tolist()
                }
                counter = 0
            else:
                counter += 1
                if counter >= args.re_patience:
                    print(f"Early stopping retrain triggered on fold {fold + 1}, re_epoch {re_epoch + 1}")
                    stop_signal.fill_(1.0)
                    
            if best_results_tr: results_tr.append(best_results_tr)
            if best_results_val: results_val.append(best_results_val)

        if args.distributed:
            dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1.0:
            break

    return results_tr, results_val, best_val_acc, best_model_state, best_probs
                      
def sigmoid(X):
   return 1/(1+np.exp(-X))

def softmax(X):
    e_X = np.exp(X - np.max(X))
    return e_X / e_X.sum()

def tanh(X):
    return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X)+ 1e-8)

def normalize_to_prob(X):
    X = np.clip(X, a_min=0, a_max=None)  # Ensure no negatives
    return X / (np.sum(X) + 1e-8)

def flatten(a):  
    res = []  
    for x in a:  
        if isinstance(x, list):  
            res.extend(flatten(x))  # Recursively flatten nested lists  
        else:  
            res.append(x)  # Append individual elements  
    return res 

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(rewards)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns

def compute_policy_loss(probs, rewards, entropies=None, beta=0.1):
    # Normalize returns for stability
    # rewards = torch.tensor(rewards)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
    # probs = torch.stack(probs)
    # entropies = torch.stack(entropies)
    print('probs:',probs)
    print('rewards:',rewards)
    print('entropies:',entropies)
    print('probs_len:',len(probs)) 
    print('rewards_len:',len(rewards))
    print('entropies_len:',len(entropies))
    loss = 0.0
    for prob, reward, entropy in zip(probs, rewards, entropies):
        # Policy gradient term + entropy regularization term
        loss += -prob * reward - beta * entropy
    print('loss:', loss)
    return loss

def crossover(updated_chromo, previous_chromo):
    cross_len = len(updated_chromo)
    p1 = updated_chromo
    p2 = previous_chromo 
    crossover_point = torch.randint(1, cross_len-1,(1,)).item()
    child = torch.cat((p1[:crossover_point], p2[crossover_point:]),dim=0)
    return child

def mutation(args,labels, probs=None):
    mutated_labels = labels.clone()
    actions = torch.zeros_like(labels, dtype=torch.float)
    if probs is not None:
        for i in range(len(mutated_labels)):
            probs = torch.stack(probs) if isinstance(probs, list) else probs
            if random.random() > probs[i]:
                mutated_labels[i] = 1 - labels[i] 
                actions[i] = 1
            else:
                actions[i] = 0
        return mutated_labels, actions
    else:
        for i in range(len(mutated_labels)):
            if random.random() > args.MUT_RATE:
                mutated_labels[i] = 1 - labels[i]
        return mutated_labels

def update_prob_maps(init_con_maps,prev_con_maps, con_maps, subset_indices, new_accuracy, prev_accuracy,K=1):
    init_con_maps = torch.stack(init_con_maps) if isinstance(init_con_maps, list) else init_con_maps
    prev_con_maps = torch.stack(prev_con_maps) if isinstance(prev_con_maps, list) else prev_con_maps
    con_maps = torch.stack(con_maps) if isinstance(con_maps, list) else con_maps
    delta_acc = new_accuracy - prev_accuracy
    lambda_val = sigmoid(delta_acc* K)
    # sf = abs(delta_acc)*10 
    updated_con_maps = con_maps.clone() 
    for i, idx in enumerate(subset_indices):
        curr_con = con_maps[i]
        prev_con = prev_con_maps[idx]
        init_con = init_con_maps[idx]
        # Proctecting term 
        protecting_term = lambda_val * (curr_con - prev_con)
        # Forgetting term
        forgetting_term = (1 - lambda_val) * (prev_con - init_con)
        updated_con_maps[i] = prev_con + protecting_term - forgetting_term
    
    return updated_con_maps

def update_label_maps(init_label_maps,prev_label_maps, label_maps, prev_labels, new_labels, new_accuracy, init_accuracy,K=5):
    init_label_maps = torch.stack(init_label_maps) if isinstance(init_label_maps, list) else init_label_maps
    prev_label_maps = torch.stack(prev_label_maps) if isinstance(prev_label_maps, list) else prev_label_maps
    label_maps = torch.stack(label_maps) if isinstance(label_maps, list) else label_maps
    delta_acc = new_accuracy - init_accuracy
    #print(f"input label probability maps:{label_maps}")
    lambda_val = sigmoid(delta_acc * K)
    #print(f"lambda_val:{lambda_val},prev_labels_len:{len(prev_labels)},label_maps_len:{len(label_maps)}")
    updated_label_maps = label_maps.clone()  
    for i in range(len(prev_labels)):
        if prev_labels[i] != new_labels[i] :
            curr_label = label_maps[i]
            prev_label = prev_label_maps[i]
            init_label = init_label_maps[i]
            # Proctecting term
            protecting_term = lambda_val * (curr_label - prev_label)
            # Forgetting term
            forgetting_term = (1 - lambda_val) * (prev_label - init_label)
            updated_label_maps[i] = prev_label + protecting_term - forgetting_term
    return updated_label_maps

def Screening_Phase(args,model_C,model_R,src_tr_loader, src_val_loader,tgt_tr_dataset, tgt_val_dataset,tgt_tr_preds, fold, epoch,
                    optimizers=None, lr_scheduler=None,criterion=None,gnt_extract=None,gnt_clf=None):
    model_state = copy.deepcopy(model_C.state_dict())
    pseudo_tgt_tr_dataset = tgt_tr_dataset.update_labels(tgt_tr_preds)
    init_idx = list(range(0, len(tgt_tr_dataset), 1))
    thre_buffer_state = [([],[],[],[],init_idx,pseudo_tgt_tr_dataset,0.0,model_state,optimizers['C'],lr_scheduler,criterion,gnt_extract,gnt_clf)] 
    thre_candidates = []
    #selective_accs=[]
    rewards_, probs_,entropies_ = [],[],[]
    failure_count=0
    org_init_prob_maps=[]
    break_flag = False
    for iterate in range(args.num_thre_iterates):
        init_prob_maps = thre_buffer_state[0][2]
        init_prob_idx = thre_buffer_state[0][4]
        prev_thre_candidates = []
        prev_thre_candidates_idx = []
        if break_flag:
            break
        for idx, (results_tr,results_val,thre_probs,joint_prob,thre_indices,thre_dataset,thre_acc,thre_model_state,optimizer,lr_scheduler,criterion,gnt_extract,gnt_clf) in enumerate(thre_buffer_state):
            if thre_acc==0.0:
                updated_indices = thre_indices
                updated_dataset = pseudo_tgt_tr_dataset
                print(f'pseudo_tgt_tr_dataset_len:{len(pseudo_tgt_tr_dataset)},updated_indices:{len(updated_indices)},updated_dataset_len:{len(updated_dataset)}')# Training for probs and rewards
            else:
                # Performing stochastic sampling
                probs_tensor = torch.stack(thre_probs) if isinstance(thre_probs, list) else thre_probs
                actions = (torch.rand_like(probs_tensor) < probs_tensor).float()
                updated_indices = [idx for (idx, val) in zip(thre_indices,actions) if val == 1]
                log_probs = torch.distributions.Bernoulli(probs=probs_tensor).log_prob(actions)
                joint_prob = log_probs.sum()/len(log_probs)
                H = - (probs_tensor * torch.log(probs_tensor + 1e-10) + (1 - probs_tensor) * torch.log(1 - probs_tensor + 1e-10)).sum()/len(probs_tensor)
                print(f'iterate:{iterate}, idx:{idx}, thre_probs_len:{len(thre_probs)},actions_len: {len(actions)},thre_indices_len: {len(thre_indices)},updated_indices_len:{len(updated_indices)}')
                print(f'probs_tensor:{probs_tensor},log_probs:{log_probs},joint_prob:{repr(joint_prob)}')
                updated_dataset = Filter_Dataset(pseudo_tgt_tr_dataset, updated_indices)
                print(f'pseudo_tgt_tr_dataset_len:{len(pseudo_tgt_tr_dataset)},updated_indices_len:{len(updated_indices)},updated_dataset_len:{len(updated_dataset)}')
                if (len(updated_indices) <= args.mini_sample_size or len(updated_indices) % args.batch_size <=2):
                    print(f"Skipping this candidate as only {len(updated_indices)} indices were threshed.")
                    prev_thre_candidates.append(None)
                    prev_thre_candidates_idx.append(None)
                    continue
                # checking consistency
                tgt_dataset = Filter_Dataset(tgt_tr_dataset, updated_indices)
                consistency_thresh = consistency_check(tgt_dataset.labels.data.cpu().numpy(),updated_dataset.labels.data.cpu().numpy())
                print(f'Thre Consistency: {consistency_thresh:.2f}%')
            # Training for probs and rewards
            updated_results_tr,updated_results_val,updated_acc,updated_model_state,updated_probs = retrain_epoch(args,model_C,model_R,thre_model_state,src_tr_loader,src_val_loader,updated_dataset,tgt_val_dataset,fold,epoch,iterate, 
                                                                                                        criterion,optimizer,lr_scheduler,gnt_extract,gnt_clf) 
            # balancing probs_map
            if thre_acc ==0.0:
                org_init_prob_maps = updated_probs
                updated_prob_maps = update_prob_maps(org_init_prob_maps,org_init_prob_maps,updated_probs, updated_indices, updated_acc, thre_acc)
            elif iterate!=0 and idx==0:
                updated_init_prob_maps = org_init_prob_maps
                for i, idx_ in enumerate(init_prob_idx):
                    updated_init_prob_maps[idx_] = init_prob_maps[i]
                updated_prob_maps = update_prob_maps(updated_init_prob_maps,updated_init_prob_maps,updated_probs, updated_indices, updated_acc, thre_acc)
            else:
                updated_init_prob_maps = org_init_prob_maps
                for i, idx_ in enumerate(init_prob_idx):
                    updated_init_prob_maps[idx_] = init_prob_maps[i]
                updated_prev_prob_maps = org_init_prob_maps
                if prev_thre_candidates_idx[idx-1] is not None:
                    for i, idx_ in enumerate(prev_thre_candidates_idx[idx-1]):
                        updated_prev_prob_maps[idx_] = prev_thre_candidates[idx-1][i]
                    print(f'idx-1:{idx-1},prev_thre_candidates_len:{len(prev_thre_candidates)},prev_thre_candidates_idx_len:{len(prev_thre_candidates_idx)}')
                    updated_prob_maps = update_prob_maps(updated_init_prob_maps,updated_prev_prob_maps,updated_probs, updated_indices, updated_acc, thre_acc)  
                else:
                    print(f"Warning: Previous candidates not available for idx {idx}. Skipping.")
                    for i, idx_ in enumerate(init_prob_idx):
                        updated_init_prob_maps[idx_] = init_prob_maps[i]
                    updated_prob_maps = update_prob_maps(updated_init_prob_maps,updated_init_prob_maps,updated_probs, updated_indices, updated_acc, thre_acc)

            if thre_acc != 0.0: # and updated_acc > thre_acc:
                probs_.append(joint_prob)
                rewards_.append(updated_acc)
                entropies_.append(H)
           
            # Sorting buffer states
            if iterate!=0 and updated_acc < thre_acc:
                failure_count += 1
                if failure_count >= args.patience:
                    break_flag = True
                    print(f"Stopping: {iterate}_{idx} consecutive times updated_acc < thre_acc.")
                    break  # Exit inner loop
            else:
                failure_count = 0
            prev_thre_candidates.append(updated_prob_maps)
            prev_thre_candidates_idx.append(updated_indices) 
            thre_candidates.append((updated_results_tr,updated_results_val,updated_prob_maps,joint_prob,updated_indices,updated_dataset,updated_acc,updated_model_state,optimizer,lr_scheduler,criterion,gnt_extract,gnt_clf))   
            print(f'Epoch {epoch+1}/{args.num_epochs},Thre_iterate {iterate+1}/{args.num_thre_iterates}, Thre_buffer_state_idx {idx + 1},tgt_dataset_length:{len(updated_dataset)},thre_accuracy: {updated_acc}') 
        # Sorting buffer states
        thre_buffer_state = sorted(thre_candidates, key=lambda x: x[-7], reverse=True)[:args.buffer_state_width]
        print(f"Top accuracy after Thre_iteration {iterate + 1}: {[x[-7] for x in thre_buffer_state]}")
        # selective_accs.append(thre_buffer_state[0][-7])

    # Return the best subdataset founding
    best_results_tr,best_results_val,best_updated_probs,best_joint_probs,best_thre_indices,best_thre_dataset,best_acc, best_thre_model_state,best_optimizer,best_scheduler,best_criterion,best_gnt_extract,best_gnt_clf = thre_buffer_state[0]
    print(f'Best_thre_acc: {best_acc}, best_thre_indices_len:{len(best_thre_indices)}') # best_updated_probs_len:{len(best_updated_probs)}, best_joint_probs_len:{len(best_joint_probs)}')
    best_thre_model_state_copy = copy.deepcopy(best_thre_model_state)
    return  best_results_tr,best_results_val,probs_,rewards_,entropies_,best_thre_indices,best_thre_dataset,best_updated_probs,best_joint_probs,best_acc,best_thre_model_state_copy,best_optimizer,best_scheduler,best_criterion,best_gnt_extract,best_gnt_clf


def Evolving_Phase(args,model_C,model_R,probs_,rewards_,entropies_,thre_model_state,thre_indices,src_tr_loader, src_val_loader, tgt_tr_dataset,
                   thre_tgt_dataset,tgt_val_dataset, thre_results_tr,thre_results_val,thre_probs,thre_joint, thre_val_acc, fold, epoch, 
                   optimizer_C=None,lr_scheduler=None, criterion=None,gnt_extract=None,gnt_clf=None
                   ):

    initial_tgt_labels = thre_tgt_dataset.labels
    thre_true_dataset = Filter_Dataset(tgt_tr_dataset, thre_indices)
    true_tgt_labels = thre_true_dataset.labels
    # consistency
    initial_consistency_thresh = consistency_check(true_tgt_labels.data.cpu().numpy(),initial_tgt_labels.data.cpu().numpy())
    print(f'tgt_data_length: {len(thre_tgt_dataset)},thre_idx_length:{len(thre_indices)}, Initial evol Consistency: {initial_consistency_thresh:.2f}%')
    init_prob_maps = thre_probs
    evol_buffer_state = [(thre_results_tr,thre_results_val,thre_probs,thre_joint,thre_tgt_dataset,thre_val_acc, thre_model_state,optimizer_C,lr_scheduler,criterion,gnt_extract,gnt_clf)] 
    evol_candidates= evol_buffer_state.copy()
    evolving_accs =[]
    failure_count=0
    break_flag = False
    for iterate in range(args.num_evol_iterates):
        if break_flag:
            break
        for idx, (results_tr,results_val,evol_probs,joint_prob,thre_dataset,evol_acc,model_state,optimizer_C,lr_scheduler,criterion,gnt_extract,gnt_clf) in enumerate(evol_buffer_state): 
            inter_labels = crossover(thre_dataset.labels, initial_tgt_labels)
            updated_tgt_labels, actions = mutation(args,inter_labels,evol_probs)
            #print(f'initial_tgt_labels: {initial_tgt_labels}, updated_tgt_labels:{updated_tgt_labels}')
            probs_tensor = torch.stack(evol_probs) if isinstance(evol_probs, list) else evol_probs
            log_probs = torch.distributions.Bernoulli(probs=probs_tensor).log_prob(actions)
            joint_prob = log_probs.sum()/len(log_probs)
            H = - (probs_tensor * torch.log(probs_tensor + 1e-10) + (1 - probs_tensor) * torch.log(1 - probs_tensor + 1e-10)).sum()/len(log_probs)
            print(f'iterate:{iterate}, idx:{idx}, evol_probs_len: {len(evol_probs)},thre_labels_len: {len(thre_dataset.labels)},inter_labels_len: {len(inter_labels)}, action_len:{len(actions)}, evol_candidates_len:{len(evol_candidates)} ')
            print(f'probs_tensor:{probs_tensor},log_probs:{log_probs},joint_prob:{repr(joint_prob)}')
            updated_dataset = thre_dataset.update_labels(updated_tgt_labels)
            # check consistency
            consistency_thresh = consistency_check(true_tgt_labels.data.cpu().numpy(),updated_tgt_labels.data.cpu().numpy())
            print(f'Evolving Consistency: {consistency_thresh:.2f}%')
            # retraining
            evol_results_tr, evol_results_val,updated_acc,updated_model_state,updated_probs = retrain_epoch(args,model_C,model_R,model_state,src_tr_loader,src_val_loader,updated_dataset,tgt_val_dataset,fold,epoch,iterate, 
                                                                                               criterion,optimizer_C,lr_scheduler,gnt_extract,gnt_clf)
            # balancing probs_map
            updated_prob_maps = update_label_maps(init_prob_maps,thre_probs,updated_probs, thre_dataset.labels,updated_tgt_labels, updated_acc, evol_acc)  
            
            #if updated_acc >evol_acc:
            probs_.append(joint_prob)
            rewards_.append(updated_acc)  
            entropies_.append(H)
            
            if updated_acc < evol_acc:
                failure_count += 1
                if failure_count >= args.patience:
                    break_flag = True
                    print(f"Stopping: {failure_count} consecutive times updated_acc < evol_acc.")
                    break  # Exit inner loop
            else:
                failure_count = 0
            evol_candidates.append((evol_results_tr,evol_results_val,updated_prob_maps,joint_prob,updated_dataset,updated_acc,updated_model_state,optimizer_C,lr_scheduler,criterion,gnt_extract,gnt_clf))
            print(f'Epoch {epoch+1}/{args.num_epochs},Evol_iterate {iterate+1}/{args.num_evol_iterates}, Evol_idx {idx + 1}, prev_accuracy: {evol_buffer_state[idx-1][-7]},evol_accuracy: {updated_acc}')     
        # Sort candidates by accuracy and keep the top buffer_state_width candidates
        evol_buffer_state = sorted(evol_candidates, key=lambda x: x[-7], reverse=True)[:args.buffer_state_width]
        print(f"Top accuracy after Evol_iteration {iterate + 1}: {[x[-7] for x in evol_buffer_state]}")   
        evolving_accs.append(evol_buffer_state[0][-7])
    # Return the best subdataset founding
    best_results_tr,best_results_val,best_evol_probs,best_joint_probs,_,best_acc,best_model_state,_,_,_,_,_ = evol_buffer_state[0]
    print(f'Best_evol_acc:{best_acc}')
    with open(opj(args.plotting_dir, f"model_fold{fold+1}_evolving_accs_{args.continual_type}_{args.probs_mapping}_{args.reinitial}_{args.network}_{args.tgt_img_file}.pkl"), "wb") as f:
        pickle.dump(evolving_accs, f)    
    model_C.load_state_dict(best_model_state)
    return best_results_tr,best_results_val,probs_,rewards_,entropies_,best_acc, model_C

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

class DebugLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    def forward(self, x):
        out = self.layer(x)
        print(f"Activation mean: {out.mean().item()}, std: {out.std().item()}")
        return out

def train_epoch(args,models,src_tr_loader, src_val_loader, tgt_tr_dataset, tgt_val_dataset,tgt_tr_preds, init_acc,fold):
    pseudo_tgt_tr_dataset = tgt_tr_dataset.update_labels(tgt_tr_preds)
    consistency_thresh = consistency_check(tgt_tr_dataset.labels.data.cpu().numpy(), pseudo_tgt_tr_dataset.labels.data.cpu().numpy())
    print(f'Pretrain Initial Consistency: {consistency_thresh:.2f}%') 
    optimizers, lr_scheduler,criterion, gnt_extract,gnt_clf = reinitialize_retrain(args, models, src_tr_loader) ##
    out_candidates =[]
    print("Start training")
    start_time = time.time()
    models['R'].apply(init_weights)
    for epoch in range(args.num_epochs):
        models['R'].train()
        '''Screening learning processes'''
        thre_results_tr,thre_results_val,probs_,rewards_,entropies_,best_thre_indices,best_thre_dataset,best_thre_probs,best_joint_probs,thre_val_acc,best_thre_model_state,best_optimizer,best_scheduler,best_criterion, best_gnt_extract,best_gnt_clf = Screening_Phase(
            args,models['C'],models['R'],src_tr_loader, src_val_loader,tgt_tr_dataset,
            tgt_val_dataset,tgt_tr_preds,fold, epoch, 
            optimizers,lr_scheduler,criterion, gnt_extract,gnt_clf)

        '''Evolving learning processes'''
        evol_results_tr,evol_results_val,probs,rewards,entropies,evol_val_acc,evol_model = Evolving_Phase(
            args,models['C'],models['R'],probs_,rewards_,entropies_,best_thre_model_state,best_thre_indices,src_tr_loader, src_val_loader,tgt_tr_dataset,
            best_thre_dataset,tgt_val_dataset,thre_results_tr,thre_results_val, best_thre_probs, best_joint_probs,thre_val_acc,fold, epoch, 
            best_optimizer,best_scheduler,best_criterion, best_gnt_extract,best_gnt_clf)

        '''Perform the policy gradient update''' 
        returns = compute_returns(rewards)
        rl_loss = compute_policy_loss(probs,returns,entropies)
        optimizers['R'].zero_grad()
        rl_loss.backward() #rl_loss
        torch.nn.utils.clip_grad_norm_(models['R'].parameters(), max_norm=10.0)
        for name, param in models['R'].named_parameters():
            print(f"Gradient norm for {name}: {param.grad.norm().item()}")
        optimizers['R'].step()
        
        
        out_candidates.append((evol_results_tr,evol_results_val, thre_results_tr,thre_results_val,evol_val_acc, evol_model.state_dict()))
        out_candidates.sort(key=lambda x: x[-2], reverse=True)
        print(f"Top accuracy after out_iteration {epoch + 1}: {[x[-2] for x in out_candidates]}")

    '''output'''
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total CRL or RL Training time {total_time_str}")
    best_evol_results_tr,best_evol_results_val,best_thre_results_tr,best_thre_results_val, best_acc, best_model_state = out_candidates[0]
    models['C'].load_state_dict(best_model_state)
    print('final_best_accuracy:', evol_val_acc)
    # Save the model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    try:
        torch.save(models['C'].state_dict(), os.path.join(args.save_dir, f"model_fold{fold + 1}_epoch{epoch + 1}_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.tgt_img_file}.pth"))
        print(f"Model saved successfully")
    except Exception as e:
        print(f"Error occurred while saving model: {e}")
    return best_evol_results_tr,best_evol_results_val,best_thre_results_tr,best_thre_results_val, models['C']
