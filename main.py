
import os
from os.path import join as opj
from skimage.transform import resize
import pickle

from copy import deepcopy
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from sklearn.model_selection import StratifiedKFold,train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler,Subset

from data.data_processing import MyDataset, loading_data
from utils.seed import set_seed
from scripts.train import train_epoch, pretrain_epoch,reassigned_labels,reassigned_labels_alone
from scripts.cluster import Cluster_train_epoch
from scripts.test import test_epoch
from scripts.init import parse_arguments
from scripts.inference import inference
from stats.stats import stats_scores
from utils.others import get_dataloader_labels,consistency_check,get_model,get_pretrain_model
from utils.others import pad_dataset,custom_iter,pad_batch
from utils.AdamGnT import AdamGnT
 
def main(args):
    set_seed(123)
    X_src, y_src = loading_data(args.path, args.src_img_file, args.src_label_file,args.desired_shape)
    X_tgt, y_tgt = loading_data(args.path, args.tgt_img_file, args.tgt_label_file,args.desired_shape)

    ## Determine device usage
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',') 
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print(f'Use GPUs: {args.device_ids}')
    elif args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print('Use CPU')
   
    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    for fold, ((tr_index_src, val_index_src),(tr_index_tgt, val_index_tgt))in enumerate(zip(skf.split(X_src, y_src),skf.split(X_tgt, y_tgt))):
        # Set a seed for each fold for consistent sampling
        set_seed(123 + fold)
        X_tr_src, X_val_src = X_src[tr_index_src], X_src[val_index_src]
        y_tr_src, y_val_src = y_src[tr_index_src], y_src[val_index_src]
        X_tr_tgt, X_val_tgt = X_tgt[tr_index_tgt], X_tgt[val_index_tgt]
        y_tr_tgt, y_val_tgt = y_tgt[tr_index_tgt], y_tgt[val_index_tgt]
        print(f"X_train_src.shape: {X_tr_src.shape}", f"y_train_src.shape: {y_tr_src.shape}")
        print(f"X_train_tgt.shape: {X_tr_tgt.shape}", f"y_train_tgt.shape: {y_tr_tgt.shape}")
        print(f"X_val_src.shape: {X_val_src.shape}", f"y_val_src.shape: {y_val_src.shape}")
        print(f"X_val_tgt.shape: {X_val_tgt.shape}", f"y_val_tgt.shape: {y_val_tgt.shape}")

        src_tr_dataset = MyDataset(X_tr_src, y_tr_src)
        tgt_tr_dataset = MyDataset(X_tr_tgt, y_tr_tgt)
        src_val_dataset = MyDataset(X_val_src, y_val_src)
        tgt_val_dataset = MyDataset(X_val_tgt, y_val_tgt)
     
        tgt_te_loader = DataLoader(tgt_val_dataset, batch_size=args.batch_size) 
        src_val_loader = DataLoader(src_val_dataset, batch_size=args.batch_size)
        #src_te_loader = DataLoader(src_val_dataset, batch_size=args.batch_size)
    
        # Create a new sampler for each iteration
        src_tr_weights = torch.ones(len(src_tr_dataset))
        src_tr_sampler = WeightedRandomSampler(src_tr_weights, num_samples=len(tgt_tr_dataset), replacement=True)
        sample_indices = list(src_tr_sampler)
        src_tr_dataset_sub = Subset(src_tr_dataset,sample_indices)
        src_tr_loader = DataLoader(src_tr_dataset_sub, batch_size=args.batch_size,shuffle=True)
        #src_tr_labels = [src_tr_dataset.labels[idx] for idx in sample_indices]

        ### Starting
        os.makedirs(args.save_dir, exist_ok=True)
        model = get_pretrain_model(args)  
        for name, param in model.named_parameters():
            print(f"Pretrain: {name} | Requires Gradient: {param.requires_grad}")            
        criterion = nn.BCEWithLogitsLoss(reduction='mean') 
        #### Stage-1: Pretraining
        num_training_steps = len(src_tr_loader) * args.num_epochs_pretrain
        num_warmup_steps = int(num_training_steps * args.lr_scheduler_warmup_ratio)
        if args.optim == 'sgd':
            optimizer_pre = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_pre)
        elif args.optim == 'adam':
            optimizer_pre = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_pre, betas=args.betas, eps=args.epsilon)
        if args.lr_scheduler_type == 'linear':
            lr_scheduler_pre = get_scheduler( 
                name=args.lr_scheduler_type,
                optimizer=optimizer_pre,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)
        criterion = nn.BCEWithLogitsLoss(reduction='mean') 
        pretrain_epoch(args, model, src_tr_loader,src_val_loader,tgt_tr_dataset,tgt_val_dataset, fold,criterion, optimizer_pre, lr_scheduler_pre)
        # init_model_state = deepcopy(model.state_dict())
        # # Stage-2: Training 
        if args.continual_type == 'CRL': 
            tgt_tr_preds, init_acc = reassigned_labels(args,model,src_tr_loader,tgt_tr_dataset)
        elif args.continual_type == 'RL':
            tgt_tr_preds, init_acc = reassigned_labels_alone(args,model,tgt_tr_dataset)

        models = get_model(args) 
        best_evolved_results_tr,best_evolved_results_val,best_threshed_results_tr,best_threshed_results_val,best_model = train_epoch(args,models,src_tr_loader,src_val_loader,tgt_tr_dataset,tgt_val_dataset,tgt_tr_preds,init_acc,fold)                 
        # Stage-3: Testing
        src_results_te = test_epoch(args, best_model, src_loader=src_val_loader, tgt_loader=tgt_te_loader,domain='source', fold=fold)
        # Save the scores of model
        with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_evolving_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.tgt_img_file}.pkl"), "wb") as f:
            pickle.dump(best_evolved_results_val, f) #tgt_test
        with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_screening_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.tgt_img_file}.pkl"), "wb") as f:
            pickle.dump(best_threshed_results_val, f) #tgt_test
        with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_evolving_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.src_img_file}.pkl"), "wb") as f:
            pickle.dump(src_results_te, f) ## src_test

if __name__ == '__main__':
    args = parse_arguments() 
    print(args)
    src_tr_loader = main(args)
    stats_scores(args,args.src_img_file)
    stats_scores(args,args.tgt_img_file,stage='screening')
    stats_scores(args,args.tgt_img_file,stage='evolving')
    
 

 



