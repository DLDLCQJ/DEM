
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
 
import os
import builtins
import pickle
from os.path import join as opj
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import StratifiedKFold

from others import set_seed, get_scheduler
from processing import MyDataset, loading_data
from models import get_pretrain_model, get_model 
from scripts import pretrain_epoch, reassigned_labels, reassigned_labels_alone, train_epoch, test_epoch, stats_scores
from init import parse_arguments

def main(args):
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    per_gpu_batch_size = args.batch_size // args.world_size
    
    if args.distributed:
        if args.local_rank != -1:  
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ: 
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
            
        torch.cuda.set_device(args.gpu)
        args.device = torch.device('cuda', args.gpu)
        
        dist.init_process_group(backend='nccl', init_method='env://', 
                                world_size=args.world_size, rank=args.rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using Single Device: {args.device}')
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(f"[Rank {args.rank}] Initialized. Global Batch: {args.batch_size}, Per-GPU: {per_gpu_batch_size}")
    
    # Set global seed
    set_seed(123 + args.rank)
    X_src, y_src = loading_data(args.path, args.src_img_file, args.src_label_file, args.desired_shape)
    X_tgt, y_tgt = loading_data(args.path, args.tgt_img_file, args.tgt_label_file, args.desired_shape)

    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    for fold, ((tr_index_src, val_index_src), (tr_index_tgt, val_index_tgt)) in enumerate(zip(skf.split(X_src, y_src), skf.split(X_tgt, y_tgt))):
        if args.rank == 0:
            print(f"--- Starting Fold {fold+1}/{args.fold} ---")         
        # Set a seed for each fold
        set_seed(123 + fold)
        X_tr_src, X_val_src = X_src[tr_index_src], X_src[val_index_src]
        y_tr_src, y_val_src = y_src[tr_index_src], y_src[val_index_src]
        X_tr_tgt, X_val_tgt = X_tgt[tr_index_tgt], X_tgt[val_index_tgt]
        y_tr_tgt, y_val_tgt = y_tgt[tr_index_tgt], y_tgt[val_index_tgt]

        if args.rank == 0:
            print(f"X_train_src: {X_tr_src.shape}, y_train_src: {y_tr_src.shape}")
            print(f"X_train_tgt: {X_tr_tgt.shape}, y_train_tgt: {y_tr_tgt.shape}")

        src_tr_dataset = MyDataset(X_tr_src, y_tr_src)
        tgt_tr_dataset = MyDataset(X_tr_tgt, y_tr_tgt)
        src_val_dataset = MyDataset(X_val_src, y_val_src)
        tgt_val_dataset = MyDataset(X_val_tgt, y_val_tgt)
        src_tr_weights = torch.ones(len(src_tr_dataset))
        g = torch.Generator()
        g.manual_seed(123 + fold) 
        sample_indices = torch.multinomial(src_tr_weights, len(tgt_tr_dataset), replacement=True, generator=g).tolist()
        src_tr_dataset_sub = Subset(src_tr_dataset, sample_indices)
        src_tr_sampler = DistributedSampler(src_tr_dataset_sub, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        tgt_tr_sampler = DistributedSampler(tgt_tr_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True) # Assuming shuffle=True for train
        src_val_sampler = DistributedSampler(src_val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        tgt_val_sampler = DistributedSampler(tgt_val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        src_tr_loader = DataLoader(src_tr_dataset_sub, batch_size=per_gpu_batch_size, sampler=src_tr_sampler, num_workers=args.num_workers, pin_memory=True)
        tgt_tr_loader = DataLoader(tgt_tr_dataset, batch_size=per_gpu_batch_size, sampler=tgt_tr_sampler, num_workers=args.num_workers, pin_memory=True) 
        
        src_val_loader = DataLoader(src_val_dataset, batch_size=per_gpu_batch_size, sampler=src_val_sampler, num_workers=args.num_workers, pin_memory=True)
        tgt_te_loader  = DataLoader(tgt_val_dataset, batch_size=per_gpu_batch_size, sampler=tgt_val_sampler, num_workers=args.num_workers, pin_memory=True)

        if args.rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
        model = get_pretrain_model(args)
        model = model.to(args.device)
        if args.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
        
        if args.rank == 0:
            for name, param in model.named_parameters():
                print(f"Pretrain: {name} | Requires Gradient: {param.requires_grad}")


        if args.optim == 'sgd':
            optimizer_pre = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_pre)
        elif args.optim == 'adam':
            optimizer_pre = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_pre, betas=args.betas, eps=args.epsilon)
        
        num_training_steps = len(src_tr_loader) * args.num_epochs_pretrain
        num_warmup_steps = int(num_training_steps * args.lr_scheduler_warmup_ratio)
        
        if args.lr_scheduler_type == 'linear':
            lr_scheduler_pre = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer_pre,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)
        
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        pretrain_epoch(args, model, src_tr_loader, src_val_loader, tgt_tr_dataset, tgt_val_dataset, fold, criterion, optimizer_pre, lr_scheduler_pre)

        if args.continual_type == 'CRL': 
            tgt_tr_preds, init_acc = reassigned_labels(args, model, src_tr_loader, tgt_tr_dataset)
        elif args.continual_type == 'RL':
            tgt_tr_preds, init_acc = reassigned_labels_alone(args, model, tgt_tr_dataset)

        models = get_model(args)
        models = models.to(args.device)
        
        if args.distributed:
            models = nn.SyncBatchNorm.convert_sync_batchnorm(models)
            models = DDP(models, device_ids=[args.gpu], output_device=args.gpu)
        
        # Run Training
        best_evolved_results_tr, best_evolved_results_val, best_threshed_results_tr, best_threshed_results_val, best_model = train_epoch(
            args, models, src_tr_loader, src_val_loader, tgt_tr_dataset, tgt_val_dataset, tgt_tr_preds, init_acc, fold
        )

        src_results_te = test_epoch(args, best_model, src_loader=src_val_loader, tgt_loader=tgt_te_loader, domain='source', fold=fold)
        
        if args.rank == 0:
            print(f"Saving results for fold {fold+1}...")
            with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_evolving_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.tgt_img_file}.pkl"), "wb") as f:
                pickle.dump(best_evolved_results_val, f)
            with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_screening_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.tgt_img_file}.pkl"), "wb") as f:
                pickle.dump(best_threshed_results_val, f)
            with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_evolving_te_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{args.src_img_file}.pkl"), "wb") as f:
                pickle.dump(src_results_te, f)

        if args.distributed:
            dist.barrier()
    if args.distributed:
        dist.destroy_process_group()
        
    return src_tr_loader 

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    
 

 



