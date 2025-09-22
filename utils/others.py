import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler 
from copy import deepcopy
from torch.nn import functional as F
from torch.autograd import Variable

from data.data_processing import MyDataset, Filter_Dataset
from scripts.pretrained import Loading_pretrained
from scripts.meta_learning import RL_Meta_extractor,CRL_Meta_extractor,RL_model
from modules.GnT import FFGnT
from utils.AdamGnT import AdamGnT

def pad_dataset(args,dataset, batch_size):
    imgs, labels =[], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        imgs.append(img) 
        labels.append(label)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    n = len(imgs)
    padding_size = (batch_size - n % batch_size) % batch_size
    if padding_size > 0:
        imgs = torch.cat([imgs, imgs[:padding_size]], dim=0)
        labels = torch.cat([labels, labels[:padding_size]], dim=0)
    return MyDataset(args.norm_type, imgs, labels)

def custom_iter(loader):
    while True:
        for data in loader:
            yield data

def pad_batch(batch, batch_size,epoch_batch):
    imgs, labels = batch
    if epoch_batch < batch_size:
        
        imgs = imgs[:epoch_batch]
        labels = labels[:epoch_batch]
    return imgs, labels

def get_dataloader_labels(data_loader):
    labels=[]
    for _, label in data_loader:
        labels.append(label.data.cpu().numpy())
    labels = np.concatenate(labels)
    return labels

def update_labels_in_subset(subset, new_labels):
    original_dataset = subset.dataset
    indices = subset.indices
    updated_dataset = MyDataset(
        norm_type=original_dataset.norm_type,
        imgs=original_dataset.imgs[indices], 
        labels=new_labels,
        meta=original_dataset.meta if hasattr(original_dataset, 'meta') else None,
        augment=original_dataset.augment if hasattr(original_dataset, 'augment') else None
    )
    
    return updated_dataset

class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, weights):
        self.base_dataset = base_dataset
        self.weights = weights

    def __getitem__(self, index):
        data, label = self.base_dataset[index]
        return data, label, self.weights[index]

    def __len__(self):
        return len(self.base_dataset)

def consistency_check(true_labels, pseudo_labels):
    matching_labels = true_labels == pseudo_labels
    num_consistent_labels = np.sum(matching_labels)
    consistency_percentage = (num_consistent_labels / len(true_labels)) *100
    return consistency_percentage

def get_model(args):
    models = {}
    if args.continual_type=='CRL':
        models['C'] = CRL_Meta_extractor(args.network,
                            args.num_classes,
                            replacement_rate=args.replacement_rate,
                            init=args.init,
                            maturity_threshold=args.maturity_threshold,
                            pretrained=args.pretrained,
                            frozen= args.frozen,
                            )
        models['R'] = RL_model(args.num_classes)
        #models['O'] = RL_model(args.num_classes)
    elif args.continual_type=='RL':
        models['C'] = RL_Meta_extractor(args.network,
                            args.num_classes,
                            replacement_rate=args.replacement_rate,
                            init=args.init,
                            maturity_threshold=args.maturity_threshold,
                            pretrained=args.pretrained,
                            )
        models['R'] = RL_model(args.num_classes)
        #models['O'] = RL_model(args.num_classes)
    #Parallel-GPU
    if args.use_multi_gpu and args.use_gpu:
        for m in models:
            models[m] = nn.SyncBatchNorm.convert_sync_batchnorm(models[m])
            models[m] = nn.DataParallel(models[m], device_ids=args.device_ids)
            models[m] = models[m].to(args.device)
    
    else:
        for m in models:
            models[m] = models[m].to(args.device)
    return models

def get_pretrain_model(args):
    if args.continual_type=='CRL':
        model = CRL_Meta_extractor(args.network,
                            args.num_classes,
                            replacement_rate=args.replacement_rate,
                            init=args.init,
                            maturity_threshold=args.maturity_threshold,
                            pretrained=args.pretrained,
                            frozen= False,
                            )
    elif args.continual_type=='RL':
        model = RL_Meta_extractor(args.network,
                            args.num_classes,
                            replacement_rate=args.replacement_rate,
                            init=args.init,
                            maturity_threshold=args.maturity_threshold,
                            pretrained=args.pretrained,
                            )
    #Parallel-GPU
    if args.use_multi_gpu and args.use_gpu:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model, device_ids=args.device_ids)
        model = model.to(args.device)
    else:
        model = model.to(args.device)
    return model

def thresholding(confidences,dataset,iterate=None, initial_percentile=95):
    dataset_size = len(dataset)-1
    # Adjust the percentile for each iteration
    dynamic_percentile = initial_percentile - (iterate * (initial_percentile / iterate))
    if dynamic_percentile < 0:
        dynamic_percentile = 0 

    values = np.concatenate([c for c in confidences])
    threshold = np.percentile(values, dynamic_percentile)

    selected_indices = [i for i, cons in enumerate(values) if cons > threshold and i < dataset_size]
    if len(selected_indices) == 0:
            print(f"Warning: No indices selected for iteration {iterate + 1}.")
        
    threshed_data = Filter_Dataset(dataset,selected_indices)
    return threshed_data, selected_indices


def variable(t, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)
    
class EWC:
    def __init__(self, args, model, dataset, importance=1000):
        self.args = args
        self.model = model
        self.importance = importance
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._FIM = {}
        self.dataset = dataset
        # Store the means and compute Fisher Information Matrix for the initial task
        self._store_means()
        self._compute_fisher()

    def _store_means(self):
        """Store the current parameters' values (means)."""
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _compute_fisher(self):
        """Compute the Fisher Information Matrix for the dataset."""
        fisher_matrix = {n: torch.zeros_like(p).to(self.args.device) for n, p in self.params.items()}
        # Compute Fisher matrix
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        for inputs,_ in self.dataset:
            self.model.zero_grad()
            inputs = inputs.to(self.args.device)
            outputs = self.model(inputs)
            outputs = outputs.squeeze(1)
            targets = (torch.sigmoid(outputs) > 0.5).float()
            loss = criterion(outputs, targets)
            loss.backward()
            for n, p in self.params.items():
                fisher_matrix[n] += p.grad ** 2 / len(self.dataset)
        self._FIM = fisher_matrix

    def consolidate(self):
        """Consolidate the Fisher Information Matrix and parameter means."""
        for n, p in self.model.named_parameters():
            if n in self._FIM:
                # Register buffers for means and Fisher diagonals
                self.model.register_buffer(f"{n}_mean", self._means[n].to(self.args.device))
                self.model.register_buffer(f"{n}_fisher", self._FIM[n].to(self.args.device))

    def penalty(self):
        """Calculate the EWC penalty for all parameters."""
        losses = []
        for n, p in self.model.named_parameters():
            # Retrieve consolidated means and Fisher values
            mean = getattr(self.model, f"{n}_mean", None)
            fisher = getattr(self.model, f"{n}_fisher", None)
            if mean is not None and fisher is not None:

                losses.append((fisher * (p - mean) ** 2).sum())
        return (self.importance / 2) * sum(losses)


def reinitialize_retrain(args, models, loader):
    optimizers ={}
    # if args.optim == 'sgd':
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_re)
    #     # base_params = [
    #     #                     p for n, p in model.named_parameters() 
    #     #                     if not n.startswith('rl')  # Exclude policy parameters
    #     #                     and p.requires_grad  # Include only trainable parameters
    #     #                 ]
    #     # optimizer = optim.SGD( base_params, lr=args.lr_re)
    # elif args.optim == 'adam':
    if args.reinitial == 'cbp':
        optimizers['C'] = AdamGnT(filter(lambda p: p.requires_grad, models['C'].parameters()), lr=args.lr_re, betas=args.betas, weight_decay=args.weight_decay) 
    optimizers['R'] = optim.Adam(filter(lambda p: p.requires_grad, models['R'].parameters()), lr=args.lr_rl, betas=args.betas,weight_decay=args.weight_decay)
    #optimizers['E'] = optim.Adam(filter(lambda p: p.requires_grad, models['E'].parameters()), lr=args.lr_re, betas=args.betas, eps=args.epsilon)
        # base_params = [
        #                 p for n, p in model.named_parameters() 
        #                 if not n.startswith('rl')  # Exclude policy parameters
        #                 and p.requires_grad  # Include only trainable parameters
        #             ]
        # optimizer = AdamGnT(
        #     base_params,  # Only non-policy parameters
        #     lr=args.lr_re,
        #     betas=args.betas,
        #     weight_decay=args.weight_decay
        # ) 
       
    num_training_steps = len(loader) * args.num_epochs_train
    num_warmup_steps = int(num_training_steps * args.lr_scheduler_warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizers['C'],
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    ) if args.lr_scheduler_type == 'linear' else None
    criterion = nn.BCEWithLogitsLoss(reduction='mean') 
    if args.use_multi_gpu:
        gnt_extract = FFGnT(
            net=models['C'].module.extractor_cbp.layers,
            hidden_activation='relu',
            opt = optimizers['C'],
            replacement_rate=args.replacement_rate,
            decay_rate=args.decay_rate,
            init=args.init,
            util_type=args.util_type,
            maturity_threshold=args.maturity_threshold,
            device=args.device
        )
        gnt_clf = FFGnT(
            net=models['C'].module.clf_cbp.layers,
            hidden_activation='relu',
            opt = optimizers['C'],
            replacement_rate=args.replacement_rate,
            decay_rate=args.decay_rate,
            init=args.init,
            util_type=args.util_type,
            maturity_threshold=args.maturity_threshold,
            device=args.device
        )
    else:
        gnt_extract = FFGnT(
            net=models['C'].extractor_cbp.layers,
            hidden_activation='relu',
            opt = optimizers['C'],
            replacement_rate=args.replacement_rate,
            decay_rate=args.decay_rate,
            init=args.init,
            util_type=args.util_type,
            maturity_threshold=args.maturity_threshold,
            device=args.device
        )

        gnt_clf = FFGnT(
            net=models['C'].clf_cbp.layers,
            hidden_activation='relu',
            opt = optimizers['C'],
            replacement_rate=args.replacement_rate,
            decay_rate=args.decay_rate,
            init=args.init,
            util_type=args.util_type,
            maturity_threshold=args.maturity_threshold,
            device=args.device
        )
        
    return optimizers, lr_scheduler, criterion,gnt_extract, gnt_clf 

def initialize_retrain_alone(args, model, loader):
    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_re)
    elif args.optim == 'adam':
        if args.reinitial == 'cbp':
            optimizer = AdamGnT(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_re, betas=args.betas, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_re, betas=args.betas, eps=args.epsilon)
                

    num_training_steps = len(loader) * args.num_epochs_train
    num_warmup_steps = int(num_training_steps * args.lr_scheduler_warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    ) if args.lr_scheduler_type == 'linear' else None
    criterion = nn.BCEWithLogitsLoss(reduction='mean') 
    return optimizer, lr_scheduler, criterion 
