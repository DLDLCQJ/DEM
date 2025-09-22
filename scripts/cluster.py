import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment
import torch.optim as optim
import copy

from utils.others import custom_iter
from utils.loss import Coral_loss, Cross_Contrastive_loss,SelfAttn_Contrastive_loss,Self_Contrastive_loss
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from utils.others import get_dataloader_labels,update_labels_in_subset
from utils.others import pad_dataset,custom_iter,pad_batch,consistency_check
from utils.loss import ClustersEntropyLoss,ClustersConsistencyLoss,TemporalConsistencyLoss



def run_cluster_epoch(args, model, src_dataset, tgt_dataset, fold, epoch=None, criterion=None, optimizer=None, lr_scheduler=None, mode='train'):
    model.train() if mode=='train' else model.eval()


    # Dataloader
    src_loader = DataLoader(src_dataset, batch_size = args.batch_size)
    tgt_loader = DataLoader(tgt_dataset, batch_size = args.batch_size)
    ## criterion
    consistency_criterion = ClustersConsistencyLoss()
    #consistency_criterion = TemporalConsistencyLoss()
    entropy_criterion = ClustersEntropyLoss(entropy_loss_weight=5)
    predictions, trues =[],[]
    epoch_loss, total, epoch_correct = 0, 0, 0
    len_dataloader = min(len(src_loader), len(tgt_loader))
    for i, ((src_imgs, src_labels),(tgt_imgs, tgt_labels)) in enumerate(zip(src_loader,tgt_loader)):
        src_imgs,src_labels = src_imgs.to(args.device), src_labels.to(args.device)
        tgt_imgs,tgt_labels = tgt_imgs.to(args.device), tgt_labels.to(args.device)
        #true_src_labels = true_src_labels.to(args.device)
        src_dlabels = torch.zeros(len(src_imgs)).to(args.device)
        tgt_dlabels = torch.ones(len(tgt_imgs)).to(args.device)
        ## calculating alpha
        p = float(i + epoch * len_dataloader) / (args.num_epochs_cluster * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        preds, domains, combined_feats,src_feats_con,tgt_feats_con, src_feats_adv, tgt_feats_adv = model(src_imgs, tgt_imgs,alpha)
        if mode=='train':
            #src_feats, tgt_feats = combined_feats[:-len(tgt_imgs)],combined_feats[-len(tgt_imgs):]
            # distance_matrix = torch.cdist(src_feats,tgt_feats, p=2)
            # row_ind, col_ind = linear_sum_assignment(distance_matrix.cpu().detach().numpy())
            # # pseudo_tgt_labels = torch.zeros(len(tgt_imgs), dtype=torch.float32, device=args.device)
            # # for i, col in enumerate(col_ind):
            # #     pseudo_tgt_labels[col] = src_labels[row_ind[i]]
            # src_labels =labels_distillation(src_labels)
            # tgt_labels =labels_distillation(tgt_labels)
            combined_dlabels = torch.cat((src_dlabels, tgt_dlabels),dim=0)
            dlabels = combined_dlabels.to(torch.float32)
            combined_labels = torch.cat((src_labels, tgt_labels),dim=0)
            labels = combined_labels.to(torch.float32)
            outputs = preds.squeeze(1)
            doutputs = domains.squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            loss_cls = criterion(outputs, labels)
            loss_domain = criterion(doutputs, dlabels)
            loss_consistency = consistency_criterion(preds) ##temporal
            loss_entropy = entropy_criterion(preds)
            # coral_loss = Coral_loss(src_feats_adv, tgt_feats_adv)
            # con_loss = Cross_Contrastive_loss(src_feats_con, tgt_feats_con, src_labels,tgt_labels)
            loss = loss_cls + loss_entropy +loss_consistency + loss_domain #+ coral_loss + con_loss
            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()*labels.size(0)
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()        # Update network parameters
                lr_scheduler.step()     # Update learning rate
                optimizer.zero_grad()   # Reset gradient
        else:
            
            labels = tgt_labels.to(torch.float32)
            outputs = preds[-len(tgt_labels):].squeeze(1)
            predicts = (torch.sigmoid(outputs) > 0.5).float()
            cls_loss = criterion(outputs, labels)
            epoch_loss += cls_loss.item()*labels.size(0)

        # Print statistics
        total += labels.size(0)
        epoch_correct += (predicts == labels).sum().item()
        predictions.extend(outputs.detach().cpu().numpy())
        trues.extend(labels.cpu().numpy())
        
        if mode=='train':
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Tr_Loss: {epoch_loss:.4f}, Tr_Acc: {epoch_correct / total:.4f}")
        else:
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Val_Loss: {epoch_loss:.4f}, Val_Acc: {epoch_correct / total:.4f}")
        print(50*"--")
    return epoch_loss, predictions, trues


def labels_distillation(args, pseudo_labels):
    pseudo_labels_distilled = (1 - args.epsilon_labels) * pseudo_labels + args.epsilon_labels / args.num_classes
    return pseudo_labels_distilled
# def initialize_clustering(args, model, src_data, tgt_data, n_clusters=2):
#     imgs=[]
#     for i in range(len(src_data)):
#         img, label = src_data[i]
#         imgs.append(img)
#     src_imgs = torch.stack(imgs).to(args.device)
#     model.eval()
#     combined_feats,src_feats_con,tgt_feats_con,src_feats_adv, tgt_feats_adv = model.extract_feats(src_imgs,tgt_data.imgs.to(args.device))
   
#     kmeans = KMeans(n_clusters=n_clusters, n_init=10)
#     combined_labels = kmeans.fit_predict(combined_feats.data.cpu().numpy())
    
#     pseudo_src_labels = combined_labels[:len(src_data)]
#     pseudo_tgt_labels = combined_labels[len(src_data):]
    
#     return pseudo_src_labels, pseudo_tgt_labels

def initial_clustering(args, model, src_imgs, tgt_imgs,n_clusters):
    model.eval()
    combined_feats, src_feats_con,tgt_feats_con,src_feats_adv,tgt_feats_adv = model.extract_feats(src_imgs,tgt_imgs)
    src_feats, tgt_feats = combined_feats[:-len(tgt_imgs)],combined_feats[-len(tgt_imgs):]
    # Re-cluster based on updated model features
    combined_feats = np.vstack((src_feats.data.cpu().numpy(), tgt_feats.data.cpu().numpy()))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    new_combined_labels = kmeans.fit_predict(combined_feats)
    # Split new combined labels back into source and target
    new_combined_labels = torch.tensor(new_combined_labels, dtype=torch.float32, device=args.device)  
    new_pseudo_src_labels = new_combined_labels[:-len(tgt_imgs)]
    new_pseudo_tgt_labels = new_combined_labels[-len(tgt_imgs):]
    return new_pseudo_src_labels, new_pseudo_tgt_labels


# def assign_clustering(args, model, src_imgs, tgt_imgs,src_labels,n_clusters=2):
#     combined_feats,src_feats_con,tgt_feats_con,src_feats_adv, tgt_feats_adv = model.extract_feats(src_imgs,tgt_imgs)
#     src_feats, tgt_feats = combined_feats[:-len(tgt_imgs)],combined_feats[-len(tgt_imgs):]
#     assert len(src_feats) == len(tgt_feats), 'sample length must be the same'

#     # ### Re-cluster based on updated model features
#     kmeans = KMeans(n_clusters=n_clusters, n_init=10)
#     tgt_feats_np = tgt_feats.cpu().detach().numpy()
#     new_pseudo_tgt_labels = kmeans.fit_predict(tgt_feats_np)
#     new_pseudo_tgt_labels_tensor = torch.tensor(new_pseudo_tgt_labels, dtype=torch.int64, device=args.device)

#     ### Hungarian algorithm
#     distance_matrix = torch.cdist(src_feats,tgt_feats, p=2)
#     row_ind, col_ind = linear_sum_assignment(distance_matrix.cpu().detach().numpy())
#     #mapped_labels = torch.zeros(len(tgt_imgs), dtype=torch.float32, device=args.device)
#     mapped_labels =new_pseudo_tgt_labels_tensor
#     for i, col in enumerate(col_ind):
#         mapped_labels[col] = src_labels[row_ind[i]]
#     # Combine the assigned labels with the new pseudo labels to finalize target labels
#     #final_tgt_labels = torch.where(new_pseudo_tgt_labels_tensor == mapped_labels, mapped_labels, new_pseudo_tgt_labels_tensor)
#     #new_tgt_labels = torch.tensor(mapped_labels, dtype=torch.float32, device=args.device)  
    
#     return mapped_labels


# def recursive_clustering(args, src_data, tgt_data, model,epoch,fold,optimizer=None,lr_scheduler=None,n_clusters=2):
#     src_imgs=[]
#     for i in range(len(src_data)):
#         img, label = src_data[i]
#         src_imgs.append(img)
#     src_imgs = torch.stack(src_imgs).to(args.device)
#     tgt_imgs = tgt_data.imgs.to(args.device)
#     pseudo_tgt_labels,pseudo_src_labels = update_clustering(args, model, src_imgs, tgt_imgs,n_clusters=n_clusters) 
#     ## Consistency
#     tgt_tr_labels = tgt_data.labels.data.cpu().numpy()
#     current_consistency_tgt = consistency_check(tgt_tr_labels, pseudo_tgt_labels.cpu().detach().numpy())
#     print(f'Epoch {epoch+1}/{args.num_epochs_pre}, Target Consistency: {current_consistency_tgt:.2f}%')
#     ## Updating labels
#     tgt_tr_dataset_update = tgt_data.update_labels(pseudo_tgt_labels)
#     src_tr_dataset_update = update_labels_in_subset(src_data, pseudo_src_labels) 
#     ## criterion
#     cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
#     consistency_criterion = ClustersConsistencyLoss()
#     entropy_criterion = ClustersEntropyLoss(entropy_loss_weight=5)
#     epoch_loss, total, epoch_correct = 0, 0, 0
#     outputs, labels = run_cluster_epoch(args, model, src_tr_dataset_update,tgt_tr_dataset_update)
#     predictions = outputs.unsqueeze(1)
#     predicts = (torch.sigmoid(outputs) > 0.5).float()
#     loss_cls = cls_criterion(outputs, labels)
#     loss_consistency = consistency_criterion(predictions)
#     loss_entropy = entropy_criterion(predictions)
#     loss = loss_cls + loss_consistency + loss_entropy
#     total = labels.size(0)
#     epoch_correct += (predicts == labels).sum().item()
#     epoch_loss += loss.item()
#     loss.backward()
#     optimizer.step()        # Update network parameters
#     lr_scheduler.step()     # Update learning rate
#     optimizer.zero_grad()   # Reset gradient
#     print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Tr_Loss: {epoch_loss:.4f}, Tr_Acc: {epoch_correct / total:.4f}")
 
def Cluster_train_epoch(args, model,  tgt_tr_dataset,tgt_val_dataset, src_tr_loader,fold,criterion=None,optimizer=None, lr_scheduler=None, n_clusters=2):
    best_loss = float('inf')

    for epoch in range(args.num_epochs_cluster):
        print(f"Starting Clustering Epoch {epoch+1}")
        src_imgs=[]
        for i in range(len(src_tr_loader.dataset)):
            img, label = src_tr_loader.dataset[i]
            src_imgs.append(img)
        src_imgs = torch.stack(src_imgs).to(args.device)
        tgt_tr_imgs = tgt_tr_dataset.imgs.to(args.device)
        tgt_val_imgs = tgt_val_dataset.imgs.to(args.device)
        tgt_imgs = torch.cat((tgt_tr_imgs,tgt_val_imgs),dim=0)
        if epoch %args.clustering_step== 0:
            pseudo_src_labels,pseudo_tgt_labels = initial_clustering(args, model, src_imgs, tgt_imgs,n_clusters=n_clusters) 
            pseudo_tgt_tr_labels, pseudo_tgt_val_labels = pseudo_tgt_labels[:-len(tgt_val_dataset)],pseudo_tgt_labels[-len(tgt_val_dataset):]
        ## Updating labels
        tgt_tr_dataset_update = tgt_tr_dataset.update_labels(pseudo_tgt_tr_labels)
        tgt_val_dataset_update = tgt_val_dataset.update_labels(pseudo_tgt_val_labels)
        src_tr_dataset_update = update_labels_in_subset(src_tr_loader.dataset, pseudo_src_labels) 
        ##
        run_cluster_epoch(args,model, src_tr_dataset_update, tgt_tr_dataset_update,fold,epoch=epoch,criterion=criterion,optimizer=optimizer,lr_scheduler=lr_scheduler, mode='train')
        with torch.no_grad():
           val_loss, predictions, trues = run_cluster_epoch(args,model, src_tr_dataset_update, tgt_val_dataset_update,fold,epoch=epoch,criterion=criterion,optimizer=optimizer,lr_scheduler=lr_scheduler, mode='eval')

        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping triggered on fold {fold + 1}, epoch {epoch + 1}")
                break


# def run_extract_metafeats(args, model, src_loader, tgt_loader,criterion, optimizer):
#     model.train()
#     total_samples = len(src_loader.dataset) + len(tgt_loader.dataset)
#     features = None
#     predictions= []
#     for i, ((src_imgs, src_labels),(tgt_imgs, tgt_labels)) in enumerate(zip(src_loader,tgt_loader)):
#         src_imgs,src_labels = src_imgs.to(args.device), src_labels.to(args.device)
#         tgt_imgs,tgt_labels = tgt_imgs.to(args.device), tgt_labels.to(args.device)
#         preds, combined_feats, src_feats_con,tgt_feats_con, src_feats_adv, tgt_feats_adv = model(src_imgs, tgt_imgs)
#         predictions.append(preds)
#         ##
#         src_outputs = preds[:len(src_imgs)].squeeze(1)
#         cls_loss = criterion(src_outputs, src_labels)
#         coral_loss = Coral_loss(src_feats_adv, tgt_feats_adv)
#         con_loss = SelfAttn_Contrastive_loss(src_feats_con, tgt_feats_con, src_labels)
#         loss = coral_loss + con_loss + cls_loss
#         # Backpropagation
#         optimizer.zero_grad() 
#         loss.backward()
#         optimizer.step()        # Update network parameters
#         ## Extracted_feats
#         extracted_feats = combined_feats.data.cpu().numpy()
#         if features is None:
#             features = np.zeros((total_samples, extracted_feats.shape[1]), dtype='float32')

#         start_idx = i * src_loader.batch_size
#         end_idx = start_idx + extracted_feats.shape[0]
#         features[start_idx:end_idx] = extracted_feats.astype('float32')
#     # Flatten predictions and combine losses
#     predictions = torch.cat((predictions),dim=0)
   
#     return features, predictions
 
# def recursive_clustering_epoch(args, true_src_labels, src_dataset, tgt_dataset, model, criterion=None,optimizer=None,n_clusters=2):
#     src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=False)
#     tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False)

#     for epoch in range(args.num_epochs_pre):
#         model.train()
#         print(f"Starting Pretraining Epoch {epoch + 1}/{args.num_epochs}")
#         combined_features, predictions = run_extract_metafeats(args, model, src_loader, tgt_loader,criterion, optimizer)
#         # Perform clustering on combined features
#         if args.cluster_method =='kmeans':
#             kmeans = KMeans(n_clusters=n_clusters, n_init=10)
#             combined_pseudo_labels = kmeans.fit_predict(combined_features)
#         elif args.cluster_method =='spectral':
#             spectral = SpectralClustering(n_clusters=n_clusters)
#             combined_pseudo_labels = spectral.fit_predict(combined_features)
#         elif args.cluster_method =='agg':
#             agg = AgglomerativeClustering(n_clusters=n_clusters)
#             combined_pseudo_labels = agg.fit_predict(combined_features)
#         # Split pseudo labels back into source and target
#         pseudo_tgt_labels = combined_pseudo_labels[-len(tgt_dataset):]
#         # true_src_labels_tensor = true_src_labels.clone().detach().unsqueeze(1).to(args.device)
#         pseudo_tgt_labels_tensor = torch.tensor(pseudo_tgt_labels, dtype=torch.float32, device=args.device).unsqueeze(1)
#         ##
#         predicts = (torch.sigmoid(predictions) > 0.5).float()
#         tgt_predicts = predicts[-len(tgt_dataset):]
#         loss = criterion(tgt_predicts, pseudo_tgt_labels_tensor)
#         # Backpropagation
#         optimizer.zero_grad() 
#         loss.backward()
#         optimizer.step()        # Update network parameters

#     model.eval()  # Set model to evaluation mode for final feature extraction
#     final_combined_features, _ = run_extract_metafeats(args, model, src_loader, tgt_loader,criterion, optimizer)
    
#     if args.cluster_method == 'kmeans':
#         final_kmeans = KMeans(n_clusters=n_clusters, n_init=10)
#         final_pseudo_labels = final_kmeans.fit_predict(final_combined_features)
#     elif args.cluster_method == 'spectral':
#         final_spectral = SpectralClustering(n_clusters=n_clusters)
#         final_pseudo_labels = final_spectral.fit_predict(final_combined_features)
#     elif args.cluster_method == 'agg':
#         final_agg = AgglomerativeClustering(n_clusters=n_clusters)
#         final_pseudo_labels = final_agg.fit_predict(final_combined_features)

#     return final_pseudo_labels

