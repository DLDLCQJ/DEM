from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from scipy import stats

from data.data_processing import MyDataset, load_test_data
from scripts.pretrained import Loading_pretrained
from scripts.meta_learning import CRL_Meta_extractor
from scripts.test import test_epoch, run_test_epoch
from utils.metrics import *

def inference(args,src_tr_loader,data_type):
    # Define the class labels
    class_labels = ['IC_improved', 'non_IC_improved']
    confusion_matrices=[]
    fold_sen, fold_spe, fold_auc, fold_acc =[],[],[],[]
    #fold_truths , fold_preds = [], []
    import glob
    for fold in range(args.fold):
        # Load the pretrained model
        model = CRL_Meta_extractor(args.network,
                            args.num_classes,
                            replacement_rate=args.replacement_rate,
                            init=args.init,
                            maturity_threshold=args.maturity_threshold,
                            pretrained=args.pretrained,
                            frozen= args.frozen,
                            )
        ##
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        else:
            model = model.to(args.device)
        ##
        model_path = glob.glob(opj(args.save_dir, f'model_fold{fold+1}_epoch*_{args.network}_{args.src_img_file}.pth'))
        if not model_path:
            raise FileNotFoundError(f"No model found for fold {fold+1}")
        print(type(model_path),model_path)
        model.load_state_dict(torch.load(model_path[0]),strict=False)
        # Load and preprocess the image
        X, y = load_test_data(args,data_type)
        test_dataset = MyDataset(args.norm_type,X, y)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        # Perform inference
        with torch.no_grad():
            predicts,truths = run_test_epoch(args, model, src_tr_loader, test_loader)

        predicts_ = (1 / (1 + np.exp(-np.array(predicts))) > 0.5).astype(np.float32)
        labels = np.array(truths)
        predicts_ = np.array(predicts_)

        # Compute confusion matrix
        cm = confusion_matrix(labels, predicts_,labels=[0, 1])
        confusion_matrices.append(cm)
        # Print the confusion matrix for the fold
        print(f"Confusion Matrix for Fold {fold}:")
        print(cm)
        # Print other evaluation indexs
        scores_dict = compute_scores(labels, predicts_)
        print ("Sensitivity = {}".format(scores_dict['sensitivity']))
        print ("Specificity = {}".format(scores_dict['specificity']))
        print ("ROC_AUC = {}".format(scores_dict['roc_auc']))
        print("Accuracy = {}".format(scores_dict['accuracy']))
        ##
        fold_sen.append(scores_dict['sensitivity'])
        fold_spe.append(scores_dict['specificity'])
        fold_auc.append(scores_dict['roc_auc'])
        fold_acc.append(scores_dict['accuracy'])
        metrics = { 'sensitivity': fold_sen, 'specificity': fold_spe,'roc_auc': fold_auc,'accuracy': fold_acc}
    return metrics    


 
