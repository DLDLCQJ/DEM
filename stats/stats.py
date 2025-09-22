## train_test_results
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix,roc_curve, auc
from scipy import stats

from scripts.init import parse_arguments
from utils.metrics import * 

def stats_scores(args,img_file,state='te',stage='evolving'):

    results =[]
    for i in range(args.fold):
        with open(os.path.join(args.save_dir,f"model_fold{i+1}_cv_predicts_and_labels_{stage}_{state}_{args.continual_type}_{args.reinitial}_{args.probs_mapping}_{args.adaptive}_{args.network}_{img_file}.pkl"), "rb") as f:
            result = pickle.load(f)
        results.append(result)

    ## Iterate over the results to compute confusion matrices for each fold
    confusion_matrices = []
    fold_sen, fold_spe, fold_auc, fold_acc =[],[],[],[]
    # truths , preds = [], []
    for fold_result in results: #5-fold
        
        predictions = fold_result[-1]["predictions"] 
        # predictions = [tensor.detach().cpu().numpy() for tensor in predictions]
        # predictions = np.concatenate(predictions).tolist()
        #predictions_= (np.array(predictions) > 0.5).astype(np.float32)
        predictions_ = (1 / (1 + np.exp(-np.array(predictions))) > 0.5).astype(np.float32)
        labels = fold_result[-1]["labels"]
        # Compute confusion matrix
        # labels = [tensor.detach().cpu().numpy() for tensor in labels]
        # labels = np.concatenate(labels).tolist()
        labels = np.array(labels)
        #print(labels.shape, predictions_.shape)
        # preds.extend(predictions_)
        # truths.extend(labels)
        
        cm = confusion_matrix(labels, predictions_,labels=[0, 1])
        confusion_matrices.append(cm)
        # Print the confusion matrix for the fold
        print(50*"--")
        print(f"Confusion Matrix for Fold {fold_result[-1]['fold']}:")
        print(cm)
        # Print other evaluation indexs
        scores_dict = compute_scores(labels, predictions_)
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

    # CI  
    ci_results = {}
    for metric_name, scores in metrics.items():
        mean_score = np.mean(scores)
        std_err = stats.sem(scores)
        h = std_err * stats.t.ppf((1 + 0.95) / 2, 5 - 1)
        ci_results[metric_name] = (mean_score, mean_score - h, mean_score + h)
    print(50*"**")
    print("CIs test_results for over all folds:")
    print ("Sensitivity = {}".format(ci_results['sensitivity']))
    print ("Specificity = {}".format(ci_results['specificity']))
    print ("ROC_AUC = {}".format(ci_results['roc_auc']))
    print("Accuracy = {}".format(ci_results['accuracy']))
    print(50*"**")

    # ## Optional: Compute the mean confusion matrix over all folds
    # mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    # print("Mean Confusion Matrix over all folds:")
    # print(mean_confusion_matrix)
    # # Print other evaluation indexs
    # tn, fp, fn, tp = mean_confusion_matrix.ravel()
    # sen = tp / (tp + fn)
    # spe = tn / (fp + tn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # F1score = 2 * (precision * sen) / (precision + sen)
    # fpr, tpr, thresholds = roc_curve(labels, predictions)
    # roc_auc = auc(fpr, tpr)
    # acc = (tp+tn) / (tp + tn + fp + fn)
    # # pos_num = (labels==1).sum().item()
    # # neg_num = (labels==0).sum().item()
    # print ("Sensitivity = {}".format(sen))
    # print ("Specificity = {}".format(spe))
    # print ("Precision = {}".format(precision))
    # print ("F1score = {}".format(F1score))
    # print ("Recall = {}".format(recall))
    # print ("ROC_AUC = {}".format(roc_auc))
    # print("Accuracy = {}".format(acc))
    # print(50*"--")

# def stats_test(args,img_file):

#     results =[]
#     for i in range(args.fold):
#         with open(os.path.join(args.save_dir,f"model_fold{i+1}_cv_predicts_and_labels_te_{args.network}_{img_file}.pkl"), "rb") as f:
#             result = pickle.load(f)
#         results.append(result)

#     ## Iterate over the results to compute confusion matrices for each fold
#     confusion_matrices = []
#     fold_sen, fold_spe, fold_auc, fold_acc =[],[],[],[]
#     # truths , preds = [], []
#     for fold_result in results: #5-fold
#         predictions = fold_result[-1]["predictions"] 
#         print('test_predictions:',fold_result[-1]["predictions"])
#         # predictions = [tensor.detach().cpu().numpy() for tensor in predictions]
#         # predictions = np.concatenate(predictions).tolist()
#         #predictions_= (np.array(predictions) > 0.5).astype(np.float32)
#         predictions_ = (1 / (1 + np.exp(-np.array(predictions))) > 0.5).astype(np.float32)
#         labels = fold_result[-1]["labels"]
#         # Compute confusion matrix
#         # labels = [tensor.detach().cpu().numpy() for tensor in labels]
#         # labels = np.concatenate(labels).tolist()
#         labels = np.array(labels)
#         #print(labels.shape, predictions_.shape)
#         # preds.extend(predictions_)
#         # truths.extend(labels)
        
#         cm = confusion_matrix(labels, predictions_,labels=[0, 1])
#         confusion_matrices.append(cm)
#         # Print the confusion matrix for the fold
#         print(50*"--")
#         print(f"Confusion Matrix for Fold {fold_result[-1]['fold']}:")
#         print(cm)
#         # Print other evaluation indexs
#         scores_dict = compute_scores(labels, predictions_)
#         print ("Sensitivity = {}".format(scores_dict['sensitivity']))
#         print ("Specificity = {}".format(scores_dict['specificity']))
#         print ("ROC_AUC = {}".format(scores_dict['roc_auc']))
#         print("Accuracy = {}".format(scores_dict['accuracy']))
#         ##
#         fold_sen.append(scores_dict['sensitivity'])
#         fold_spe.append(scores_dict['specificity'])
#         fold_auc.append(scores_dict['roc_auc'])
#         fold_acc.append(scores_dict['accuracy'])
#         metrics = { 'sensitivity': fold_sen, 'specificity': fold_spe,'roc_auc': fold_auc,'accuracy': fold_acc}

#     # CI  
#     ci_results = {}
#     for metric_name, scores in metrics.items():
#         mean_score = np.mean(scores)
#         std_err = stats.sem(scores)
#         h = std_err * stats.t.ppf((1 + 0.95) / 2, 5 - 1)
#         ci_results[metric_name] = (mean_score, mean_score - h, mean_score + h)
#     print(50*"**")
#     print("CIs test_results for over all folds:")
#     print ("Sensitivity = {}".format(ci_results['sensitivity']))
#     print ("Specificity = {}".format(ci_results['specificity']))
#     print ("ROC_AUC = {}".format(ci_results['roc_auc']))
#     print("Accuracy = {}".format(ci_results['accuracy']))
#     print(50*"**")

    # ## Optional: Compute the mean confusion matrix over all folds
    # mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    # print("Mean Confusion Matrix over all folds:")
    # print(mean_confusion_matrix)
    # # Print other evaluation indexs
    # tn, fp, fn, tp = mean_confusion_matrix.ravel()
    # sen = tp / (tp + fn)
    # spe = tn / (fp + tn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # F1score = 2 * (precision * sen) / (precision + sen)
    # fpr, tpr, thresholds = roc_curve(labels, predictions)
    # roc_auc = auc(fpr, tpr)
    # acc = (tp+tn) / (tp + tn + fp + fn)
    # # pos_num = (labels==1).sum().item()
    # # neg_num = (labels==0).sum().item()
    # print ("Sensitivity = {}".format(sen))
    # print ("Specificity = {}".format(spe))
    # print ("Precision = {}".format(precision))
    # print ("F1score = {}".format(F1score))
    # print ("Recall = {}".format(recall))
    # print ("ROC_AUC = {}".format(roc_auc))
    # print("Accuracy = {}".format(acc))
    # print(50*"--")

## inference_results
def stats_infer(metrics):
    ci_results = {}
    for metric_name, scores in metrics.items():
        mean_score = np.mean(scores)
        std_err = stats.sem(scores)
        h = std_err * stats.t.ppf((1 + 0.95) / 2, 5 - 1)
        ci_results[metric_name] = (mean_score, mean_score - h, mean_score + h)
    print(50*"**")
    print("CIs inference_results for over all folds:")
    print ("Sensitivity = {}".format(ci_results['sensitivity']))
    print ("Specificity = {}".format(ci_results['specificity']))
    print ("ROC_AUC = {}".format(ci_results['roc_auc']))
    print("Accuracy = {}".format(ci_results['accuracy']))
    print(50*"**")


