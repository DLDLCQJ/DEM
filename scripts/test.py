import torch
import torch.nn as nn

from utils.others import pad_dataset,custom_iter,pad_batch

def run_test_epoch(args, model, src_loader, tgt_loader,domain=None):

    predictions, truths =[], []
    for i, ((imgs_src, labels_src),(imgs_tgt,labels_tgt)) in enumerate(zip(src_loader,tgt_loader)):
        imgs_src,labels_src = imgs_src.to(args.device),labels_src.to(args.device)
        imgs_tgt,labels_tgt = imgs_tgt.to(args.device),labels_tgt.to(args.device)
        src_preds,tgt_preds, domains,src_feats_adv,tgt_feats_adv,features1,features2 = model(imgs_src,imgs_tgt,alpha=0)
        if domain=='source':
            outputs = src_preds.squeeze(1)
            labels = labels_src.to(torch.float32)
        else:
            outputs = tgt_preds.squeeze(1)
            labels = labels_tgt.to(torch.float32)
        # Print statistics
        predictions.extend(outputs.detach().cpu().numpy())
        truths.extend(labels.cpu().numpy())
    return predictions, truths
 

def run_alone_test_epoch(args, model,loader):

    #print('test_model:',models.state_dict()[list(models.state_dict().keys())[-1]])
    predictions, truths =[], []
    for i, (imgs, labels) in enumerate(loader):
        imgs,labels = imgs.to(args.device),labels.to(args.device)
        preds,_,_ = model(imgs)

        outputs = preds.squeeze(1)
        labels = labels.to(torch.float32)
        # Print statistics
        predictions.extend(outputs.detach().cpu().numpy())
        truths.extend(labels.cpu().numpy())
    return predictions, truths

def test_epoch(args, model, src_loader=None, tgt_loader=None, domain=None,fold=None,epoch=0):
    results_te =[]
    with torch.no_grad():
        if args.continual_type=='CRL':
            te_predicts,te_truths = run_test_epoch(args, model, src_loader, tgt_loader,domain)
        elif args.continual_type=='RL' and domain=='source':
            te_predicts,te_truths = run_alone_test_epoch(args, model, src_loader)
        elif args.continual_type=='RL' and domain=='target':
            te_predicts,te_truths = run_alone_test_epoch(args, model, tgt_loader)
    results_te.append({
        "fold": fold + 1,
        "epoch": epoch + 1,
        "predictions": te_predicts,
        "labels": te_truths
        })
    
    return results_te


