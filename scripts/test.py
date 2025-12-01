import torch
import torch.nn as nn
import torch.distributed as dist

from utils.others import pad_dataset,custom_iter,pad_batch


def run_test_epoch(args, model, src_loader, tgt_loader, domain=None):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for i, ((imgs_src, labels_src), (imgs_tgt, labels_tgt)) in enumerate(zip(src_loader, tgt_loader)):
            imgs_src, labels_src = imgs_src.to(args.device), labels_src.to(args.device)
            imgs_tgt, labels_tgt = imgs_tgt.to(args.device), labels_tgt.to(args.device)
            
            src_preds, tgt_preds, domains, src_feats_adv, tgt_feats_adv, features1, features2 = model(imgs_src, imgs_tgt, alpha=0)
            
            if domain == 'source':
                outputs = src_preds.squeeze(1)
                labels = labels_src.to(torch.float32)
            else:
                outputs = tgt_preds.squeeze(1)
                labels = labels_tgt.to(torch.float32)

            predictions.extend(outputs.detach().cpu().numpy())
            truths.extend(labels.cpu().numpy())
            
    return predictions, truths


def run_alone_test_epoch(args, model, loader):
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            preds, _, _ = model(imgs)

            outputs = preds.squeeze(1)
            labels = labels.to(torch.float32)

            predictions.extend(outputs.detach().cpu().numpy())
            truths.extend(labels.cpu().numpy())
            
    return predictions, truths


def test_epoch(args, model, src_loader=None, tgt_loader=None, domain=None, fold=None, epoch=0):
    results_te = []
    if args.continual_type == 'CRL':
        te_predicts, te_truths = run_test_epoch(args, model, src_loader, tgt_loader, domain)
    elif args.continual_type == 'RL' and domain == 'source':
        te_predicts, te_truths = run_alone_test_epoch(args, model, src_loader)
    elif args.continual_type == 'RL' and domain == 'target':
        te_predicts, te_truths = run_alone_test_epoch(args, model, tgt_loader)
    else:
        te_predicts, te_truths = [], []
        
    if args.distributed:
        preds_tensor = torch.tensor(te_predicts, device=args.device)
        truths_tensor = torch.tensor(te_truths, device=args.device)
        preds_list = [torch.zeros_like(preds_tensor) for _ in range(args.world_size)]
        truths_list = [torch.zeros_like(truths_tensor) for _ in range(args.world_size)]
        
        dist.all_gather(preds_list, preds_tensor)
        dist.all_gather(truths_list, truths_tensor)
        final_predicts = torch.cat(preds_list).cpu().tolist()
        final_truths = torch.cat(truths_list).cpu().tolist()
    else:
        final_predicts = te_predicts
        final_truths = te_truths
        
    if args.rank == 0:
        results_te.append({
            "fold": fold + 1 if fold is not None else 1,
            "epoch": epoch + 1,
            "predictions": final_predicts,
            "labels": final_truths
        })
        return results_te
    else:
        return []

