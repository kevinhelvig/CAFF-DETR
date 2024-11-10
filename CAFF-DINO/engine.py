# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
DINO two streams
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MaP 
import numpy as np 
import matplotlib.pyplot as plt

import albumentations as A

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    # backbones freezing
    for name, param in model.named_parameters():
        if name.split(".")[0] == 'backbone_visible' or name.split(".")[0] == 'backbone' or name.split(".")[0] == 'backbone_ir': # to uncomment for full training 
            pass 
            # param.requires_grad = False
        """
        # Freeze the fusion module after learning on LLVIP : extreme learning hack  
        if name.split(".")[0] == 'conv_corr' : # or name.split(".")[0] == 'redim_corr': # to uncomment for full training 
            param.requires_grad = False 
        """
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, samples_thermal, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        samples_thermal = [t.to(device) for t in samples_thermal]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, samples_thermal, targets)
            else:
                outputs = model(samples, samples_thermal)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, samples_thermal, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        samples_thermal = [t.to(device) for t in samples_thermal]
        
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, samples_thermal, targets)
            else:
                outputs = model(samples, samples_thermal)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #coco_evaluator = None
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
    
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

@torch.no_grad()
def evaluate_map(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = None #CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    metric = MaP(box_format='xyxy', class_metrics=True, iou_type="bbox", average='macro', backend='pycocotools')
    count = 0
    import cv2
    for samples, samples_thermal, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        
        samples_thermal = [t.to(device) for t in samples_thermal]
        
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        
        targets_2 = {"boxes": targets[0]["boxes"][:], "labels": targets[0]["labels"][:]}

        targets_2["boxes"] = torch.tensor(targets_2["boxes"])
        targets_2["labels"] = torch.tensor(targets_2["labels"])

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, samples_thermal, targets)
            else:
                outputs = model(samples, samples_thermal)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #coco_evaluator = None
        
        if count<=7000: #save all pairs => experiment results
            image = samples.tensors[0]        
            image_IR = samples_thermal[0]
            boxes = targets[0]['boxes'].cpu().numpy()
            #print("GT")
            a, _ = np.shape(boxes)
            
            #Dataset boxes loading
            # cxcywh conversion 
            for b in range(a): 
                x_c, y_c, w, h = boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3]
                boxes[b] = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                     (x_c + 0.5 * w), (y_c + 0.5 * h)]

            h,w = image[0].shape # for de normalizing images
            h_init, w_init = orig_target_sizes[0].detach().cpu()
            boxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(boxes,h,w)]
            image = denormalize(image, mean, std)
            image_IR = denormalize(image_IR, mean, std)
            sample = image.permute(1,2,0).cpu().numpy()
            sample_IR = image_IR.permute(1,2,0).cpu().numpy()

            sample = np.array(sample)        
            
            #fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            color = (0, 0, 220) # red = ground truth => issue with localization ? 
            
            sample = cv2.UMat(sample)
            sample_IR = cv2.UMat(sample_IR)
            normalized_sample =  cv2.UMat.get(sample) * 255# cv2.normalize(sample, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            normalized_sample_ir = cv2.UMat.get(sample_IR) * 255 # cv2.normalize(sample_IR, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            normalized_sample = cv2.cvtColor(normalized_sample, cv2.COLOR_RGB2BGR)
            normalized_sample_ir = cv2.cvtColor(normalized_sample_ir, cv2.COLOR_RGB2BGR)
            
            for box in boxes:
                cv2.rectangle(normalized_sample,
                          ((box[0]), (box[1])),
                          (box[2], box[3]),
                          color, 2)
                
                cv2.rectangle(normalized_sample_ir,((box[0]), (box[1])),(box[2], box[3]),color, 2)
                
             
            # pred boxes are xyxy format, not the case for targets ... (cxcywh)
            oboxes = outputs['pred_boxes'][0].detach().cpu().numpy()
            #oboxes = results[0]['boxes'].detach().cpu().numpy()

            a, _ = np.shape(oboxes)
            h_img,w_img = image[0].shape # for de normalizing images


            for b in range(a):
                x_c, y_c, w, h = oboxes[b][0], oboxes[b][1], oboxes[b][2], oboxes[b][3]
                oboxes[b] = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                     (x_c + 0.5 * w), (y_c + 0.5 * h)]
            
            h,w = image[0].shape # for de normalizing images
            
            #oboxes = [np.array(box).astype(np.int32) for box in oboxes ] # A.core.bbox_utils.denormalize_bboxes(oboxes,h,w)]
            oboxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(oboxes,h,w)]
            #prob = outputs['pred_logits'][0].softmax(1).detach().cpu().numpy()[:,0]
            prob = results[0]['scores'].cpu().numpy()

            for box,p in zip(oboxes, prob):
                if p > 0.5 :
                    color = (220,0,0) #if p>0.5 else (0,0,0)
                    
                    cv2.rectangle(normalized_sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          color, 2)
                    cv2.rectangle(normalized_sample_ir,(box[0], box[1]),(box[2], box[3]),color, 2)
                    

            #ax.set_axis_off()
            img_array = normalized_sample # cv2.UMat.get(normalized_sample)
            img_ir_array = normalized_sample_ir # cv2.UMat.get(normalized_sample_ir)
            """
            plt.imshow(np.asarray(img_array)); plt.xlabel("x [px]"); plt.ylabel("y [px]")
            plt.savefig('/d/khelvig/DINO-LLVIP/DINO_twostreams/ex/plot_ir_'+str(count)+'.png')
            plt.figure()
            plt.imshow(np.asarray(img_ir_array)); plt.xlabel("x [px]"); plt.ylabel("y [px]")
            plt.savefig('/d/khelvig/DINO-LLVIP/DINO_twostreams/ex/plot_vis_'+str(count)+'.png')
            plt.figure()
            """
            cv2.imwrite('/d/khelvig/DINO-LLVIP/DINO_caff64_shuffle/ex_vedai/plot_ir_'+str(count)+'.png', np.asarray(img_ir_array)) 
            cv2.imwrite('/d/khelvig/DINO-LLVIP/DINO_caff64_shuffle/ex_vedai/plot_vis_'+str(count)+'.png', np.asarray(img_array))
            
        count += 1  
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        res_list, target_list = [], []
        
        for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
            gt_bbox = tgt['boxes']
            gt_label = tgt['labels']
            gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
            # img_h, img_w = tgt['orig_size'].unbind()
            # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            # _res_bbox = res['boxes'] / scale_fct
            _res_bbox = outbbox
            _res_prob = res['scores']
            _res_label = res['labels']
            #res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
            # import ipdb;ipdb.set_trace()
            
        #new evaluation way ? 
        image = samples.tensors[0] 
        h,w = image[0].shape # for de normalizing images
        #oboxes = results[0]['boxes'].detach().cpu().numpy()
        oboxes = outputs['pred_boxes'][0].detach().cpu().numpy()
        #oboxes = results[0]['boxes'].detach().cpu().numpy()
        
        a, _ = np.shape(oboxes)
        for b in range(a):
            x_c, y_c, w, h = oboxes[b][0], oboxes[b][1], oboxes[b][2], oboxes[b][3]
            oboxes[b] = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
        #oboxes = [np.array(box).astype(np.int32) for box in oboxes] # A.core.bbox_utils.denormalize_bboxes(oboxes,h,w)]
        oboxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(oboxes,h,w)]
        #prob = outputs['pred_logits'][0].softmax(1).detach().cpu().numpy()[:,0]
        prob = results[0]['scores'].cpu().numpy()
        pred_labels = results[0]['labels'].cpu().numpy()
        keep_boxes = []; keep_prob = []; keep_labels = []
        c = 0 
        """
        for box,p in zip(oboxes, prob):
            if  1 : # Threshold: tou adjust in order to compare performance, base p>0.001
                keep_labels.append(pred_labels[c])
                keep_prob.append(p)
                if len(keep_boxes) == 0: 
                    keep_boxes = np.array([box])
                else: 
                    keep_boxes = np.vstack((keep_boxes, box))
            c += 1        
        """
        # -------- targets rewriting and correction 
        boxes = targets[0]['boxes'].cpu().numpy()
        #print("GT")
        a, _ = np.shape(boxes)
        
        
        # cxcywh conversion 
        for b in range(a): 
            x_c, y_c, w, h = boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3]
            boxes[b] = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
        h,w = image[0].shape # for de normalizing images
        boxes = [np.array(box).astype(np.int32) for box in A.core.bbox_utils.denormalize_bboxes(boxes,h,w)]
        targets_2 = {"boxes": torch.tensor(boxes, device='cuda:0'), "labels": targets[0]["labels"][:]}

        res_thresh = {"boxes": torch.tensor(oboxes[:300], device='cuda:0'),
                        "scores": torch.tensor(prob, device='cuda:0'), 
                        "labels":torch.tensor(pred_labels, device='cuda:0')}

        metric.update([res_thresh], [targets_2])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
                
    print(metric.compute())
    
    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    coco_evaluator = None 
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator
