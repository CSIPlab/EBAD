"""
    Attack object detectors and segmentation model simultaneously in a blackbox setting
"""
# https://github.com/open-mmlab/mmcv#installation
import argparse
import random
import sys
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


mmdet_root = Path('mmdetection/')
sys.path.insert(0, str(mmdet_root))
from utils_mmdet import (COCO_BBOX_LABEL_NAMES, VOC_BBOX_LABEL_NAMES, get_det,
                         get_iou, is_success, model_train, vis_bbox, voc2coco)

mmseg_root = Path('mmsegmentation/')
sys.path.insert(0, str(mmseg_root))
from utils_mmseg import model_train_seg, vis_sseg
from mmseg.core.evaluation import get_classes, get_palette, mean_iou


def recursive_find_file(folder: Path, suffixes: typing.List[str]) -> typing.List[Path]:
    """find all files with desired suffixes in a folder"""
    files = []
    for child in folder.iterdir():
        if not child.is_file():
            child_files = recursive_find_file(child, suffixes)
            files.extend(child_files)
        if child.suffix in suffixes:
            files.append(child)
    return files


def PM_tensor_weight_balancing(im, adv, target, target_seg, w, ensemble, ensemble_seg, eps, n_iters, alpha, dataset='voc', weight_balancing=False, type='None'):
    """perturbation machine, balance the weights of different surrogate models
    args:
        im (tensor): original image, shape [1,3,h,w].cuda()
        adv (tensor): adversarial image
        target (numpy.ndarray): label for object detection, (xyxy, cls, conf)
        w (numpy.ndarray): ensemble weights
        ensemble (): surrogate ensemble
        eps (int): linf norm bound (0-255)
        n_iters (int): number of iterations
        alpha (flaot): step size

    returns:
        adv_list (list of Tensors): list of adversarial images for all iterations
        LOSS (dict of lists): 'ens' is the ensemble loss, and other individual surrogate losses
    """
    # prepare target label input: voc -> coco, since models are trained on coco
    bboxes_tgt = target[:,:4].astype(np.float32)
    labels_tgt = target[:,4].astype(int).copy()
    if dataset == 'voc':
        for i in range(len(labels_tgt)): 
            labels_tgt[i] = voc2coco[labels_tgt[i]]

    
    n_det = len(ensemble)
    im_np = im.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_list = []
    pert = adv - im
    LOSS = defaultdict(list) # loss lists for different models
    for i in range(n_iters):
        pert.requires_grad = True
        loss_list = []
        loss_list_np = []
        for model_idx, model in enumerate(ensemble + ensemble_seg):
            if model_idx < n_det:
                loss = model.loss(im_np, pert, bboxes_tgt, labels_tgt)
                # print(f"det_loss ({model.model_name}) {model_idx}: {loss}")
            else:
                loss = model.loss(im_np, pert, target_seg)
                # print(f"seg_loss ({model.model_name}) {model_idx}: {loss}")
            loss_list.append(loss)
            loss_list_np.append(loss.item())
            LOSS[model.model_name].append(loss.item())
        
        # if balance the weights at every iteration
        if weight_balancing:
            w_inv = 1/np.array(loss_list_np)
            w = w_inv / w_inv.sum()

        # print(f"w: {w}")
        if type == 'None':
            loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble + ensemble_seg)))
        elif type == 'det_only':
            loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble)))
        else:
            loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble),len(ensemble)+len(ensemble_seg)))
        loss_ens.backward()
        with torch.no_grad():
            pert = pert - alpha*torch.sign(pert.grad)
            pert = pert.clamp(min=-eps, max=eps)
            LOSS['ens'].append(loss_ens.item())
            adv = (im + pert).clip(0, 255)
            adv_list.append(adv)
    return adv_list, LOSS


def PM_tensor_weight_balancing_np(im_np, target, target_seg, w_np, ensemble, ensemble_seg, eps, n_iters, alpha, dataset='voc', weight_balancing=False, adv_init=None, type='None'):
    """perturbation machine, numpy input version
    
    """
    device = next(ensemble[0].parameters()).device
    im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).permute(2,0,1).unsqueeze(0).float().to(device)

    # w = torch.from_numpy(w_np).float().to(device)
    adv_list, LOSS = PM_tensor_weight_balancing(im, adv, target, target_seg, w_np, ensemble, ensemble_seg, eps, n_iters, alpha, dataset, weight_balancing, type)
    adv_np = adv_list[-1].squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return adv_np, LOSS


def get_bb_loss(detections, target_clean, LOSS):
    """define the blackbox attack loss
        if the original object is detected, the loss is the conf score of the victim object
        otherwise, the original object disappears, the conf is below the threshold, the loss is the wb ensemble loss
    args:
        detections ():
        target_clean ():
        LOSS ():
    return:
        bb_loss (): the blackbox loss
    """
    max_iou = 0
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > max(max_iou, 0.3) and items[4] == target_clean[0][4]:
            max_iou = iou
            bb_loss = 1e3 + items[5] # add a large const to make sure it is larger than conf ens loss
    # if it disappears
    if max_iou < 0.3:
        bb_loss = LOSS['ens'][-1]
    return bb_loss


def get_bb_loss_seg(segmentation, seg_clean, victim_class_seg, target_class_seg, target_clean):
    """get the ration of pixels from the region
    """
    x1,y1,x2,y2 = target_clean[0,:4].astype(int)
    # pdb.set_trace()
    gt_area = np.sum(seg_clean[y1:y2,x1:x2] == victim_class_seg)
    adv_area = np.sum(segmentation[y1:y2,x1:x2] == target_class_seg)
    psr = adv_area / gt_area
    bb_loss_seg = 1 - psr
    return bb_loss_seg


def save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, all_models_seg, victim_class_seg, target_class_seg, im_id, im_idx, attack_goal, log_root, dataset, n_query):    
    """get the loss bb, success_list on all surrogate models, and save detections to fig
    
    args:

    returns:
        loss_bb (float): loss on the victim model
        success_list (list of 0/1s): successful for all models
    """
    fig_h = 5
    fig_w = 5
    n_all = len(all_models) + len(all_models_seg)
    fig, ax = plt.subplots(2,1+n_all,figsize=((1+n_all)*fig_w,2*fig_h))
    # 1st row, clean image, detection on surrogate models, detection on victim model
    # 2nd row, perturbed image, detection on surrogate models, detection on victim model
    row = 0
    ax[row,0].imshow(im_np)
    ax[row,0].set_title('clean image')
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(im_np)
        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(im_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

    # save seg
    for seg_idx, model_seg in enumerate(all_models_seg):
        title = f'{model_seg.model_name}'
        model_seg.vis_seg(im_np, ax=ax[row,seg_idx+model_idx+2], title=title)

    row = 1
    ax[row,0].imshow(adv_np)
    ax[row,0].set_title(f'adv image @ iter {n_query} \n {attack_goal}')
    success_list = [] # 1 for success, 0 for fail for all models
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(adv_np)
        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(adv_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

        # check for success and get bb loss
        if model_idx == len(all_models)-1:
            loss_bb = get_bb_loss(det_adv, target_clean, LOSS)

        # victim model is at the last index
        success_list.append(is_success(det_adv, target_clean))

    # save seg
    for seg_idx, model_seg in enumerate(all_models_seg):
        seg_save_path = log_root / f"{im_idx}_{im_id}_iter{n_query}_seg_{model_seg.model_name}.png"
        title = f'{model_seg.model_name}'
        # out_file = seg_save_path
        model_seg.vis_seg(adv_np, ax=ax[row,seg_idx+model_idx+2], title=title, save_path=seg_save_path)
        if seg_idx == len(all_models_seg)-1:
            seg_clean = model_seg.seg(im_np)[0]
            segmentation = model_seg.seg(adv_np)[0]

            loss_bb_seg = get_bb_loss_seg(segmentation, seg_clean, victim_class_seg, target_class_seg, target_clean)

    
    plt.tight_layout()
    if success_list[-1]:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}_success.png")
    else:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}.png")
    plt.close()

    return loss_bb, success_list, loss_bb_seg
    

def main():
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", type=int, default=10, help="perturbation level: 10,20,30,40,50")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--root", type=str, default='result', help="the folder name of result")
    parser.add_argument("--victim", type=str, default='RetinaNet', help="victim model")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--n_wb", type=int, default=2, help="number of models in the ensemble")
    parser.add_argument("--surrogate", type=str, default='Faster R-CNN', help="surrogate model when n_wb=1")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=20, help="iterations of updating w")
    parser.add_argument("--dataset", type=str, default='coco', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    parser.add_argument("-single", action='store_true', help="only care about one obj")
    parser.add_argument("-no_balancing", action='store_true', help="do not balance weights at beginning")
    parser.add_argument("--type", type=str, default='None', help="det_only, seg_only")
    args = parser.parse_args()
    
    print(f"args.single: {args.single}")
    eps = args.eps
    n_iters = args.iters
    x_alpha = args.x
    alpha = eps / n_iters * x_alpha
    iterw = args.iterw
    n_wb = args.n_wb
    lr_w = args.lr
    dataset = args.dataset
    victim_name = args.victim

    # load surrogate models
    # detection
    ensemble = []
    models_all = ['Faster R-CNN', 'YOLOv3', 'FCOS', 'Grid R-CNN', 'SSD']
    model_list = models_all[:n_wb]
    if n_wb == 1:
        model_list = [args.surrogate]
    for model_name in model_list:
        ensemble.append(model_train(model_name=model_name, dataset=dataset))
    if victim_name == 'Libra':
        victim_name = 'Libra R-CNN'
    elif victim_name == 'Deformable':
        victim_name = 'Deformable DETR'
    model_victim = model_train(model_name=victim_name, dataset=dataset)
    
    # segmentation
    wb_names_seg = ['FCN','UPerNet','DeepLabV3+', 'PSANet', 'EncNet', 'CCNet','APCNet','GCNet','DMNet','ANN'] # 'DeepLabV3', 'PSPNet',
    model_list_seg = wb_names_seg[:n_wb]
    ensemble_seg = [model_train_seg(model_name=model_name) for model_name in model_list_seg]
    victim_name_seg = 'PSPNet'
    model_victim_seg = model_train_seg(model_name=victim_name_seg)
    dataset_seg = 'cityscapes'
    classes_seg = get_classes(dataset_seg)
    palette = get_palette(dataset_seg)
    num_classes_seg = len(classes_seg)


    n_wb_all = 2*n_wb
    all_model_names = model_list + [victim_name]
    all_models = ensemble + [model_victim]
    all_models_seg = ensemble_seg + [model_victim_seg]
    all_model_names_seg = model_list_seg + [victim_name_seg]
    all_model_names_wb = model_list + model_list_seg

    # create folders
    exp_name = f'BB_{n_wb}wb_linf_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_lr{lr_w}_iterw{iterw}'
    if dataset != 'voc':
        exp_name += f'_{dataset}'
    if n_wb == 1:
        exp_name += f'_{args.surrogate}'
    if args.single:
        exp_name += '_single'
    if args.no_balancing:
        exp_name += '_noBalancing'
    if args.type != 'None':
        exp_name += args.type

    print(f"\nExperiment: {exp_name} \n")
    result_root = Path(f"results_joint_det_seg/")
    exp_root = result_root / exp_name
    log_root = exp_root / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log_loss_root = exp_root / 'logs_loss'
    log_loss_root.mkdir(parents=True, exist_ok=True)
    adv_root = exp_root / 'advs'
    adv_root.mkdir(parents=True, exist_ok=True)
    target_root = exp_root / 'targets'
    target_root.mkdir(parents=True, exist_ok=True)

    
    data_root = Path("/data/SalmanAsif/")
    im_root = data_root / "Cityscapes/leftImg8bit/val/"
    # n_labels = 80
    label_names = COCO_BBOX_LABEL_NAMES

    dict_k_sucess_id_v_query = {} # query counts of successful im_ids
    dict_k_valid_id_v_success_list = {} # lists of success for all mdoels for valid im_ids
    dict_k_valid_id_v_seg_loss = {} # lists of seg loss for all seg mdoels for valid im_ids
    n_obj_list = []

    img_paths = recursive_find_file(folder=im_root, suffixes=[".jpg", ".png"])
    for im_idx, im_path in tqdm(enumerate(img_paths)):
        im_id = im_path.stem
        im_pil = Image.open(im_path).convert('RGB')
        im_pil = im_pil.resize((1024, 512))
        im_np = np.array(im_pil)


        # save seg
        for seg_idx, model_seg in enumerate(all_models_seg):
            seg_save_path = log_root / f"{im_idx}_{im_id}_GT_seg_{all_model_names_seg[seg_idx]}.png"
            seg_temp = model_seg.seg(im_np)
            vis_sseg(model_seg.model, im_np, seg_temp, palette, classes_seg, opacity=1, show_class=True, out_file=seg_save_path);
            plt.close()


        # get detection on clean images and determine the target class
        det = model_victim.det(im_np)
        seg = model_victim_seg.seg(im_np)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        print(f"n_objects: {len(det)}")
        n_obj_list.append(len(det))
        if len(det) == 0: # if nothing is detected, skip this image
            continue
        else:
            dict_k_valid_id_v_success_list[im_id] = []
            dict_k_valid_id_v_seg_loss[im_id] = []

        all_categories = set(labels.astype(int))  # all apperaing objects in the scene

        # person (0) to car (2) potted plant (58)
        # victim_class = 0
        # target_class = 58
        # a list of target objects:     0-person, 1-bicycle, 2-car, 3-motorcycle, 5-bus, 6-train, 7-truck, 9-traffic light, 58-potted plant, 
        # a list of target objects seg: 11-person, 18-bicycle, 13-car, 17-motorcycle, 15-bus, 16-train, 14-truck, 6-traffic light, 8-vegetation
        target_pool = [0,1,2,3,5,6,7,9,58]
        target_pool_seg = [11,18,13,17,15,16,14,6,8]

        # select a victim class from object detection
        victim_idx = random.randint(0,len(det)-1)
        victim_class = int(det[victim_idx,4])
        while victim_class not in target_pool:
            victim_idx = random.randint(0,len(det)-1)
            victim_class = int(det[victim_idx,4])
        victim_bbox = det[victim_idx,:4]
        
        
        target_class = np.random.permutation(list(set(target_pool) - set([victim_class])))[0]
        target = det.copy()
        # victim_idx = (target[:, 4] == victim_class)[0]
        # victim_idx = 0
        # only change one label
        target[victim_idx, 4] = target_class
        # only keep one label
        target_clean = target[victim_idx,:][None]
        if args.single: # only care about the target object
            target = target_clean
        # save target to np
        np.save(target_root/f"{im_id}_target", target)
        
        # seg
        gt_seg = seg[0]
        seg_clean = gt_seg
        target_seg = gt_seg.copy()
        # 13 car, 11 person
        # target_seg[target_seg == 11] = 13
        # target_seg[target_seg == 11] = 8 # veg
        # pdb.set_trace()

        x1,y1,x2,y2 = victim_bbox.astype(int)
        victim_class_seg = target_pool_seg[target_pool.index(victim_class)]
        target_class_seg = target_pool_seg[target_pool.index(target_class)]
        # pdb.set_trace()
        target_seg[y1:y2,x1:x2][target_seg[y1:y2,x1:x2] == victim_class_seg] = target_class_seg


        # basic information of attack
        attack_goal = f"{label_names[victim_class]} to {label_names[target_class]}"
        info = f"im_idx: {im_idx}, im_id: {im_id}, victim_class: {label_names[victim_class]}, target_class: {label_names[target_class]}\n"
        print(info)
        file = open(exp_root / f'{exp_name}.txt', 'a')
        file.write(f"{info}\n\n")
        file.close()

        if args.no_balancing:
            print(f"no_balancing, using equal weights")
            w_inv = np.ones(n_wb) 
            w_np = np.ones(n_wb) / n_wb

        else:
            # determine the initial w, via weight balancing
            dummy_w = np.ones(n_wb_all)
            _, LOSS = PM_tensor_weight_balancing_np(im_np, target, target_seg, dummy_w, ensemble, ensemble_seg, eps, n_iters=1, alpha=alpha, dataset=dataset)
            loss_list_np = [LOSS[name][0] for name in all_model_names_wb]
            w_inv = 1 / np.array(loss_list_np)
            w_np = w_inv / w_inv.sum()
            print(f"loss_list: {loss_list_np}")
            print(f"w_np: {w_np}")

        

        adv_np, LOSS = PM_tensor_weight_balancing_np(im_np, target, target_seg, w_np, ensemble, ensemble_seg, eps, n_iters, alpha=alpha, dataset=dataset, type=args.type)
        n_query = 0
        loss_bb, success_list, loss_bb_seg = save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, all_models_seg, victim_class_seg, target_class_seg, im_id, im_idx, attack_goal, log_root, dataset, n_query)
        print(f"loss_bb_seg: {loss_bb_seg}")
        dict_k_valid_id_v_success_list[im_id].append(success_list)
        dict_k_valid_id_v_seg_loss[im_id].append(loss_bb_seg)

        # save adv in folder
        adv_path = adv_root / f"{im_id}_iter{n_query:02d}.png"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)

        # stop whenever successful
        if success_list[-1]:
            dict_k_sucess_id_v_query[im_id] = n_query
            print(f"success! image im idx: {im_idx}")
            
            w_list = []
            loss_bb_list = [loss_bb]
            loss_ens_list = LOSS['ens'] # ensemble losses during training

        n_query += 1
        w_list = []        
        loss_bb_list = [loss_bb]
        loss_ens_list = LOSS['ens'] # ensemble losses during training

        idx_w = 0 # idx of wb in W, rotate
        while n_query < iterw:

            ##################################### query plus #####################################
            w_np_temp_plus = w_np.copy()
            w_np_temp_plus[idx_w] += lr_w * w_inv[idx_w]
            adv_np_plus, LOSS_plus = PM_tensor_weight_balancing_np(im_np, target, target_seg, w_np_temp_plus, ensemble, ensemble_seg, eps, n_iters, alpha=alpha, dataset=dataset, adv_init=adv_np, type=args.type)
            # loss_bb_plus, success_list = save_det_to_fig(im_np, adv_np_plus, LOSS_plus, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)
            loss_bb_plus, success_list, loss_bb_seg_plus = save_det_to_fig(im_np, adv_np_plus, LOSS_plus, target_clean, all_models, all_models_seg, victim_class_seg, target_class_seg, im_id, im_idx, attack_goal, log_root, dataset, n_query)
            dict_k_valid_id_v_success_list[im_id].append(success_list)
            dict_k_valid_id_v_seg_loss[im_id].append(loss_bb_seg_plus)

            n_query += 1
            print(f"iter: {n_query}, {idx_w} +, loss_bb: {loss_bb_plus}, loss_bb_seg_plus: {loss_bb_seg_plus}")

            # save adv in folder
            adv_path = adv_root / f"{im_id}_iter{n_query:02d}.png"
            adv_png = Image.fromarray(adv_np_plus.astype(np.uint8))
            adv_png.save(adv_path)

            # stop whenever successful
            if success_list[-1]:
                dict_k_sucess_id_v_query[im_id] = n_query
                print(f"success! image im idx: {im_idx}")
                loss_bb = loss_bb_plus
                loss_ens = LOSS_plus["ens"]
                w_np = w_np_temp_plus
                adv_np = adv_np_plus
                # break

            #######################################################################################
            

            ##################################### query minus #####################################
            w_np_temp_minus = w_np.copy()
            w_np_temp_minus[idx_w] -= lr_w * w_inv[idx_w]
            adv_np_minus, LOSS_minus = PM_tensor_weight_balancing_np(im_np, target, target_seg, w_np_temp_minus, ensemble, ensemble_seg, eps, n_iters, alpha=alpha, dataset=dataset, adv_init=adv_np, type=args.type)
            # loss_bb_minus, success_list = save_det_to_fig(im_np, adv_np_minus, LOSS_minus, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)
            loss_bb_minus, success_list, loss_bb_seg_minus = save_det_to_fig(im_np, adv_np_minus, LOSS_minus, target_clean, all_models, all_models_seg, victim_class_seg, target_class_seg, im_id, im_idx, attack_goal, log_root, dataset, n_query)
            dict_k_valid_id_v_success_list[im_id].append(success_list)
            dict_k_valid_id_v_seg_loss[im_id].append(loss_bb_seg_minus)

            n_query += 1
            print(f"iter: {n_query}, {idx_w} -, loss_bb: {loss_bb_minus}, loss_bb_seg_minus: {loss_bb_seg_minus}")

            # save adv in folder
            adv_path = adv_root / f"{im_id}_iter{n_query:02d}.png"
            adv_png = Image.fromarray(adv_np_minus.astype(np.uint8))
            adv_png.save(adv_path)

            # stop whenever successful
            if success_list[-1]:
                dict_k_sucess_id_v_query[im_id] = n_query
                print(f"success! image im idx: {im_idx}")
                loss_bb = loss_bb_minus
                loss_ens = LOSS_minus["ens"]
                w_np = w_np_temp_minus
                adv_np = adv_np_minus
                # break

            #######################################################################################


            ##################################### update w, adv #####################################
            if loss_bb_plus < loss_bb_minus:
                loss_bb = loss_bb_plus
                loss_ens = LOSS_plus["ens"]
                w_np = w_np_temp_plus
                adv_np = adv_np_plus
            else:
                loss_bb = loss_bb_minus
                loss_ens = LOSS_minus["ens"]
                w_np = w_np_temp_minus
                adv_np = adv_np_minus

            # relu and normalize
            w_np = np.maximum(0, w_np)
            w_np = w_np + 0.005 # minimum set to 0.005
            w_np = w_np / w_np.sum()
            print(f"w: {w_np}, idx_w: {idx_w}")
            #######################################################################################
                
            idx_w = (idx_w+1)%n_wb_all
            w_list.append(w_np.tolist())
            loss_bb_list.append(loss_bb)
            loss_ens_list += loss_ens


        if im_id in dict_k_sucess_id_v_query:
            # save to txt
            info = f"im_idx: {im_idx}, id: {im_id}, query: {n_query}, loss_bb: {loss_bb:.4f}, w: {w_np}\n"
            file = open(exp_root / f'{exp_name}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"im_idx: {im_idx}; total_success: {len(dict_k_sucess_id_v_query)}")

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_ens_list)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        im = im_np
        im_temp = im if model_victim.rgb else im[:,:,::-1]
        det = get_det(model_victim.model, victim_name, im_temp, dataset)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        vis_bbox(im, bboxes, labels, scores, ax=ax[1], dataset=dataset)
        ax[1].set_title(f"clean image")

        adv = adv_np
        im_temp = adv if model_victim.rgb else adv[:,:,::-1]
        det = get_det(model_victim.model, victim_name, im_temp, dataset)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        vis_bbox(adv, bboxes, labels, scores, ax=ax[2], dataset=dataset)
        ax[2].set_title(f'adv image @ iter {n_query} \n {label_names[victim_class]} to {label_names[target_class]}')
        ax[3].plot(loss_bb_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        ax[4].plot(w_list)
        ax[4].legend(model_list, shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        ax[4].set_yscale('log')
        plt.tight_layout()
        if im_id in dict_k_sucess_id_v_query:
            plt.savefig(log_loss_root / f"{im_id}_success_iter{n_query}.png")
        else:
            plt.savefig(log_loss_root / f"{im_id}.png")
        plt.close()

        if len(dict_k_sucess_id_v_query) > 0:
            query_list = [dict_k_sucess_id_v_query[key] for key in dict_k_sucess_id_v_query]
            print(f"query_list: {query_list}")
            print(f"avg queries: {np.mean(query_list)}")
            print(f"success rate (victim): {len(dict_k_sucess_id_v_query) / len(dict_k_valid_id_v_success_list)}")

        # print surrogate success rates
        success_list_stack = []
        for valid_id in dict_k_valid_id_v_success_list:
            success_list = np.array(dict_k_valid_id_v_success_list[valid_id])
            success_list = success_list.sum(axis=0).astype(bool).astype(int).tolist()
            success_list_stack.append(success_list)

        success_list_stack = np.array(success_list_stack).sum(axis=0)
        # pdb.set_trace()
        for idx, success_cnt in enumerate(success_list_stack):
            print(f"success rate of {all_model_names[idx]}: {success_cnt / len(dict_k_valid_id_v_success_list)}")
    
        # save np files / save at each iteration in case got cut off in the middle
        np.save(exp_root/f"dict_k_sucess_id_v_query", dict_k_sucess_id_v_query)
        np.save(exp_root/f"dict_k_valid_id_v_success_list", dict_k_valid_id_v_success_list)
        np.save(exp_root/f"dict_k_valid_id_v_seg_loss", dict_k_valid_id_v_seg_loss)


if __name__ == '__main__':
    main()
