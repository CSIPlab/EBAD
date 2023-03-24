"""
    Attack segmentation models in a blackbox setting
    design blackbox loss
"""
# https://github.com/open-mmlab/mmcv#installation
import sys, os, glob
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


mmseg_root = Path('mmsegmentation/')
sys.path.insert(0, str(mmseg_root))
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import mean_iou, get_palette, get_classes
from utils_mmseg import is_to_rgb, get_target_seg
from utils_mmseg import get_train_model, get_train_data, get_test_data, get_loss_from_dict, vis_sseg


def PM_np(im, adv, target, w, ensemble, ensemble_train, ensemble_names, eps, n_iters, alpha, untargeted):
    """perturbation machine (numpy.ndarray)
    Args:
        im (numpy.ndarray): original image
        adv (numpy.ndarray): adversarial image
        target (numpy.ndarray): label for object detection, (xyxy, cls, conf)
        w (numpy.ndarray): ensemble weights
        ensemble (): surrogate ensemble
        eps (int): linf norm bound (0-255)
        n_iters (int): 
        alpha (float):
        untargeted (boolean):
    """
    device = next(ensemble[0].parameters()).device
    n_wb = len(ensemble)

    pert = torch.from_numpy((adv-im).transpose((2, 0, 1)))[None].float().cuda()

    loss_list = []
    for i in range(n_iters):
        pert.requires_grad = True
        loss_joint = []
        for model, model_train, model_name in zip(ensemble, ensemble_train, ensemble_names):
            
            device = next(model.parameters()).device
            data = get_test_data(model, im)
            data_train = get_train_data(model, im, pert.to(device), data, target)

            loss_dict = model_train(return_loss=True, **data_train)
            losses = get_loss_from_dict(model_name, loss_dict)
            loss_joint.append(losses)
        loss_joint = sum(w[i]*loss_joint[i] for i in range(n_wb))
        loss_joint.backward()
        with torch.no_grad():
            if untargeted:
                pert = pert + alpha*torch.sign(pert.grad)
            else:
                pert = pert - alpha*torch.sign(pert.grad)         
            pert = pert.clamp(min=-eps, max=eps)

            loss_list.append(loss_joint.item())
    pert = pert.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv = (im + pert).clip(0, 255)

    return adv, loss_list



def get_loss(im, model, model_name, gt_seg, target_seg, attack_class, num_classes, untargeted):
    """get the mIoU score of the target_label as loss for blackbox attack
    Args:
        target_seg (np.array): targeted segmentation
    Return:
        loss (float): value to be minimized. return mIoU if untargeted; else, return 1-PSR
    """
    loss = 0
    # query victim model
    im_temp = im if is_to_rgb(model) else im[:,:,::-1]
    seg_temp = inference_segmentor(model, im_temp)
    
    if untargeted:
        # calculate mIoU between query result and targeted segmentation
        miou_cls = mean_iou([target_seg], [target_seg], num_classes, nan_to_num=-1, ignore_index=25)
        miou_cls = miou_cls['IoU']!=-1
        miou = mean_iou(seg_temp, [target_seg], num_classes, nan_to_num=-1, ignore_index=25)
        miou = miou['IoU'][miou_cls].mean()
        loss = miou
        print(f'Blackbox: {model_name}   Attack mIoU: {miou*100:.2f}    loss: {loss*100:.2f}')
    else:
        # calculate PSR between query result and targeted segmentation
        obj_area = (gt_seg == attack_class)
        total_pixs = obj_area.sum()
        correct_pixs = (target_seg[obj_area] == seg_temp[0][obj_area]).sum()
        psr = correct_pixs/total_pixs
        loss = 1 - psr
        print(f'Blackbox: {model_name}   Attack PSR: {psr*100:.2f}    loss: {loss*100:.2f}')
    
    return loss


def main():
    parser = argparse.ArgumentParser(description="Weight Balancing attacks on Segmentation")
    parser.add_argument("--eps", nargs="?", default=8, help="perturbation budget: 10,20,30,40,50")
    parser.add_argument("--iters", nargs="?", default=5, help="iterations of PGD attack: 5,6,10,20...")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--victim", nargs="?", default='PSPNet', choices=['PSPNet','DeepLabV3'], help="victim model, more options under mmseg_model_info.py")
    parser.add_argument("--x", nargs="?", default=2, help="times alpha by x")
    parser.add_argument("--n_wb", nargs="?", default=2, help="ensemble size")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument("--target", nargs="?", default='ll', help="using 2nd most-likely(ml) or least-likely(ll) as target label")
    parser.add_argument("-save_queries", action='store_true', help="save results for every queries")
    parser.add_argument("-visualize", action='store_true', help="save visualization results")
    parser.add_argument("--lr", nargs="?", default=1e-4, help="Weight Balancing learning rate")
    parser.add_argument("--iterw", nargs="?", default=20, help="number of queries for Weight Balancing")
    parser.add_argument("--n_imgs", nargs="?", default=2, help="number of experiment images")
    parser.add_argument("--data", nargs="?", default='cityscapes', choices=['cityscapes', 'voc'], help="experiment datasets")
    parser.add_argument("--backbone", nargs="?", default='r50', choices=['r50', 'r101'], help="backbone of models")

    args = parser.parse_args()

    device = f'cuda:{int(args.gpu)}'
    attack_type = 'untargeted' if args.untargeted else f'targeted_{args.target}'
    n_imgs = int(args.n_imgs)
    dataset = args.data
    
    eps = int(args.eps)
    n_iters = int(args.iters)
    x_alpha = int(args.x)
    alpha = eps / n_iters * x_alpha
    iterw = int(args.iterw)
    n_wb = int(args.n_wb)
    lr_w = 1/n_wb/20

    if dataset == 'cityscapes':
        from mmseg_model_info_cityscapes import model_info
        wb_names = ['FCN','UPerNet', 'PSANet', 'GCNet','ANN', 'EncNet', 'CCNet', 'APCNet', 'DMNet', 'DeepLabV3+'] # 'DeepLabV3', 'PSPNet',
        size = (1024,512)
    elif dataset == 'voc':
        from mmseg_model_info_voc import model_info
        if args.backbone == 'r50':
            wb_names = ['FCN','UPerNet', 'PSANet', 'GCNet','ANN', 'EncNet'] # 'DeepLabV3', 'PSPNet','DeepLabV3+',
        elif args.backbone == 'r101':
            wb_names = ['FCN-r101','UPerNet-r101','DeepLabV3+-r101', 'PSANet-r101', 'GCNet-r101','ANN']
        else:
            print(f'{dataset} models only support Res50,101 backbones!')
        size = (512,512)
    else:
        print(f'Dataset {dataset} not supported!')

    classes = get_classes(dataset)
    palette = get_palette(dataset)
    num_classes = len(classes)

    wb_names = wb_names[:n_wb]
    victim_name = args.victim
    model_list = wb_names + [victim_name]

    # load models
    models_train = []    
    for model_name in model_list:
        config_file = model_info[model_name]['config_file']
        checkpoint_file = model_info[model_name]['checkpoint_file']
        config_file = str(mmseg_root/config_file)
        checkpoint_file = str(mmseg_root/checkpoint_file)
        models_train.append(get_train_model(config_file, checkpoint_file, device=device, size=size))
    models = models_train

    ensemble = models[:n_wb]
    ensemble_train = models_train[:n_wb]
    ensemble_names = wb_names
    model_victim = models[-1]

    data_root = Path("/data/SalmanAsif/")
    # read images
    image_paths = []
    if dataset == 'cityscapes':
        im_dirs = ['frankfurt','lindau','munster']
        for im_dir in im_dirs:
            image_root = data_root / f'Cityscapes/leftImg8bit/val/{im_dir}'
            image_paths += glob.glob(os.path.join(image_root, '*.png'))
    else:
        image_root = f'mmsegmentation/data/VOCdevkit/VOC2012/'
        val_idxs = image_root + 'ImageSets/Segmentation/val.txt'
        with open(val_idxs,'r') as file:
            for line in file.readlines():
                image_paths.append(image_root + f'JPEGImages/{line[:-1]}.jpg')
        file.close()

    print(f'{len(image_paths)} images founded ...')
    image_paths = image_paths[:n_imgs]
    print(f'{n_imgs} images loaded ...')

    # create experiment folders
    exp = f'{dataset}_{args.backbone}_{attack_type}_{n_wb}wb_eps_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_lr{lr_w}_iterw{iterw}'
    result_root = Path(f"results_segmentation/")
    exp_root = result_root / exp / 'logs'
    exp_root.mkdir(parents=True, exist_ok=True)
    adv_root = result_root / exp / 'advs'
    adv_root.mkdir(parents=True, exist_ok=True)
    target_root = result_root / exp / 'targets'
    target_root.mkdir(parents=True, exist_ok=True)

    # summary of experiment
    print(f'\nPerforming {attack_type} attack on {n_imgs} {dataset} images ...')
    print(f'PGD iters: {n_iters}\tW.O. iters: {iterw}\tModel backbone: {args.backbone}\tImage size: {size}')
    print(f'Ensemble: {wb_names}')
    print(f'Victim: {victim_name}\n')

    # attack
    glob_miou_list = []
    for im_idx, im_path in tqdm(enumerate(image_paths)):
        miou_list = []
        im_name = im_path.split('/')[-1].split('.')[0]
        exp_name = f"idx{im_idx}_{im_name}"

        im = np.array(Image.open(im_path).convert('RGB'))
        adv = im.copy()
        
        # get ground-truth segmentation and craft attack plan
        # inference_segmentor() returns three segmentation maps in total
        # {ground-truth prediction} + {2nd most-likely segmentation} + {least-likely segmentation}
        seg = inference_segmentor(models[0], im)
        gt_seg = seg[0]
        if args.untargeted:
            tgt_seg = gt_seg # most-likely label
            target = tgt_seg
            attack_cls = -1
        else:
            if args.target =='ml':
                tgt_seg = seg[1] # 2nd most-likely label
            else:
                tgt_seg = seg[2] # least-likely label
            target, attack_cls = get_target_seg(tgt_seg, gt_seg)

        n_query = 0
        w_np = np.array([1 for _ in range(n_wb)]) / n_wb

        adv, loss_wb_list = PM_np(im, adv, target, w_np, ensemble, ensemble_train, ensemble_names, eps, n_iters, alpha, args.untargeted)
        loss = get_loss(adv, model_victim, victim_name, gt_seg, target, attack_cls, num_classes, args.untargeted)
        miou = loss if args.untargeted else 1 - loss
        miou_list.append(miou)
        if args.save_queries:
            # output zero-query segmentation
            im_temp = adv if is_to_rgb(model_victim) else adv[:,:,::-1]
            seg_temp = inference_segmentor(model_victim, im_temp)
            vis_sseg(model_victim, im, seg_temp, palette, classes, opacity=1, show_class=True, title=f'{n_query} query - {victim_name}'); 
            plt.savefig(exp_root / f"{im_name}_{n_query}_query_seg.png") # save attacked segmentation
            plt.close()

        n_query += 1
        w_list = []
        loss_bb_list = [] # loss of victim model

        idx_w = 0
        lr_w = 1/n_wb/20
        last_idx = 0
        # begin query
        while n_query < iterw:
            if n_wb == 1:
                break
            # get +ve
            w_np_temp_plus = w_np.copy()
            w_np_temp_plus[idx_w] += lr_w
            adv_plus, losses_plus = PM_np(im, adv, target, w_np_temp_plus, ensemble, ensemble_train, ensemble_names, eps, n_iters, alpha, args.untargeted)
            loss_plus = get_loss(adv_plus, model_victim, victim_name, gt_seg, target, attack_cls, num_classes, args.untargeted)
            n_query += 1

            # get -ve
            w_np_temp_minus = w_np.copy()
            w_np_temp_minus[idx_w] -= lr_w
            adv_minus, losses_minus = PM_np(im, adv, target, w_np_temp_minus, ensemble, ensemble_train, ensemble_names, eps, n_iters, alpha, args.untargeted)
            loss_minus = get_loss(adv_minus, model_victim, victim_name, gt_seg, target, attack_cls, num_classes, args.untargeted)
            n_query += 1

            # update
            if loss_plus < loss and loss_plus < loss_minus:
            # if loss_plus < loss_minus:
                loss = loss_plus
                w_np = w_np_temp_plus
                adv = adv_plus
                loss_wb_list += losses_plus
                print(f"{idx_w} +")
                last_idx = idx_w
                if args.save_queries:
                    # output intermidiate-query segmentations
                    im_temp = adv if is_to_rgb(model_victim) else adv[:,:,::-1]
                    seg_temp = inference_segmentor(model_victim, im_temp)
                    vis_sseg(model_victim, im, seg_temp, palette, classes, opacity=1, show_class=True, title=f'{n_query} query - {victim_name}'); 
                    plt.savefig(exp_root / f"{im_name}_{n_query}_query_seg.png") # save attacked segmentation
                    plt.close()
            elif loss_minus < loss and loss_minus < loss_plus:
            # else:
                loss = loss_minus
                w_np = w_np_temp_minus
                adv = adv_minus
                loss_wb_list += losses_minus
                print(f"{idx_w} -")
                last_idx = idx_w
                if args.save_queries:
                    # output intermidiate-query segmentations
                    im_temp = adv if is_to_rgb(model_victim) else adv[:,:,::-1]
                    seg_temp = inference_segmentor(model_victim, im_temp)
                    vis_sseg(model_victim, im, seg_temp, palette, classes, opacity=1, show_class=True, title=f'{n_query} query - {victim_name}'); 
                    plt.savefig(exp_root / f"{im_name}_{n_query}_query_seg.png") # save attacked segmentation
                    plt.close()
            
                
            idx_w = (idx_w+1)%n_wb
            if n_query > 5 and last_idx == idx_w:
                lr_w /= 2 # half the lr if there is no change
                print(f"lr_w: {lr_w}")

            w_list.append(w_np.tolist())
            loss_bb_list.append(loss)

            miou = loss if args.untargeted else 1 - loss
            miou_list.append(miou)
        
        glob_miou_list.append(miou_list)

        # logging
        metric = 'mIoU' if args.untargeted else 'PSR'
        info = f"im_idx: {im_idx}, zero-query {metric}: {miou_list[0]*100:.2f}, zero-query avg {metric}: {np.array(glob_miou_list)[:,0].mean()*100:.2f}, {iterw}-queries {metric}: {miou_list[-1]*100:.2f}, {iterw}-queries avg {metric}: {np.array(glob_miou_list)[:,-1].mean()*100:.2f}, w: {w_np.squeeze().tolist()}, {metric}_vs_queries:{miou_list}\n"

        file = open(exp_root / f'{exp}.txt', 'a')
        file.write(f"{info}")
        file.close()
        
        print(info)

        # plot average metric value v.s. queries curve
        plt.figure()
        plt.plot(np.array(glob_miou_list).mean(axis=0))
        plt.yscale('log')
        plt.savefig(exp_root / f"{metric}_vs_queries.png")
        plt.close()

        # save adv image
        adv_path = adv_root / f"{im_idx}.png"
        adv_png = Image.fromarray(adv.astype(np.uint8))
        adv_png.save(adv_path)

        if not args.untargeted:
            # save target seg
            tgt_path = target_root / f"{im_name}_target_ann.png"
            tgt_png = Image.fromarray(tgt_seg.astype(np.uint8))
            tgt_png.save(tgt_path)

        if args.visualize:
            if not args.untargeted:
                # visualize target segmentation map
                vis_sseg(models[0], im, [tgt_seg], palette, classes, opacity=1, show_class=True, title=f'target seg - {model_name}'); 
                plt.savefig(exp_root / f"{im_name}_target_seg.png") # save attacked segmentation
                plt.close()

            # plot attack summary figs
            fig, ax = plt.subplots(1,5,figsize=(30,5))
            ax[0].plot(loss_wb_list)
            ax[0].set_xlabel('iters')
            ax[0].set_title('loss on surrogate ensemble')

            # ax[1]
            im_temp = im if is_to_rgb(model_victim) else im[:,:,::-1]
            seg_temp = inference_segmentor(model_victim, im_temp)
            vis_sseg(model_victim, im, seg_temp, palette, classes, opacity=1, ax=ax[1], title=f'clean image - {victim_name}'); 
            # ax[2]
            im_temp = adv if is_to_rgb(model_victim) else adv[:,:,::-1]
            seg_temp = inference_segmentor(model_victim, im_temp)
            vis_sseg(model_victim, im, seg_temp, palette, classes, opacity=1, ax=ax[2], title=f'adv image - {victim_name}'); 

            ax[3].plot(loss_bb_list)
            ax[3].set_title('loss on victim model')
            ax[3].set_xlabel('iters')

            ax[4].plot(w_list)
            ax[4].legend(wb_names, shadow=True, bbox_to_anchor=(1.05, 1))
            ax[4].set_title('w of surrogate models')
            ax[4].set_xlabel('iters')

            plt.tight_layout()
            plt.savefig(exp_root / f"{exp_name}.png")
            plt.close()

            # plot surrogate segmentations
            fig, ax = plt.subplots(2,n_wb,figsize=(8*n_wb,7))
            for idx, model in enumerate(ensemble):
                im_temp = im if is_to_rgb(model) else im[:,:,::-1]
                seg_temp = inference_segmentor(model, im_temp)
                if n_wb == 1:
                    vis_sseg(model, im, seg_temp, palette, classes, opacity=1, ax=ax[0], title=f'clean - {wb_names[idx]}');

                    im_temp = adv if is_to_rgb(model) else adv[:,:,::-1]
                    seg_temp = inference_segmentor(model, im_temp)
                    vis_sseg(model, im, seg_temp, palette, classes, opacity=1, ax=ax[1], title=f'adv - {wb_names[idx]}');
                else:
                    vis_sseg(model, im, seg_temp, palette, classes, opacity=1, ax=ax[0,idx], show_class=True, title=f'clean - {wb_names[idx]}');

                    im_temp = adv if is_to_rgb(model) else adv[:,:,::-1]
                    seg_temp = inference_segmentor(model, im_temp)
                    vis_sseg(model, im, seg_temp, palette, classes, opacity=1, ax=ax[1,idx], title=f'adv - {wb_names[idx]}');
            
            plt.tight_layout()
            plt.savefig(exp_root / f"{exp_name}_wb.png")
            plt.close()

    # summary attack metric value evolution
    info = f"global {metric}: {glob_miou_list}\n"
    file = open(exp_root / f'{exp}.txt', 'a')
    file.write(f"{info}")
    file.close()



if __name__ == '__main__':
    main()