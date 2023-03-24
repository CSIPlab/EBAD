# this module is related to mmsegmentation models
import sys
from pathlib import Path

import mmcv
import torch
import numpy as np
from matplotlib import pyplot as plt 

mmseg_root = Path('mmsegmentation/')
sys.path.insert(0, str(mmseg_root))
from mmseg.core.evaluation import mean_iou, get_palette, get_classes

CITYSCAPES_CLASSES = (
        'road', 
        'sidewalk', 
        'building', 
        'wall', 
        'fence', 
        'pole',
        'traffic light', 
        'traffic sign', 
        'vegetation', 
        'terrain', 
        'sky',
        'person', 
        'rider', 
        'car', 
        'truck', 
        'bus', 
        'train', 
        'motorcycle',
        'bicycle'
    )

class LoadImage:
    """
        A pipeline to load image to the testpipeline.
        Adapt from mmsegementation/mmseg/apis/inference.py
    """
    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(height, width, 3)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img
    ax.imshow(img.astype(np.uint8))
    ax.axis('off')
    return ax

def vis_sseg(model,
            img,
            result,
            palette=None,
            classes=None,
            fig_size=None,
            opacity=0.5,
            show_class=False,
            ax=None,
            title='',
            block=True,
            out_file=None):
    """Visualize the senmantic segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    from matplotlib import patches as mpatches
    if hasattr(model, 'module'):
        model = model.module
    if show_class:  # adjust fig size if show class legends
        fig_size=(8,8)

    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    if ax is not None:
        ax.imshow(mmcv.bgr2rgb(img))
        ax.axis('off')
        ax.set_title(title)
        if show_class:
            patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                                    label=classes[i]) for i in np.unique(result[0]).astype(np.int32)]
            # put those patched as legend-handles into the legend
            ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., ncol=1, 
                    fontsize='large')
        #return ax
    
    else:
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        
        if show_class:
            patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                                    label=classes[i]) for i in np.unique(result[0]).astype(np.int32)]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2, 
                    fontsize='large')
        plt.title(title)
        plt.tight_layout()


def is_to_rgb(model):
    """check if a model takes rgb images or not
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
    """
    to_rgb = True
    for item in model.cfg.data.test.pipeline[1]['transforms']:
        if 'to_rgb' in item:
            to_rgb = item['to_rgb']
    return to_rgb


def get_conf_thres(model_name):
    """assign a different confidence threshold for every model
    Args: 
        model_name (str): the name of model
    Returns:
        conf_thres (~ float): the confidence threshold
        conf_thres is selected to reduce false positive rate
    """
    if model_name in ['Grid R-CNN']:
        conf_thres = 0.7
    elif model_name in ['Faster R-CNN', 'FreeAnchor', 'SSD']:
        conf_thres = 0.6    
    elif model_name in ['YOLOv3', 'RetinaNet', 'Libra R-CNN', 'GN+WS']:
        conf_thres = 0.5
    elif model_name in ['FoveaBox', 'RepPoints', 'DETR']:
        conf_thres = 0.4
    elif model_name in ['FCOS', 'Deformable DETR', 'CenterNet']:
        conf_thres = 0.3
    else:
        conf_thres = 0.2
    return conf_thres


def output2det(outputs, im, conf_thres = 0.5, dataset='voc'):
    """Convert the model outputs to targeted format
    Args: 
        conf_thres (float): confidence threshold
    Returns:
        det (numpy.ndarray): _bboxes(xyxy) - 4, _cls - 1, _prob - 1
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    """
    det = []
    for idx, items in enumerate(outputs):
        for item in items:
            det.append(item[:4].tolist() + [idx] + item[4:].tolist())
    det = np.array(det)
    
    # if det is empty
    if len(det) == 0: 
        return np.zeros([0,6])

    # thresholding the confidence score
    det = det[det[:,-1] >= conf_thres]
    
    if dataset == 'voc':
        # map the labels from coco to voc
        voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
        for idx, item in enumerate(det):
            if item[4] not in voc2coco:
                item[4] = -1
            else:
                det[idx,4] = voc2coco.index(item[4])
        det = det[det[:,4] != -1]

    # make the value in range
    m, n, _ = im.shape
    for item in det:
        item[0] = min(max(item[0],0),n)
        item[2] = min(max(item[2],0),n)
        item[1] = min(max(item[1],0),m)
        item[3] = min(max(item[3],0),m)
    return det


def get_det(model, model_name, im, dataset='voc'):
    """input an image to a model and get the detection
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
        im (~ numpy.ndarray): input image (in rgb format)
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    Returns:
        det (~ numpy.ndarray): nx6 array
    """
    from mmseg.apis import inference_segmentor
    if not is_to_rgb(model):
        im = im[:,:,::-1]
    result = inference_segmentor(model, im)
    conf_thres = get_conf_thres(model_name)
    det = output2det(result, im, conf_thres, dataset)
    return det


def show_det(models, im, dataset='voc', save_path=None):
    """show detection of a list of models
    Args: 
        models (~ mmdet.models.detectors or list): a single model or a list of models
        im (~ numpy.ndarray): input image (in rgb format)
    """
    from vis_tool import vis_bbox

    det_all = []
    if isinstance(models,list):    # a list of models
        n_mdoels = len(models)
        fig, ax = plt.subplots(1, n_mdoels, figsize=(6*n_mdoels, 5))
        for idx, model in enumerate(models):
            det = get_det(model, im, dataset) if is_to_rgb(model) else get_det(model, im[:,:,::-1], dataset)
            det_all.append(det)
            bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
            vis_bbox(im, bboxes, labels, scores, ax=ax[idx], dataset=dataset)
    else:   # a single model
        model = models
        det = get_det(model, im, dataset) if is_to_rgb(model) else get_det(model, im[:,:,::-1], dataset)
        det_all.append(det)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1, 1, 1)
        vis_bbox(im, bboxes, labels, scores, ax=ax, dataset=dataset)
    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
    return det_all


def get_test_data(model, im):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format)
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """

    from mmseg.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    if not is_to_rgb(model): im = im[:,:,::-1]
    cfg = model.cfg
    device = next(model.parameters()).device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=im)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    return data

    
def get_train_data(model, im, pert, data, tgt_seg):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        pert (torch.tensor): perturbation
        data (test_data): data fromat forged by test_pipeline
        tgt_seg (np.ndarray): desired semantic segmentation labels
    Returns:
        data_train (): train data format
    """
    import torch
    from torch.nn import functional as F
    from torchvision import transforms

    # get model device
    device = next(model.parameters()).device

    # BELOW IS TRAIN
    data_train = data.copy()
    data_train['img_metas'] = data_train['img_metas'][0]
    data_train['img'] = data_train['img'][0]
    ''' from file: datasets/pipelines/transforms.py '''
    
    if not is_to_rgb(model): im = im[:,:,::-1]
    img = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().to(device).contiguous()
    img = (img + pert).clamp(0,255)

    # 'type': 'Resize', 'keep_ratio': True, (1333, 800)
    ori_sizes = im.shape[:2]
    image_sizes = data_train['img_metas'][0]['img_shape'][:2]
    w_scale = image_sizes[1] / ori_sizes[1]
    h_scale = image_sizes[0] / ori_sizes[0]
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img = F.interpolate(img, size=image_sizes, mode='bilinear', align_corners=True)

    # 'type': 'Normalize', 'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], 'to_rgb': False
    img_norm_cfg = data_train['img_metas'][0]['img_norm_cfg']
    mean = img_norm_cfg['mean']
    std = img_norm_cfg['std']
    transform = transforms.Normalize(mean=mean, std=std)
    img = transform(img)

    # 'type': 'Pad', 'size_divisor': 32
    pad_sizes = data_train['img_metas'][0]['pad_shape'][:2]
    left = top = 0
    bottom = pad_sizes[0] - image_sizes[0]
    right = pad_sizes[1] - image_sizes[1]
    img = F.pad(img, (left, right, top, bottom), "constant", 0)
    data_train['img'] = img
    data_train['gt_semantic_seg'] = torch.from_numpy(tgt_seg).view([1,1]+list(tgt_seg.shape)).to(device)
    return data_train

import sys
from mmseg_model_info_cityscapes import model_info
sys.path.insert(0, 'mmsegmentation/')
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot

class model_train_seg(torch.nn.Module):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """

    def __init__(self, model_name, device='cuda:0', dataset='cityscapes') -> None:
        super().__init__()
        mmdet_root = Path('mmsegmentation/') # root for config files
        config_file = str(mmdet_root / model_info[model_name]['config_file'])
        checkpoint_file = str(mmdet_root / model_info[model_name]['checkpoint_file'])
        config = mmcv.Config.fromfile(config_file)

        model_train = get_train_model(config_file, checkpoint_file, device=device, size = (1024,512))

        self.model = model_train
        self.model_name = model_name
        self.device = device

        self.dataset = dataset

    def forward(self, x):
        """inference model using image x
        Args:
            x (numpy.ndarray): input image
            result (list): a list of output from mmdet model
        """
        result = inference_segmentor(self.model, x)
        return result

    def loss(self, x, pert, target):
        """get the loss

        args:
            x (numpy.ndarray):
            pert (tensor):             

        """
        data = get_test_data(self.model, x)
        data_train = get_train_data(self.model, x, pert.to(self.device), data, target)
        loss_dict = self.model(return_loss=True, **data_train)
        # print(f"loss_dict: {loss_dict}")
        loss = get_loss_from_dict(self.model_name, loss_dict)
        return loss

    def rgb(self):
        to_rgb = False # false by default
        for item in self.model.cfg.data.test.pipeline[1]['transforms']:
            if 'to_rgb' in item:
                to_rgb = item['to_rgb']
                return to_rgb

    def seg(self, x):
        """inference model using image x, get the processed output as detection
        
        args:
            x (numpy.ndarray): input image
        """
        seg = inference_segmentor(self.model, x)

        return seg

    def vis_seg(self, x, ax=None, title='', save_path=None):
        """plot the segmentation map
        """

        seg = self.seg(x)
        dataset_seg = self.dataset
        classes_seg = get_classes(dataset_seg)
        palette = get_palette(dataset_seg)
        vis_sseg(self.model, x, seg, palette, classes_seg, opacity=1, show_class=True, ax=ax, title=title, out_file=save_path);


def get_train_model(config_file, checkpoint_file, device='cuda:0', size = None):
    """return a model in train mode
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """
    import mmcv
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint
    # adjust config
    config = mmcv.Config.fromfile(config_file)
    config.norm_cfg = dict(type='BN', requires_grad=True)
    config.model.backbone.norm_cfg = config.norm_cfg
    config.model.decode_head.norm_cfg = config.norm_cfg
    if 'auxiliary_head' in config.model.keys():
        config.model.auxiliary_head.norm_cfg = config.norm_cfg
    config.model.pretrained = None
    config.model.train_cfg = None

    if size:
        config.train_pipeline[2].img_scale = size
        config.test_pipeline[1].img_scale = size
        config.data.train.pipeline[2].img_scale = size
        config.data.val.pipeline[1].img_scale = size
        config.data.test.pipeline[1].img_scale = size


    model_train = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model_train, checkpoint_file, map_location=map_loc)
    model_train.CLASSES = checkpoint['meta']['CLASSES']
    model_train.PALETTE = checkpoint['meta']['PALETTE']
    model_train.cfg = config  # save the config in the model for convenience
    model_train.to(device)
    model_train.eval()
    return model_train


def get_loss_from_dict(model_name, loss_dict):
    """Return the correct loss based on the model type
    Args:
        model_name (~ str): the mmdet model name, eg: 'Faster R-CNN', 'YOLOv3', 'RetinaNet', 'FreeAnchor' ...
        loss_dict (~ dict): the loss of the model, stored in a dictionary
    Returns:
        losses (~ torch.Tensor): the summation of the loss
    """

    if 'aux.loss_ce' in loss_dict.keys():
        losses = loss_dict['decode.loss_ce'] + loss_dict['aux.loss_ce']
    else:
        losses = loss_dict['decode.loss_ce']
    return losses

def get_target_seg(raw_tgt_seg, gt_seg):
    import cv2
    import random
    random.seed(0)
    class_labels = np.unique(gt_seg)
    sum = [np.sum(gt_seg==lbl) for lbl in class_labels]
    attack_class = np.array(sum).argmax()
    attack_class = class_labels[attack_class]
    print(f'Attack Class: {attack_class}')
    for label in class_labels:
        label_map = (gt_seg==label)
        label_map_mask = np.zeros_like(gt_seg).astype(np.uint8)
        label_map_mask[label_map] = 1

        if label == attack_class:
            n_objs, obj_map = cv2.connectedComponents(label_map_mask,8)
            for obj in range(1,n_objs):
                # get object
                obj_area = (obj_map == obj)
                obj_area_mask = np.zeros_like(gt_seg).astype(np.uint8)
                obj_area_mask[obj_area] = 1
                # get dominant label and replace
                lbl_pool = raw_tgt_seg * obj_area_mask
                counts = np.bincount(lbl_pool.reshape(-1))
                void_lbls = gt_seg.reshape(-1).shape - obj_area_mask.sum()
                counts[0] =- void_lbls
                new_lbl = np.argmax(counts) # exclude label zero (road) in cityscapes
                raw_tgt_seg[obj_area] = new_lbl
        else:
            raw_tgt_seg[label_map] = gt_seg[label_map]
    return raw_tgt_seg,attack_class