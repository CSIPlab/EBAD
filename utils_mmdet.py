# this module is related to mmdetection models
import pdb
import sys
import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from torchvision import transforms
import mmcv
from mmcv.runner import load_checkpoint

sys.path.insert(0, 'mmdetection/')
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet_model_info import model_info


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

COCO_BBOX_LABEL_NAMES = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')

# voc index to coco index
voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]



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


def vis_bbox(img, bbox, label=None, score=None, ax=None, dataset='voc'):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(height, width, 3)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        dataset (str): voc or coco.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    random.seed(1)
    rgbas = [[random.randint(0, 255)/255 for _ in range(3)] + [1] for _ in range(80)] # to unify colors pick from 80 classes
    voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

    if dataset == 'voc':
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        rgbas = np.array(rgbas)[voc2coco].tolist()
    else:
        label_names = list(COCO_BBOX_LABEL_NAMES) + ['bg']

    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')
    
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    # Draw bbox and captions
    for i, bb in enumerate(bbox):
        
        caption = list()

        if label is not None and label_names is not None:
            lb = int(label[i])
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=rgbas[lb], linewidth=3))

        if len(caption) > 0:
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    color='white',
                    size="large",
                    style='italic',
                    bbox={'facecolor': rgbas[lb], 'edgecolor': rgbas[lb], 
                    'boxstyle':'round', 'alpha': 0.8, 'pad': 0.2})
    return ax



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
    elif model_name in ['FCOS', 'CenterNet']:
        conf_thres = 0.3
    elif model_name in ['Deformable DETR']: # was 0.3
        conf_thres = 0.3 # in context paper it was 0.1
    elif model_name in ['ATSS']:
        conf_thres = 0.2
    else:
        conf_thres = 0.2
        # tested for YOLOX, 
    return conf_thres


def output2det(outputs, im, conf_thres=0.5, dataset='voc'):
    """Convert the model outputs to targeted format
    Args: 
        outputs (lists): 80 lists, each has a numpy array of Nx5, (bbox and conf)
        conf_thres (float): confidence threshold
        im (np.ndarray): input image for get the size and clip at the boundary
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
    from mmdet.apis import inference_detector
    if not is_to_rgb(model):
        im = im[:,:,::-1]
    result = inference_detector(model, im)
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
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    if not is_to_rgb(model): im = im[:,:,::-1]
    cfg = model.cfg
    device = next(model.parameters()).device
    cfg = cfg.copy()
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    data = dict(img=im)
    data = test_pipeline(data)
    datas.append(data)
    data = collate(datas, samples_per_gpu=1)
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    data = scatter(data, [device])[0]
    return data

    
def get_train_data(model, im, pert, data, bboxes, labels):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """

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
    gt_bboxes = bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, image_sizes[1])
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, image_sizes[0])
    data_train['gt_bboxes'] = [torch.from_numpy(gt_bboxes).to(device)]
    data_train['gt_labels'] = [torch.from_numpy(labels).to(device)]
    # img = F.interpolate(img, size=image_sizes, mode='nearest')
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
    return data_train


def get_loss_from_dict(model_name, loss_dict):
    """Return the correct loss based on the model type
    Args:
        model_name (~ str): the mmdet model name, eg: 'Faster R-CNN', 'YOLOv3', 'RetinaNet', 'FreeAnchor' ...
        loss_dict (~ dict): the loss of the model, stored in a dictionary
    Returns:
        losses (~ torch.Tensor): the summation of the loss
    """
    if model_name in ['Faster R-CNN', 'Libra R-CNN', 'GN+WS']:
        losses = loss_dict['loss_cls'] + loss_dict['loss_bbox'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
        # losses = sum(loss_dict.values())
    elif model_name in ['Grid R-CNN']:
        losses = loss_dict['loss_cls'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
    elif model_name in ['YOLOv3', 'RetinaNet', 'RepPoints', 'SSD']:
        losses = sum(sum(loss_dict[key]) for key in loss_dict)
    else: # ['FreeAnchor', 'DETR', 'CenterNet', 'YOLOX', 'FoveaBox']
        losses = sum(loss_dict.values())
    return losses


class model_train(torch.nn.Module):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """

    def __init__(self, model_name, device='cuda:0', dataset='voc') -> None:
        super().__init__()
        mmdet_root = Path('mmdetection/') # root for config files
        config_file = str(mmdet_root / model_info[model_name]['config_file'])
        checkpoint_file = str(mmdet_root / model_info[model_name]['checkpoint_file'])
        config = mmcv.Config.fromfile(config_file)
        model_train = build_detector(config.model, test_cfg=config.get('test_cfg'))
        checkpoint = load_checkpoint(model_train, checkpoint_file, map_location=None)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model_train.CLASSES = checkpoint['meta']['CLASSES']
        model_train.cfg = config  # save the config in the model for convenience
        model_train.to(device)
        model_train.eval()
        self.model = model_train
        self.model_name = model_name
        self.device = device
        self.conf_thres = get_conf_thres(model_name)
        self.dataset = dataset

    def forward(self, x):
        """inference model using image x
        Args:
            x (numpy.ndarray): input image
            result (list): a list of output from mmdet model
        """
        result = inference_detector(self.model, x)
        return result

    def loss(self, x, pert, bboxes_tgt, labels_tgt):
        """get the loss

        args:
            x (numpy.ndarray):
            pert (tensor):             

        """
        data = get_test_data(self.model, x)
        data_train = get_train_data(self.model, x, pert.to(self.device), data, bboxes_tgt, labels_tgt)
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

    def det(self, x):
        """inference model using image x, get the processed output as detection
        
        args:
            x (numpy.ndarray): input image
        """
        result = inference_detector(self.model, x)
        det = output2det(result, x, conf_thres=self.conf_thres, dataset=self.dataset)
        return det


def get_train_model(config_file, checkpoint_file, device='cuda:0'):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """
    import mmcv
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
    config = mmcv.Config.fromfile(config_file)
    model_train = build_detector(config.model, test_cfg=config.get('test_cfg'))
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model_train, checkpoint_file, map_location=map_loc)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model_train.CLASSES = checkpoint['meta']['CLASSES']
    model_train.cfg = config  # save the config in the model for convenience
    model_train.to(device)
    # model_train.train()
    model_train.eval()
    return model_train


def get_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        iou (float): iou in [0, 1]
    """
    w1,h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2,h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    try:
        assert all([w1,h1,w2,h2])
    except:
        return 0

    # determine the coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = w1*h1
    area2 = w2*h2
    iou = area_inter / float(area1 + area2 - area_inter)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_success(detections, target_clean, iou_threshhold=0.9):
    """ see if the detection has target label at the corresponding location with IOU > 0.3
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
        target_clean (np.ndarray): a single object, our desired output. Shape (1,6) - [xyxy,cls,score]
    Returens:
        (bool): whether the detection is a success or not
    """
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > iou_threshhold and items[4] == target_clean[0][4]:
            return True
    return False


def is_success_hiding(detections):
    """ if nothing is detected it is a success
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
    Returens:
        (bool): whether the detection is a success or not
    """
    if len(detections) == 0:
        return True
    else:
        return False
