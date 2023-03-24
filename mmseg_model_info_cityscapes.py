# model library pre-trained on Cityscape dataset
# dataset and usage: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html

model_info = {
    'PSPNet': {
        'config_file': 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth',
    },
    'PSPNet-r101':{
        'config_file':'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file':'checkpoints/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth'
    },
    'PSPNet-r18':{
        'config_file':'configs/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file':'checkpoints/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth'
    },
    'PSPNet-mv2':{ # mobileNet-v2
        'config_file':'configs/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file':'checkpoints/pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes/pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth'
    },
    'PSPNet-s101':{ # ResNeSt: S-101-D8
        'config_file':'configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file':'checkpoints/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x1024_80k_cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth'
    },
    'FCN': {
        'config_file': 'configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x1024_40k_cityscapes/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth',
    },
    'FCN-hr18':{ # HRNetV2p-W18
        'config_file':'configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py',
        'checkpoint_file':'checkpoints/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth'
    },
    'FCN-hr48':{ # HRNetV2p-W48
        'config_file':'configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py',
        'checkpoint_file':'checkpoints/fcn_hr48_512x1024_40k_cityscapes_20200601_014240-a989b146.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_40k_cityscapes/fcn_hr48_512x1024_40k_cityscapes_20200601_014240-a989b146.pth'
    },
    'FCN-mv2':{ # mobileNet-v2
        'config_file':'configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file':'checkpoints/fcn_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-d24c28c1.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-d24c28c1.pth'
    },
    'FCN-s101':{ # ResNeSt: S-101-D8
        'config_file':'configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file':'checkpoints/fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x1024_80k_cityscapes/fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth'
    },
    'DeepLabV3': {
        'config_file': 'configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth',
    },
    'DeepLabV3-r101': {
        'config_file': 'configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth',
    },
    'DeepLabV3-s101': {
        'config_file': 'configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file': 'checkpoints/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth',
    },
    'DeepLabV3-mv2': {
        'config_file': 'configs/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes.py',
        'checkpoint_file': 'checkpoints/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth',
    },
    'DeepLabV3+': {
        'config_file': 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth',
    },
    'PSANet': {
        'config_file': 'configs/psanet/psanet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/psanet_r50-d8_512x1024_40k_cityscapes_20200606_103117-99fac37c.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/psanet/psanet_r50-d8_512x1024_40k_cityscapes/psanet_r50-d8_512x1024_40k_cityscapes_20200606_103117-99fac37c.pth',
    },
    'UPerNet': {
        'config_file': 'configs/upernet/upernet_r50_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth',
    },
    'UPerNet-r101':{
        'config_file':'configs/upernet/upernet_r101_512x1024_40k_cityscapes.py',
        'checkpoint_file':'checkpoints/upernet_r101_512x1024_40k_cityscapes_20200605_094933-ebce3b10.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r101_512x1024_40k_cityscapes/upernet_r101_512x1024_40k_cityscapes_20200605_094933-ebce3b10.pth'
    },
    'UPerNet-r18':{
        'config_file':'configs/upernet/upernet_r18_512x1024_40k_cityscapes.py',
        'checkpoint_file':'checkpoints/upernet_r18_512x1024_40k_cityscapes_20220615_113231-12ee861d.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r18_512x1024_40k_cityscapes/upernet_r18_512x1024_40k_cityscapes_20220615_113231-12ee861d.pth'
    },
    'NonLocalNet': {
        'config_file': 'configs/nonlocal_net/nonlocal_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/nonlocal_r50-d8_512x1024_40k_cityscapes_20200605_210748-c75e81e3.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/nonlocal_net/nonlocal_r50-d8_512x1024_40k_cityscapes/nonlocal_r50-d8_512x1024_40k_cityscapes_20200605_210748-c75e81e3.pth',
    },
    'EncNet': {
        'config_file': 'configs/encnet/encnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/encnet_r50-d8_512x1024_40k_cityscapes_20200621_220958-68638a47.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_40k_cityscapes/encnet_r50-d8_512x1024_40k_cityscapes_20200621_220958-68638a47.pth',
    },
    'CCNet': {
        'config_file': 'configs/ccnet/ccnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/ccnet_r50-d8_512x1024_40k_cityscapes_20200616_142517-4123f401.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ccnet/ccnet_r50-d8_512x1024_40k_cityscapes/ccnet_r50-d8_512x1024_40k_cityscapes_20200616_142517-4123f401.pth',
    },
    'DANet': {
        'config_file': 'configs/danet/danet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/danet_r50-d8_512x1024_40k_cityscapes_20200605_191324-c0dbfa5f.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/danet/danet_r50-d8_512x1024_40k_cityscapes/danet_r50-d8_512x1024_40k_cityscapes_20200605_191324-c0dbfa5f.pth',
    },
    'APCNet': {
        'config_file': 'configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/apcnet_r50-d8_512x1024_40k_cityscapes_20201214_115717-5e88fa33.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes/apcnet_r50-d8_512x1024_40k_cityscapes_20201214_115717-5e88fa33.pth',
    },
    'HRNet': {
        'config_file': 'configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth',
    },
    'GCNet': {
        'config_file': 'configs/gcnet/gcnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/gcnet_r50-d8_512x1024_40k_cityscapes_20200618_074436-4b0fd17b.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/gcnet/gcnet_r50-d8_512x1024_40k_cityscapes/gcnet_r50-d8_512x1024_40k_cityscapes_20200618_074436-4b0fd17b.pth',
    },
    'DMNet': {
        'config_file': 'configs/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/dmnet_r50-d8_512x1024_40k_cityscapes_20201215_042326-615373cf.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes/dmnet_r50-d8_512x1024_40k_cityscapes_20201215_042326-615373cf.pth',
    },
    'ANN': {
        'config_file': 'configs/ann/ann_r50-d8_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/ann_r50-d8_512x1024_40k_cityscapes_20200605_095211-049fc292.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_40k_cityscapes/ann_r50-d8_512x1024_40k_cityscapes_20200605_095211-049fc292.pth',
    },
    'OCRNet': {
        'config_file': 'configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes.py',
        'checkpoint_file': 'checkpoints/ocrnet_hr18_512x1024_40k_cityscapes_20200601_033320-401c5bdd.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes/ocrnet_hr18_512x1024_40k_cityscapes_20200601_033320-401c5bdd.pth',
    },
}


def main():
    import urllib.request
    from pathlib import Path

    checkpoints_root = Path('mmsegmentation/checkpoints')
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_files = list(checkpoints_root.glob('*.pth'))
    existing_files = [file.name for file in existing_files]

    for idx,model_name in enumerate(model_info):
        url = model_info[model_name]['download_link']
        file_name = url.split('/')[-1]
        if file_name in existing_files:
            print(f"{model_name} already exists, {idx+1}/{len(model_info)}")
            continue
        print(f'downloading {model_name} {idx+1}/{len(model_info)}')
        file_data = urllib.request.urlopen(url).read()
        with open(checkpoints_root / file_name, 'wb') as f:
            f.write(file_data)


if __name__ == "__main__":
    main()