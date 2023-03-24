# model library pre-trained on Pascal VOC dataset
# dataset and usage: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html

model_info = {
    'PSPNet': {
        'config_file': 'configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth',
    },
    'PSPNet-r101':{
        'config_file':'configs/pspnet/pspnet_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file':'checkpoints/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x512_20k_voc12aug/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth'
    },
    'FCN': {
        'config_file': 'configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth',
    },
    'FCN-r101':{
        'config_file':'configs/fcn/fcn_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file':'checkpoints/fcn_r101-d8_512x512_20k_voc12aug_20200617_010842-0bb4e798.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r101-d8_512x512_20k_voc12aug/fcn_r101-d8_512x512_20k_voc12aug_20200617_010842-0bb4e798.pth'
    },
    'DeepLabV3': {
        'config_file': 'configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth',
    },
    'DeepLabV3-r101': {
        'config_file': 'configs/deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth',
    },
    'DeepLabV3+': {
        'config_file': 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth',
    },
    'DeepLabV3+-r101': {
        'config_file': 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/deeplabv3plus_r101-d8_512x512_20k_voc12aug_20200617_102345-c7ff3d56.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_voc12aug/deeplabv3plus_r101-d8_512x512_20k_voc12aug_20200617_102345-c7ff3d56.pth',
    },
    'PSANet': {
        'config_file': 'configs/psanet/psanet_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/psanet_r50-d8_512x512_20k_voc12aug_20200617_102413-2f1bbaa1.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/psanet/psanet_r50-d8_512x512_20k_voc12aug/psanet_r50-d8_512x512_20k_voc12aug_20200617_102413-2f1bbaa1.pth',
    },
    'PSANet-r101': {
        'config_file': 'configs/psanet/psanet_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/psanet_r101-d8_512x512_20k_voc12aug_20200617_110624-946fef11.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/psanet/psanet_r101-d8_512x512_20k_voc12aug/psanet_r101-d8_512x512_20k_voc12aug_20200617_110624-946fef11.pth',
    },
    'UPerNet': {
        'config_file': 'configs/upernet/upernet_r50_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/upernet_r50_512x512_20k_voc12aug_20200617_165330-5b5890a7.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x512_20k_voc12aug/upernet_r50_512x512_20k_voc12aug_20200617_165330-5b5890a7.pth',
    },
    'UPerNet-r101':{
        'config_file':'configs/upernet/upernet_r101_512x512_20k_voc12aug.py',
        'checkpoint_file':'checkpoints/upernet_r101_512x512_20k_voc12aug_20200617_165629-f14e7f27.pth',
        'download_link':'https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r101_512x512_20k_voc12aug/upernet_r101_512x512_20k_voc12aug_20200617_165629-f14e7f27.pth'
    },
    'NonLocalNet': {
        'config_file': 'configs/nonlocal_net/nonlocal_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/nonlocal_r50-d8_512x512_20k_voc12aug_20200617_222613-07f2a57c.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/nonlocal_net/nonlocal_r50-d8_512x512_20k_voc12aug/nonlocal_r50-d8_512x512_20k_voc12aug_20200617_222613-07f2a57c.pth',
    },
    'NonLocalNet-r101': {
        'config_file': 'configs/nonlocal_net/nonlocal_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/nonlocal_r101-d8_512x512_20k_voc12aug_20200617_222615-948c68ab.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/nonlocal_net/nonlocal_r101-d8_512x512_20k_voc12aug/nonlocal_r101-d8_512x512_20k_voc12aug_20200617_222615-948c68ab.pth',
    },
    'CCNet': {
        'config_file': 'configs/ccnet/ccnet_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/ccnet_r50-d8_512x512_20k_voc12aug_20200617_193212-fad81784.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ccnet/ccnet_r50-d8_512x512_20k_voc12aug/ccnet_r50-d8_512x512_20k_voc12aug_20200617_193212-fad81784.pth',
    },
    'CCNet-r101': {
        'config_file': 'configs/ccnet/ccnet_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/ccnet_r101-d8_512x512_20k_voc12aug_20200617_193212-0007b61d.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ccnet/ccnet_r101-d8_512x512_20k_voc12aug/ccnet_r101-d8_512x512_20k_voc12aug_20200617_193212-0007b61d.pth',
    },
    'DANet': {
        'config_file': 'configs/danet/danet_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/danet_r50-d8_512x512_20k_voc12aug_20200618_070026-9e9e3ab3.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/danet/danet_r50-d8_512x512_20k_voc12aug/danet_r50-d8_512x512_20k_voc12aug_20200618_070026-9e9e3ab3.pth',
    },
    'GCNet': {
        'config_file': 'configs/gcnet/gcnet_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/gcnet_r50-d8_512x512_20k_voc12aug_20200617_165701-3cbfdab1.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/gcnet/gcnet_r50-d8_512x512_20k_voc12aug/gcnet_r50-d8_512x512_20k_voc12aug_20200617_165701-3cbfdab1.pth',
    },
    'GCNet-r101': {
        'config_file': 'configs/gcnet/gcnet_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/gcnet_r101-d8_512x512_20k_voc12aug_20200617_165713-6c720aa9.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/gcnet/gcnet_r101-d8_512x512_20k_voc12aug/gcnet_r101-d8_512x512_20k_voc12aug_20200617_165713-6c720aa9.pth',
    },
    'ANN': {
        'config_file': 'configs/ann/ann_r50-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/ann_r50-d8_512x512_20k_voc12aug_20200617_222246-dfcb1c62.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_20k_voc12aug/ann_r50-d8_512x512_20k_voc12aug_20200617_222246-dfcb1c62.pth',
    },
    'ANN-r101': {
        'config_file': 'configs/ann/ann_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint_file': 'checkpoints/ann_r101-d8_512x512_20k_voc12aug_20200617_222246-2fad0042.pth',
        'download_link': 'https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_20k_voc12aug/ann_r101-d8_512x512_20k_voc12aug_20200617_222246-2fad0042.pth',
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