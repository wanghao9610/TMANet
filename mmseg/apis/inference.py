import matplotlib.pyplot as plt
import torch
import os

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_segmentor(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            sequence_num = 2
            # sequence_index = 2  # 2 for Citys, 1 for Camvid
            # sequence_suffix = '_leftImg8bit.png'  # for Citys
            # sequence_dir = 'data/cityscapes/leftImg8bit_sequence'
            sequence_index = 1  # 2 for Citys, 1 for Camvid
            sequence_dir = 'data/camvid/images_sequence'
            sequence_suffix = '.png'  # for Camvid
            img_name = results['img'].split('/')[-1].split('.')[0]
            # from ipdb import set_trace
            # set_trace()
            # img_name = '/'.join(results['img'].split('/')[3:]).split('.')[0]
            results['filename'] = results['img']
            current_frame = img_name.split('_')[sequence_index]  # 2 for Citys, 1 for Camvid
            frames = []
            sequence_imgs = []
            for index in range(1, sequence_num + 1):
                if 'f' in current_frame:
                    frame = str(int(current_frame.replace('f', '')) - index).zfill(len(current_frame) - 1)
                    frame_name = '_'.join(img_name.split('_')[
                                          :sequence_index]) + f'_f{frame}' + sequence_suffix
                    frame_path = os.path.join(sequence_dir, frame_name)
                else:
                    frame = str(int(current_frame) - index).zfill(len(current_frame))
                    frame_name = '_'.join(img_name.split('_')[
                                          :sequence_index]) + f'_{frame}' + sequence_suffix
                    frame_path = os.path.join(sequence_dir, frame_name)
                frames.append(frame_path)
                frame = mmcv.imread(frame_path)
                sequence_imgs.append(frame)
                frames.append(frame_path)
            results['ori_filename'] = results['img']
            results['sequence_filename'] = frames
            results['sequence_imgs'] = sequence_imgs
        else:
            results['filename'] = None
            results['ori_filename'] = None
            results['sequence_filename'] = None
            results['sequence_imgs'] = None
        # img = mmcv.imread(results['img'])
        img = mmcv.imread(os.path.join(sequence_dir, img_name + '.png'))
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, out_file=None, palette=None, fig_size=(15, 10)):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, out_file=out_file, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()
