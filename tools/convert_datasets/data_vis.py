import argparse
import os
import os.path as osp
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('image_path', help='predicted label path.')
    parser.add_argument('label_path', help='predicted label path.')
    parser.add_argument('out_path', help='ground truth label path.')
    args = parser.parse_args()
    return args


naic_class = ['void', 'sky', 'building', 'column_pole', 'road',
              'sidewalk', 'tree', 'sing_symbol', 'fence', 'car',
              'pedestrian', 'bicyclist']

palette = [[0, 0, 0], [128, 128, 128], [128, 0, 0], [192, 192, 128],
           [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128],
           [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]]


def vis(image_path, label_path, out_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(11):
        color_seg = np.zeros((256, 256, 3), dtype=np.uint8)
        color_seg[:, :, :] = palette[i]  # RGB
        color_seg = color_seg[..., ::-1]  # convert to BGR
        png_filename = osp.join(out_path, f'{naic_class[i]}.png')
        cv2.imwrite(png_filename, color_seg)

    is_file = False
    is_dir = False
    if os.path.isdir(image_path):
        img_list = os.listdir(image_path)
        is_dir = True
    elif os.path.isfile(image_path):
        img_list = [image_path]
        is_file = True
    for img_name in tqdm(img_list):
        if is_dir:
            img_file_name = osp.join(image_path, img_name)
        if is_file:
            img_file_name = img_name
        img = Image.open(img_file_name)  # Image open is RGB, opencv is BGR
        img = np.array(img).astype(np.uint8)
        img = img[..., ::-1]  # conver to BGR
        basename = osp.splitext(osp.basename(img_name))[0]
        label_name = osp.join(label_path, f'{basename}.png')
        label = Image.open(osp.join(label_name))
        label = np.array(label)
        label = label + 1
        label[label == 12] = 0
        print(img.shape, label.shape)
        if label.shape != img.shape[:2]:
            img = cv2.resize(img, (label.shape[1], label.shape[0]))
            png_filename = '/data/hwang/proj/video_mmseg/data/camvid_video/360x480/images/' + f'{basename}.png'
            cv2.imwrite(png_filename, img)
        # if label.ndim >= 3 and label.shape[2] >= 3:
        #     label = label[..., ::-1]  # conver to BGR
        # label = label.astype(np.uint8)
        # # label = label.astype(np.uint8)
        # # label[label == 255] = 55
        # # label[label > 3] = label[label > 3] + 3
        # # label[mask] = 255
        # # print(set(label.flatten()))
        # color_seg = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        # for label_id, color in enumerate(palette):
        #     # print(basename, set(label.flatten()), label_id, color)
        #     color_seg[label == label_id, :] = color
        # # convert to BGR
        # color_seg = color_seg[..., ::-1]
        # img = img * 0.3 + color_seg * 0.7
        # img = img.astype(np.uint8)
        # png_filename = osp.join(out_path, f'{basename}.png')
        # cv2.imwrite(png_filename, img)
        # shutil.copy(img_file_name, out_path)
        # png_filename = osp.join(out_path, f'{basename}_over.png')
        # cv2.imwrite(png_filename, img)


def main():
    args = parse_args()
    vis(args.image_path, args.label_path, args.out_path)


if __name__ == '__main__':
    main()
