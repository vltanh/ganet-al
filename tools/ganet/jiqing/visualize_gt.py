import argparse
import os

import cv2
import mmcv
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils.general_utils import mkdir

from tools.ganet.common import COLORS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument(
        'config',
        help='test config file path',
    )
    parser.add_argument(
        '--test_list',
        default=None,
        help='path to list of testing images',
    )
    parser.add_argument(
        '--show_dst',
        default='./work_dirs/culane/watch',
        help='path to save visualized results.',
    )
    args = parser.parse_args()
    return args


def normalize_coords(coords):
    res = []
    for coord in coords:
        res.append((int(coord[0] + 0.5), int(coord[1] + 0.5)))
    return res


def parse_lanes(filename, ext='.jpg'):
    anno_dir = filename.replace(
        'images_train',
        'txt_label'
    ).replace(ext, '.txt')
    annos = []
    with open(anno_dir, 'r') as anno_f:
        lines = anno_f.readlines()
    for line in lines:
        coords = []
        numbers = line.strip().split(' ')
        coords_tmp = [float(n) for n in numbers]

        for i in range(len(coords_tmp) // 2):
            coords.append([coords_tmp[2 * i], coords_tmp[2 * i + 1]])
        annos.append(normalize_coords(coords))
    return annos


def vis_one(filename, width=9):
    img_gt = cv2.imread(filename)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    annos = parse_lanes(filename, ext='.png')
    # print('anno length {}'.format(len(annos)))
    for idx, anno_lane in enumerate(annos):
        PIL.ImageDraw.Draw(img_gt_pil).line(
            xy=anno_lane,
            fill=COLORS[idx + 1],
            width=width,
        )
    img_gt = np.array(img_gt_pil, dtype=np.uint8)
    return img_gt


def single_gpu_test(
    data_loader,
    show=None,
):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        sub_name = data['img_metas'].data[0][0]['sub_img_name']
        if show is not None and show:
            filename = data['img_metas'].data[0][0]['filename']

            img_gt_vis = vis_one(filename, width=10)

            mkdir(show)
            dst_gt_dir = os.path.join(show, f'{i+1}.png')
            cv2.imwrite(dst_gt_dir, img_gt_vis)

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.test_list is not None:
        cfg.data.test.data_list = args.test_list

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    single_gpu_test(
        data_loader=data_loader,
        show=args.show_dst,
    )


if __name__ == '__main__':
    main()
