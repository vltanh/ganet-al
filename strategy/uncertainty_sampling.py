import argparse
import csv
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader, build_dataset

from tools.ganet.post_process import PostProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inp_path',
        required=True,
    )
    parser.add_argument(
        '--out_path',
        required=True,
    )

    parser.add_argument(
        '--cfg',
        required=True,
    )

    parser.add_argument(
        '--teacher_cfg',
        required=True,
    )
    parser.add_argument(
        '--teacher_ckpt',
        required=True,
    )

    parser.add_argument(
        '--student_cfg',
        required=True,
    )
    parser.add_argument(
        '--student_ckpt',
        required=True,
    )

    parser.add_argument(
        '--studentkd_cfg',
        required=True,
    )
    parser.add_argument(
        '--studentkd_ckpt',
        required=True,
    )

    parser.add_argument(
        '--n_sample',
        type=int,
        required=True,
    )
    return parser.parse_args()


def adjust_result(lanes, centers, crop_bbox, img_shape, points_thr):
    h_img, w_img = img_shape[:2]
    ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
    ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
    offset_x, offset_y = crop_bbox[:2]

    results = []
    virtual_centers = []
    cluster_centers = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            cts = []
            for pt in lanes[key]['points']:
                pt[0] = int(pt[0] * ratio_x + offset_x)
                pt[1] = int(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            for ct in lanes[key]['centers']:
                ct[0] = int(ct[0] * ratio_x + offset_x)
                ct[1] = int(ct[1] * ratio_y + offset_y)
                cts.append(tuple(ct))
            # print('lane {} ====== \npoint nums {}'.format(key, len(pts)))
            # print('lane {} ====== \n point coord {}  \nvirtual center coord {}'.format(key, pts, cts))
            if len(pts) > points_thr:
                results.append(pts)
                virtual_centers.append(cts)
        # print('lane number:{}  virtual center number:{}'.format(len(results), len(virtual_centers)))
    if centers is not None:
        for center in centers:
            center_coord = center['center']
            center_coord[0] = int(center_coord[0] * ratio_x + offset_x)
            center_coord[1] = int(center_coord[1] * ratio_y + offset_y)
            cluster_centers.append(tuple(center_coord))

    return results, virtual_centers, cluster_centers


def load_model(cfg, ckpt):
    cfg = Config.fromfile(cfg)
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    load_checkpoint(model, ckpt)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    return cfg, model


def load_data(cfg, path):
    cfg = Config.fromfile(cfg)
    cfg.data.test.data_list = path
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    return dataloader


@torch.no_grad()
def infer_single(cfg, model, post_processor, data):
    output = model(
        return_loss=False,
        rescale=False,
        thr=cfg.hm_thr,
        kpt_thr=cfg.kpt_thr,
        cpt_thr=cfg.cpt_thr,
        **data,
    )

    downscale = data['img_metas'].data[0][0]['down_scale']
    img_shape = data['img_metas'].data[0][0]['img_shape']

    lanes, cluster_centers = post_processor(output, downscale)
    lanes, virtual_center, cluster_center = \
        adjust_result(
            lanes=lanes,
            centers=cluster_centers,
            crop_bbox=cfg.crop_bbox,
            img_shape=img_shape,
            points_thr=cfg.points_thr,
        )
    return lanes


def infer(cfg, model, dataloader):
    post_processor = PostProcessor(
        use_offset=True,
        cluster_thr=cfg.cluster_thr,
        cluster_by_center_thr=cfg.cluster_by_center_thr,
        group_fast=cfg.group_fast,
    )
    model.eval()
    return [
        infer_single(cfg, model, post_processor, data)
        for data in tqdm(dataloader)
    ]


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    '''
        https://github.com/talcs/simpledtw/blob/master/simpledtw.py
    '''
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1 - vec2)
            matrix[i + 1, j + 1] = cost + \
                min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def calc_dist(lanes_1, lanes_2):
    max_dist = 0.0
    for lanes_1i in lanes_1:
        min_dist = float("inf")
        for lanes_2j in lanes_2:
            _, dist, _, _, _ = dtw(
                np.array(lanes_1i),
                np.array(lanes_2j)
            )

            if dist < min_dist:
                min_dist = dist
        if min_dist > max_dist:
            max_dist = min_dist
    return max_dist


def calc_uncertainty_score(dist_ss, dist_st):
    return (dist_ss + dist_st) * \
        max(dist_st / (dist_ss + 1e-8), dist_ss / (dist_st + 1e-8))


def uncertainty_sampling(
    path,
    n_sample,
    cfg,
    teacher_cfg,
    teacher_ckpt,
    studentkd_cfg,
    studentkd_ckpt,
    student_cfg,
    student_ckpt,
):
    df = {
        "img_name": [],
        "dist_ss": [],
        "dist_st": [],
        "uncertainty_score": []
    }
    with open(path, "r") as f:
        test_list = f.readlines()
        for names in test_list:
            img_name = names.split()[0]
            df["img_name"].append(img_name)

    teacher_cfg, teacher = load_model(teacher_cfg, teacher_ckpt)
    studentkd_cfg, studentkd = load_model(studentkd_cfg, studentkd_ckpt)
    student_cfg, student = load_model(student_cfg, student_ckpt)

    dataloader = load_data(cfg, path)

    teacher_frame_lanes = infer(teacher_cfg, teacher, dataloader)
    studentkd_frame_lanes = infer(studentkd_cfg, studentkd, dataloader)
    student_frame_lanes = infer(student_cfg, student, dataloader)

    for i in tqdm(range(len(teacher_frame_lanes))):
        dist_st = calc_dist(studentkd_frame_lanes[i], teacher_frame_lanes[i])
        dist_ss = calc_dist(studentkd_frame_lanes[i], student_frame_lanes[i])
        uncertainty_score = calc_uncertainty_score(dist_ss, dist_st)

        df["dist_ss"].append(dist_ss)
        df["dist_st"].append(dist_st)
        df["uncertainty_score"].append(uncertainty_score)

    df = pd.DataFrame(df)

    sorted_df = df.sort_values(by=["uncertainty_score"], ascending=False)
    top_uncertain_names = sorted_df[:n_sample]["img_name"].tolist()

    return top_uncertain_names


if __name__ == '__main__':
    args = parse_args()

    # Sample using strategy
    picked_lines = uncertainty_sampling(
        args.inp_path,
        args.n_sample,
        args.cfg,
        args.teacher_cfg,
        args.teacher_ckpt,
        args.studentkd_cfg,
        args.studentkd_ckpt,
        args.student_cfg,
        args.student_ckpt,
    )

    # Write to output file
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        csv.writer(f).writerows([[x] for x in picked_lines])
