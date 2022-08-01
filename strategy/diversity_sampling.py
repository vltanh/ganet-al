import argparse
import csv
import os

import torch
from tqdm import tqdm

import dino.vision_transformer as vits
from dino import utils

from sklearn.metrics import pairwise_distances
from scipy import stats
import numpy as np

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset


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
        '--n_sample',
        type=int,
        required=True,
    )
    return parser.parse_args()


def init_centers(X, K):
    '''
        https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
    '''
    if K >= len(X):
        return list(range(len(X)))

    # kmeans ++ initialization
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(
            name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def load_data(cfg, path):
    cfg = Config.fromfile(cfg)
    cfg.data.test.data_list = path
    cfg.img_scale = (224, 224)
    cfg.val_al_pipeline[-1].height = 224
    cfg.val_al_pipeline[-1].width = 224
    cfg.val_pipeline[0].pipelines = cfg.val_al_pipeline
    cfg.data.test.pipeline = cfg.val_pipeline

    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    return dataloader


def load_model(agent='teacher'):
    model_info = dict(
        dino_model_path='dino_models/dino_vitbase8_pretrain_full_checkpoint.pth',
        model_arch='vit_base',
        patch_size=8,
        agent=agent,
        model_name='dino_vitbase8',
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vits.__dict__[model_info['model_arch']](
        patch_size=model_info['patch_size'],
        num_classes=0,
    )
    model = model.to(device)
    utils.load_pretrained_weights(
        model,
        model_info['dino_model_path'],
        model_info['agent'],
        model_info['model_arch'],
        model_info['patch_size'],
    )
    model.eval()
    return model, device


def diversity_sampling(
    path,
    n_sample,
    cfg,
    # dino_version,
):
    with open(args.inp_path) as f:
        items = sum(csv.reader(f), [])

    dataloader = load_data(cfg, path)

    dino, device = load_model()

    backbone_features = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            img = data['img'].data[0].to(device)
            backbone_feature = dino(img).cpu()
            backbone_features.append(backbone_feature)

    backbone_features = torch.cat(backbone_features, dim=0).cpu().numpy()

    chosen = init_centers(backbone_features, n_sample)

    return [items[i] for i in chosen]


if __name__ == '__main__':
    args = parse_args()

    # Sample using strategy
    picked_lines = diversity_sampling(
        args.inp_path,
        args.n_sample,
        args.cfg,
        # args.dino_version,
    )

    # Write to output file
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        csv.writer(f).writerows([[x] for x in picked_lines])
