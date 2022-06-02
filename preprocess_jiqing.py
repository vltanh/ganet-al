import os
import glob
from tqdm import tqdm

INP_DIR = 'data/jiqing/Lane_Parameters'
OUT_DIR = 'data/jiqing/txt_label'

os.makedirs(OUT_DIR, exist_ok=True)

anno_fns = glob.glob(INP_DIR + '/*/*.txt')

for anno_fn in tqdm(anno_fns):
    with open(anno_fn, 'r') as f:
        lanes = []
        for line in f.readlines():
            line = line.strip().split(':')[-1].replace(')(', ') (').split()
            lane = [list(map(int, x[1:-1].split(','))) for x in line]
            lane = sum(lane, [])
            lanes.append(lane)

    out_fn = anno_fn.replace(INP_DIR, OUT_DIR)
    dirname, basename = os.path.split(out_fn)

    os.makedirs(dirname, exist_ok=True)
    with open(out_fn, 'w') as f:
        for lane in lanes:
            lane = ' '.join(map(str, lane))
            f.write(lane + '\n')
