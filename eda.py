import os
import glob

import pandas as pd
from tqdm import tqdm

all_videos_dir = glob.glob('data/jiqing/txt_label/*')
all_videos_dir = sorted(
    all_videos_dir,
    key=lambda x: int(os.path.split(x)[-1])
)

video_df = {
    'video_id': [],
    'n_frame': [],
    'count': [],
}
for video_dir in tqdm(all_videos_dir):
    video_id = os.path.split(video_dir)[-1]

    frames_path = glob.glob(video_dir + '/*.txt')
    frames_path = sorted(
        frames_path,
        key=lambda x: int(os.path.basename(x)[:-4])
    )

    df = {
        'filepath': [],
        'n_frame': [],
    }
    for frame_path in tqdm(frames_path, leave=False):
        lines = open(frame_path).readlines()
        df['filepath'].append(frame_path)
        df['n_frame'].append(len(lines))

    cnt_dict = \
        pd.DataFrame(df).groupby('n_frame').count().to_dict()['filepath']

    for n_frame, cnt in cnt_dict.items():
        video_df['video_id'].append(video_id)
        video_df['n_frame'].append(n_frame)
        video_df['count'].append(cnt)

pd.DataFrame(video_df).to_csv(
    'jiqing_lane_counts.csv',
    header=False,
    index=False,
)
