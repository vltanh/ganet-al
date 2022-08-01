#!/bin/bash -e

DATASET=$1
VIDEO_IDX=$3

STRATEGY="random"
INIT_STRATEGY=$2

CFG=$4

N_ROUND=10
N_SAMPLE=100

N_EPOCH=50

INIT_ROOT="checkpoints/"$DATASET"/init_"$INIT_STRATEGY"/"$VIDEO_IDX"/"
OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/"

FULL_VID_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
VAL_VID_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

unlabeled_pool=""$INIT_ROOT"/unlabeled.txt"
checkpoint=""$INIT_ROOT"/"$CFG"/latest.pth"

for (( round_id=1; round_id <= $N_ROUND; round_id++ ))
do
    round_out_root=""$OUT_ROOT"/"$round_id"/"

    labeled_pool=""$round_out_root"/labeled.txt"

    # Random sampling
    python strategy/random_sampling.py \
        --inp_path $unlabeled_pool \
        --out_path $labeled_pool \
        --n_sample $N_SAMPLE

    # Remove used samples
    new_unlabeled_pool=""$round_out_root"/unlabeled.txt"

    python active_learning/remove_used.py \
        --old_list_path $unlabeled_pool \
        --used_list_path $labeled_pool \
        --new_list_path $new_unlabeled_pool

    unlabeled_pool=$new_unlabeled_pool

    # Train detector
    python tools/train.py \
        "configs/"$DATASET"/"$CFG".py" \
        --work-dir ""$round_out_root"/"$CFG"/" \
        --train-list $labeled_pool \
        --load-from $checkpoint \
        --n_epoch $N_EPOCH

    checkpoint=""$round_out_root"/"$CFG"/latest.pth"

    # Inference
    python "tools/ganet/"$DATASET"/test_dataset.py" \
        configs/"$DATASET"/"$CFG".py \
        $checkpoint \
        --test_list $VAL_VID_PATH \
        --result_dst ""$round_out_root"/txts/" \
        --show \
        --show_dst ""$round_out_root"/imgs/"

    # Evaluate
    ./tools/ganet/$DATASET/evaluate/evaluate \
        -a data/$DATASET/txt_labels/ \
        -d ""$round_out_root"/txts/" \
        -i data/$DATASET/ \
        -l data/$DATASET/list/$VIDEO_IDX.txt \
        -w 30 \
        -t 0.5 \
        -c 1080 \
        -r 1920 \
        -f 1 \
        -o ""$round_out_root"/eval.txt"

    # Visualize
    ffmpeg \
        -framerate 30 \
        -i ""$round_out_root"/imgs/"$VIDEO_IDX"/pred/%d.png" \
        "$round_out_root"/pred.mp4
done