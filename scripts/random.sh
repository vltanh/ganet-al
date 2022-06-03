#!/bin/bash -e

STRATEGY="random"

DATASET="jiqing"
VIDEO_IDX=$1

MODEL=res101s4

N_ROUND=10
N_INIT=200
N_SAMPLE=200

N_EPOCH=25

OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/"

FULL_VID_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

unlabeled_pool=$FULL_VID_PATH

# Initial random
round_out_root=""$OUT_ROOT"/0/"

labeled_pool=""$round_out_root"/labeled.txt"

python strategy/random_sampling.py \
    --inp_path $unlabeled_pool \
    --out_path $labeled_pool \
    --n_sample $N_INIT

# Remove from unlabeled pool
new_unlabeled_pool=""$round_out_root"/unlabeled.txt"

python active_learning/remove_used.py \
    --old_list_path $unlabeled_pool \
    --used_list_path $labeled_pool \
    --new_list_path $new_unlabeled_pool

unlabeled_pool=$new_unlabeled_pool

# Train detector
python tools/train.py \
    "configs/"$DATASET"/"$MODEL".py" \
    --work-dir ""$round_out_root"/"$MODEL"/" \
    --train-list $labeled_pool \
    --n_epoch $N_EPOCH

checkpoint=""$round_out_root"/"$MODEL"/latest.pth"

# # Inference
# python "tools/ganet/"$DATASET"/test_dataset.py" \
#     configs/"$DATASET"/"$MODEL".py \
#     $checkpoint \
#     --test_list $FULL_VID_PATH \
#     --result_dst ""$round_out_root"/txts/" \
#     --show \
#     --show_dst ""$round_out_root"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$round_out_root"/imgs/pred/%d.png" \
#     "$round_out_root"/pred.mp4

for (( round_id=1; round_id <= $N_ROUND; round_id++ ))
do
    round_out_root=""$OUT_ROOT"/"$round_id"/"

    labeled_pool=""$round_out_root"/labeled.txt"

    # Random sampling
    python strategy/random_sampling.py \
        --inp_path $unlabeled_pool \
        --out_path $labeled_pool \
        --n_sample $N_INIT

    # Remove used samples
    new_unlabeled_pool=""$round_out_root"/unlabeled.txt"

    python active_learning/remove_used.py \
        --old_list_path $unlabeled_pool \
        --used_list_path $labeled_pool \
        --new_list_path $new_unlabeled_pool

    unlabeled_pool=$new_unlabeled_pool

    # Train detector
    python tools/train.py \
        "configs/"$DATASET"/"$MODEL".py" \
        --work-dir ""$round_out_root"/"$MODEL"/" \
        --train-list $labeled_pool \
        --load-from $checkpoint \
        --n_epoch $N_EPOCH

    checkpoint=""$round_out_root"/"$MODEL"/latest.pth"
done