#!/bin/bash -e

export CUDA_VISIBLE_DEVICES=3

STRATEGY="random"

DATASET="jiqing"
ROOT=/home/ubuntu/datasets/jiqing_expressway_dataset
VIDEO_IDX=$1

MODEL=res18s8

N_ROUND=10
N_INIT=200
N_SAMPLE=200

N_EPOCH=1

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
    --train_list $labeled_pool \
    --n_epoch $N_EPOCH

checkpoint=""$round_out_root"/"$MODEL"/latest.pth"

# # Inference
# python infer_det.py \
#     configs/resnet34_det.py \
#     --output checkpoints/jiqing_"$VIDEO_IDX"_random/round_0/imgs \
#     --data_root /home/ubuntu/datasets/jiqing_expressway_dataset \
#     --input $FULL_VID_PATH \
#     --test_model $teacher_ckpt \
#     --out_det \
#     --save_img

# ffmpeg \
#     -framerate 30 \
#     -i checkpoints/jiqing_"$VIDEO_IDX"_random/round_0/imgs/%d.jpg \
#     checkpoints/jiqing_"$VIDEO_IDX"_random/round_0/"$VIDEO_IDX"_random_0.mp4

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
        --train_list $labeled_pool \
        --resume-from $checkpoint \
        --n_epoch $N_EPOCH

    checkpoint=""$round_out_root"/"$MODEL"/latest.pth"

    # # Inference
    # python infer_det.py \
    #     configs/resnet34_det.py \
    #     --output checkpoints/jiqing_"$VIDEO_IDX"_random/round_"$round_id"/imgs \
    #     --data_root /home/ubuntu/datasets/jiqing_expressway_dataset \
    #     --num_workers 4 \
    #     --batch_size 256 \
    #     --input $FULL_VID_PATH \
    #     --test_model $teacher_ckpt \
    #     --out_det \
    #     --save_img

    # ffmpeg \
    #     -framerate 30 \
    #     -i checkpoints/jiqing_"$VIDEO_IDX"_random/round_"$round_id"/imgs/%d.jpg \
    #     checkpoints/jiqing_"$VIDEO_IDX"_random/round_"$round_id"/"$VIDEO_IDX"_random_"$round_id".mp4
done