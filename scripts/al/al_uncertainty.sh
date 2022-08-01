#!/bin/bash -e

DATASET=$1
VIDEO_IDX=$3

STRATEGY="random"
INIT_STRATEGY=$2

TEACHER_CFG=$4
STUDENTKD_CFG=$5
STUDENT_CFG=$6

N_ROUND=10
N_SAMPLE=100

N_EPOCH=50

INIT_ROOT="checkpoints/"$DATASET"/init_"$INIT_STRATEGY"/"$VIDEO_IDX"/"
OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/"

FULL_VID_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
VAL_VID_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

unlabeled_pool=""$INIT_ROOT"/unlabeled.txt"
teacher_ckpt=""$round_out_root"/"$TEACHER_CFG"/latest.pth"

# Train student-kd
python tools/train_kd.py \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    --work-dir ""$round_out_root"/"$STUDENTKD_CFG"/" \
    --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --teacher-ckpt $teacher_ckpt \
    --train-list $labeled_pool \
    --n_epoch $N_EPOCH

studentkd_ckpt=""$round_out_root"/"$STUDENTKD_CFG"/latest.pth"

# Train student
python tools/train.py \
    "configs/"$DATASET"/"$STUDENT_CFG".py" \
    --work-dir ""$round_out_root"/"$STUDENT_CFG"/" \
    --train-list $labeled_pool \
    --n_epoch $N_EPOCH

student_ckpt=""$round_out_root"/"$STUDENT_CFG"/latest.pth"

for (( round_id=1; round_id <= $N_ROUND; round_id++ ))
do
    round_out_root=""$OUT_ROOT"/"$round_id"/"

    labeled_pool=""$round_out_root"/labeled.txt"

    # Uncertainty sampling
    python strategy/uncertainty_sampling.py \
        --inp_path $unlabeled_pool \
        --out_path $labeled_pool \
        --cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
        --teacher_cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
        --teacher_ckpt $teacher_ckpt \
        --studentkd_cfg "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
        --studentkd_ckpt $studentkd_ckpt \
        --student_cfg "configs/"$DATASET"/"$STUDENT_CFG".py" \
        --student_ckpt $student_ckpt \
        --n_sample $N_SAMPLE

    # Remove used samples
    new_unlabeled_pool=""$round_out_root"/unlabeled.txt"

    python active_learning/remove_used.py \
        --old_list_path $unlabeled_pool \
        --used_list_path $labeled_pool \
        --new_list_path $new_unlabeled_pool

    unlabeled_pool=$new_unlabeled_pool

    # Train teacher
    python tools/train.py \
        "configs/"$DATASET"/"$TEACHER_CFG".py" \
        --work-dir ""$round_out_root"/"$TEACHER_CFG"/" \
        --train-list $labeled_pool \
        --load-from $teacher_ckpt \
        --n_epoch $N_EPOCH

    teacher_ckpt=""$round_out_root"/"$TEACHER_CFG"/latest.pth"

    # Train student-kd
    python tools/train_kd.py \
        "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
        --work-dir ""$round_out_root"/"$STUDENTKD_CFG"/" \
        --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
        --teacher-ckpt $teacher_ckpt \
        --train-list $labeled_pool \
        --load-from $studentkd_ckpt \
        --n_epoch $N_EPOCH

    studentkd_ckpt=""$round_out_root"/"$STUDENTKD_CFG"/latest.pth"

    # Train student
    python tools/train.py \
        "configs/"$DATASET"/"$STUDENT_CFG".py" \
        --work-dir ""$round_out_root"/"$STUDENT_CFG"/" \
        --train-list $labeled_pool \
        --load-from $student_ckpt \
        --n_epoch $N_EPOCH

    student_ckpt=""$round_out_root"/"$STUDENT_CFG"/latest.pth"

    # Inference
    python "tools/ganet/"$DATASET"/test_dataset.py" \
        configs/"$DATASET"/"$TEACHER_CFG".py \
        $teacher_ckpt \
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