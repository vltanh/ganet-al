DATASET=$1
VIDEO_IDX=$2

TEACHER_CFG=$3
STUDENTKD_CFG=$4

N_EPOCH=50

# ===========================================================

OUT_ROOT="checkpoints/"$DATASET"/"$VIDEO_IDX"/"
LABELED_POOL="data/"$DATASET"/list/"$VIDEO_IDX".txt"
VAL_POOL="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

# === TEACHER

# Train
python tools/train.py \
    "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --work-dir ""$OUT_ROOT"/"$TEACHER_CFG"/" \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH

teacher_ckpt=""$OUT_ROOT"/"$TEACHER_CFG"/latest.pth"

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$TEACHER_CFG".py" \
    $teacher_ckpt \
    --test_list $VAL_POOL \
    --result_dst ""$OUT_ROOT"/"$TEACHER_CFG"/txts/" \
    --show \
    --show_dst ""$OUT_ROOT"/"$TEACHER_CFG"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$round_out_root"/"$TEACHER_CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$round_out_root"/"$TEACHER_CFG"/pred.mp4"

# === STUDENT-KD

# Train
python tools/train_kd.py \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    --work-dir ""$OUT_ROOT"/"$STUDENTKD_CFG"/" \
    --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --teacher-ckpt $teacher_ckpt \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH

studentkd_ckpt=""$OUT_ROOT"/"$STUDENTKD_CFG"/latest.pth"

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    $studentkd_ckpt \
    --test_list $VAL_POOL \
    --result_dst ""$OUT_ROOT"/"$STUDENTKD_CFG"/txts/" \
    --show \
    --show_dst ""$OUT_ROOT"/"$STUDENTKD_CFG"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$round_out_root"/"$STUDENTKD_CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$round_out_root"/"$STUDENTKD_CFG"/pred.mp4"