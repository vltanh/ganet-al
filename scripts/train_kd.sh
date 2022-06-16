STRATEGY="kd"

DATASET="jiqing"
VIDEO_IDX=$1

TEACHER_CFG=res101s4
STUDENTKD_CFG=res18s8_kd

N_EPOCH=50

OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/"

FULL_VID_PATH="data/"$DATASET"/list_10fps/"$VIDEO_IDX".txt"
VAL_VID_PATH="data/"$DATASET"/list_20fps/"$VIDEO_IDX".txt"
 
# ===========================================================

round_out_root=""$OUT_ROOT"/0/"

labeled_pool=$VAL_VID_PATH

# === TEACHER

# Train
python tools/train.py \
    "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --work-dir ""$round_out_root"/"$TEACHER_CFG"/" \
    --train-list $labeled_pool \
    --n_epoch $N_EPOCH

teacher_ckpt=""$round_out_root"/"$TEACHER_CFG"/latest.pth"

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$TEACHER_CFG".py" \
    $teacher_ckpt \
    --test_list $FULL_VID_PATH \
    --result_dst ""$round_out_root"/"$TEACHER_CFG"/txts/" \
    --show \
    --show_dst ""$round_out_root"/"$TEACHER_CFG"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$round_out_root"/"$TEACHER_CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$round_out_root"/"$TEACHER_CFG"/pred.mp4"

# === STUDENT-KD

# Train
python tools/train_kd.py \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    --work-dir ""$round_out_root"/"$STUDENTKD_CFG"/" \
    --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --teacher-ckpt $teacher_ckpt \
    --train-list $labeled_pool \
    --n_epoch $N_EPOCH

studentkd_ckpt=""$round_out_root"/"$STUDENTKD_CFG"/latest.pth"

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    $studentkd_ckpt \
    --test_list $FULL_VID_PATH \
    --result_dst ""$round_out_root"/"$STUDENTKD_CFG"/txts/" \
    --show \
    --show_dst ""$round_out_root"/"$STUDENTKD_CFG"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$round_out_root"/"$STUDENTKD_CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$round_out_root"/"$STUDENTKD_CFG"/pred.mp4"