DATASET=$1
VIDEO_IDX=$2

TEACHER_CFG=$3
STUDENTKD_CFG=$4

N_EPOCH=50

# ===========================================================

OUT_ROOT="checkpoints/"$DATASET"/"$VIDEO_IDX"/"
LABELED_POOL="data/"$DATASET"/list/"$VIDEO_IDX".txt"
TEACHER_CKPT=""$OUT_ROOT"/"$TEACHER_CFG"/latest.pth"
 
# ===========================================================

python tools/train_kd.py \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    --work-dir ""$OUT_ROOT"/"$STUDENTKD_CFG"/" \
    --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --teacher-ckpt $TEACHER_CKPT \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH