DATASET=$1
STRATEGY=$2
VIDEO_IDX=$3

TEACHER_CFG=$4
STUDENTKD_CFG=$5

N_EPOCH=50

# ===========================================================

OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/0/"
LABELED_POOL="data/"$DATASET"/list_20fps/"$VIDEO_IDX".txt"
TEACHER_CKPT=""$OUT_ROOT"/"$TEACHER_CFG"/latest.pth"
 
# ===========================================================

python tools/train_kd.py \
    "configs/"$DATASET"/"$STUDENTKD_CFG".py" \
    --work-dir ""$OUT_ROOT"/"$STUDENTKD_CFG"/" \
    --teacher-cfg "configs/"$DATASET"/"$TEACHER_CFG".py" \
    --teacher-ckpt $TEACHER_CKPT \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH