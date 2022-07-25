DATASET=$1
VIDEO_IDX=$2
CFG=$3

N_EPOCH=50

OUT_ROOT="checkpoints/"$DATASET"/"$VIDEO_IDX"/"
LABELED_POOL="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

python tools/train.py \
    "configs/"$DATASET"/"$CFG".py" \
    --work-dir ""$OUT_ROOT"/"$CFG"/" \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH