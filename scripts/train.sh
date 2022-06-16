DATASET=$1
STRATEGY=$2
VIDEO_IDX=$3
CFG=$4

N_EPOCH=50

OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/0/"
LABELED_POOL="data/"$DATASET"/list_20fps/"$VIDEO_IDX".txt"
 
# ===========================================================

python tools/train.py \
    "configs/"$DATASET"/"$CFG".py" \
    --work-dir ""$OUT_ROOT"/"$CFG"/" \
    --train-list $LABELED_POOL \
    --n_epoch $N_EPOCH