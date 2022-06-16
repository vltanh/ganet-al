DATASET=$1
STRATEGY=$2
VIDEO_IDX=$3
CFG=$4

OUT_ROOT="checkpoints/"$DATASET"/"$STRATEGY"/"$VIDEO_IDX"/0/"
INPUT_PATH="data/"$DATASET"/list_10fps/"$VIDEO_IDX".txt"
CKPT=""$OUT_ROOT"/"$CFG"/latest.pth"
 
# ===========================================================

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$CFG".py" \
    $CKPT \
    --test_list $INPUT_PATH \
    --result_dst ""$OUT_ROOT"/"$CFG"/txts/" \
    --show \
    --show_dst ""$OUT_ROOT"/"$CFG"/imgs/"

# # ffmpeg \
# #     -framerate 30 \
# #     -i ""$OUT_ROOT"/"$CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
# #     ""$OUT_ROOT"/"$CFG"/pred.mp4"