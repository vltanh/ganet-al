DATASET=$1
VIDEO_IDX=$2
CFG=$3

OUT_ROOT="checkpoints/"$DATASET"/gt/"$VIDEO_IDX"/"
INPUT_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

python "tools/ganet/"$DATASET"/visualize_gt.py" \
    "configs/"$DATASET"/"$CFG".py" \
    --test_list $INPUT_PATH \
    --show_dst ""$OUT_ROOT"/imgs/"

ffmpeg \
    -framerate 30 \
    -i ""$OUT_ROOT"/imgs/%d.png" \
    ""$OUT_ROOT"/pred.mp4"