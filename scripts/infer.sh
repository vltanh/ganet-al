DATASET=$1
VIDEO_IDX=$2
CFG=$3
OUT_ROOT=$4

INPUT_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$CFG".py" \
    ""$OUT_ROOT"/latest.pth" \
    --test_list $INPUT_PATH \
    --result_dst ""$OUT_ROOT"/txts/" \
    --show \
    --show_dst ""$OUT_ROOT"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$OUT_ROOT"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$OUT_ROOT"/pred.mp4"