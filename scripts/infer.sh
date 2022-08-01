DATASET=$1
OUT_ROOT=$2
CFG=$3

INPUT_PATH="data/"$DATASET"/list/"$VIDEO_IDX".txt"
 
# ===========================================================

# Infer
python "tools/ganet/"$DATASET"/test_dataset.py" \
    "configs/"$DATASET"/"$CFG".py" \
    ""$OUT_ROOT"/"$CFG"/latest.pth" \
    --test_list $INPUT_PATH \
    --result_dst ""$OUT_ROOT"/"$CFG"/txts/" \
    --show \
    --show_dst ""$OUT_ROOT"/"$CFG"/imgs/"

# ffmpeg \
#     -framerate 30 \
#     -i ""$OUT_ROOT"/"$CFG"/imgs/"$VIDEO_IDX"/pred/%d.png" \
#     ""$OUT_ROOT"/"$CFG"/pred.mp4"