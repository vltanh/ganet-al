DATASET=$1
VIDEO_IDX=$2
RESULT_DIR=$3

./tools/ganet/$DATASET/evaluate/evaluate \
    -a data/$DATASET/txt_labels/ \
    -d $RESULT_DIR/txts/ \
    -i data/$DATASET/ \
    -l data/$DATASET/list/$VIDEO_IDX.txt \
    -w 30 \
    -t 0.5 \
    -c 1080 \
    -r 1920 \
    -f 1 \
    -o $RESULT_DIR/eval.txt