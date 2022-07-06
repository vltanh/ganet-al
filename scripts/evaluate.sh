DATASET=$1
STRATEGY=$2
VIDEO_IDX=$3
ROUND=$4

if [[ $VIDEO_IDX = '' ]]
then
    if [[ $ROUND = '' ]]
    then
        for video_dir in checkpoints/$DATASET/$STRATEGY/*
        do
            video_idx=${video_dir##*/}
            for round_dir in $video_dir/*
            do
                ./tools/ganet/$DATASET/evaluate/evaluate \
                    -a data/$DATASET/txt_labels/ \
                    -d $round_dir/txts/ \
                    -i data/$DATASET/ \
                    -l data/$DATASET/list/$video_idx.txt \
                    -w 30 \
                    -t 0.5 \
                    -c 1080 \
                    -r 1920 \
                    -f 1 \
                    -o $round_dir/eval.txt
            done
        done
    else
        for video_dir in checkpoints/$DATASET/$STRATEGY/*
        do
            video_idx=${video_dir##*/}
            round_dir=$video_dir/$ROUND
            ./tools/ganet/$DATASET/evaluate/evaluate \
                -a data/$DATASET/txt_labels/ \
                -d $round_dir/txts/ \
                -i data/$DATASET/ \
                -l data/$DATASET/list/$video_idx.txt \
                -w 30 \
                -t 0.5 \
                -c 1080 \
                -r 1920 \
                -f 1 \
                -o $round_dir/eval.txt
        done
    fi
else
    if [[ $ROUND = '' ]]
    then
        for round_dir in checkpoints/$DATASET/$STRATEGY/$VIDEO_IDX/*
        do
            ./tools/ganet/$DATASET/evaluate/evaluate \
                -a data/$DATASET/txt_labels/ \
                -d $round_dir/txts/ \
                -i data/$DATASET/ \
                -l data/$DATASET/list/$VIDEO_IDX.txt \
                -w 30 \
                -t 0.5 \
                -c 1080 \
                -r 1920 \
                -f 1 \
                -o $round_dir/eval.txt
        done
    else
        round_dir=checkpoints/$DATASET/$STRATEGY/$VIDEO_IDX/$ROUND
        ./tools/ganet/$DATASET/evaluate/evaluate \
            -a data/$DATASET/txt_labels/ \
            -d $round_dir/txts/ \
            -i data/$DATASET/ \
            -l data/$DATASET/list/$VIDEO_IDX.txt \
            -w 30 \
            -t 0.5 \
            -c 1080 \
            -r 1920 \
            -f 1 \
            -o $round_dir/eval.txt
    fi
fi