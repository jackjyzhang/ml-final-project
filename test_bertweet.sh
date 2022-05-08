#!/bin/bash
module load anaconda/2021a; source activate jack

MODEL=vinai/bertweet-large
BATCHSIZE=16

LOAD_DIR=$1
EPC=$2

echo "LOAD_DIR=${LOAD_DIR}"
echo "epoch $EPC"
WANDB_DIR=/home/gridsan/jzhang2/repos/nl-command/wandb \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_bertweet.py \
    --data-dir data/hate_speech-3splits \
    --output-dir ckpt/stub \
    --model $MODEL \
    --batchsize $BATCHSIZE \
    --num-labels 3 \
    --test-eval \
    --no-train \
    --load-dir ${LOAD_DIR}/epoch_$EPC


# python ~/loop.py