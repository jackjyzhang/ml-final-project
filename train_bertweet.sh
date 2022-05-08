#!/bin/bash
module load anaconda/2021a; source activate jack

# MODEL=vinai/bertweet-base
MODEL=vinai/bertweet-large
EPOCHS=10
BATCHSIZE=16
LR=0.00001
AUG_STR="_aug6-0-1"

WANDB_DIR=/home/gridsan/jzhang2/repos/nl-command/wandb \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_bertweet.py \
    --data-dir data/hate_speech-3splits${AUG_STR} \
    --output-dir ckpt/${MODEL}_lr${LR}_bz${BATCHSIZE}_schedule${AUG_STR}_lossbal2-1-1\
    --model $MODEL \
    --epochs $EPOCHS \
    --batchsize $BATCHSIZE \
    --learning-rate $LR \
    --num-labels 3 \
    --test-eval \
    --schedule \
    --loss-balance 2 1 1


# python ~/loop.py