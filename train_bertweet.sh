#!/bin/bash

#SBATCH -o train_bertweet-%j.log
#SBATCH --gres=gpu:volta:1

# MODEL=vinai/bertweet-base
MODEL=vinai/bertweet-large
EPOCHS=10
BATCHSIZE=16
LR=0.00001

WANDB_DIR=/home/gridsan/jzhang2/repos/nl-command/wandb \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_bertweet.py \
    --data-dir data/hate_speech-3splits \
    --output-dir ckpt/${MODEL}_lr${LR}_bz${BATCHSIZE}_schedule_lossbal \
    --model $MODEL \
    --epochs $EPOCHS \
    --batchsize $BATCHSIZE \
    --learning-rate $LR \
    --num-labels 3 \
    --test-eval \
    --schedule \
    --loss-balance 12.98 1 4.56 \
    # --no-train \
    # --load-dir ckpt/vinai/bertweet-large_lr0.00001_bz16/epoch_0


# python ~/loop.py