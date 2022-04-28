MODEL=vinai/bertweet-large
BATCHSIZE=16

LOAD_DIR="ckpt/vinai/bertweet-large_lr0.00001_bz16_schedule_lossbal"
EPC=3

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


python ~/loop.py