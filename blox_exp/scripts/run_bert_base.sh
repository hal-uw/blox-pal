#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

CHECKPOINT_PATH=checkpoints/bert_pretrain
VOCAB_FILE=/scratch1/08503/rnjain/data-files/bert/bert-large-uncased-vocab.txt
DATA_PATH=scratch1/08503/rnjain/data-files/bert/my-bert_text_sentence

BERT_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size $8 \
    --global-batch-size 256 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --blox-setting \
    --is-manual-pipeline $9 \
    --manual-pipeline-list ${10} \
    --job-id ${11}
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

python /scratch1/08503/rnjain/Megatron-Resource/pretrain_bert.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
