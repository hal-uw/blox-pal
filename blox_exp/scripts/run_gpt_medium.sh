#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

CHECKPOINT_PATH=checkpoints/gpt2_pretrain
VOCAB_FILE=/global/cfs/cdirs/m4207/song/gpt2-vocab.json
MERGE_FILE=/global/cfs/cdirs/m4207/song/gpt2-merges.txt
DATA_PATH=/global/cfs/cdirs/m4207/song/my-gpt2_text_document

GPT_ARGS="
    --tensor-model-parallel-size $6 \
    --pipeline-model-parallel-size $7 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $8 \
    --global-batch-size 512 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --no-async-tensor-model-parallel-allreduce \
    --blox-setting \
    --is-manual-pipeline $9 \
    --manual-pipeline-list ${10} \
    --job-id ${11}
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

python /global/homes/s/songbian/Megatron-Resource/pretrain_gpt.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

