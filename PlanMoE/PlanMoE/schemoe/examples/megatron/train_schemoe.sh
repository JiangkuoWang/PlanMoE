#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

VOCAB_FILE=/workspace/datasets/megatron/gpt2-config/vocab.json
MERGE_FILE=/workspace/datasets/megatron/gpt2-config/merges.txt
DATA_PATH=/workspace/datasets/megatron/wikipedia/wikipedia_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 4096
    --num-attention-heads 16
    --seq-length 1024
    --max-position-embeddings 1024
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 32
    --disable-bias-linear
    --train-iters 60000
    --weight-decay 1e-2
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 0.00015
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .01 
    --lr-decay-iters 320000
    --transformer-impl local
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-token-dispatcher-type alltoall
    --expert-model-parallel-size $WORLD_SIZE
    --moe-expert-capacity-factor 1.5
    --schemoe
    --schemoe-overlap-degree 1
    --schemoe-compress-name 'no'
    --schemoe-comm-name 'naive'
)


DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)

torchrun ${DISTRIBUTED_ARGS[@]} ./pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee ./logs/schemoe.log

