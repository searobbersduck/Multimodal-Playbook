#!/bin/bash
# Pretrain a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_NAME="mcore-llava-mistral-7b-instruct-clip336-pretraining"


WORKSPACE=output
# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# if [[ -z $LOAD_NAME ]]; then
#     echo "Please set LOAD_NAME for input model name."
#     exit 1
# fi

TOKENIZER_MODEL=/workspace/data/mm/model/Mistral-7B-v0.1
if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

echo $FINETUNE_DIR

CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/pretrain_dataset.yaml"

echo DATA_TRAIN: $DATA_TRAIN

DEBUG=1
if [[ $DEBUG -eq 1 ]]; then
    BZ=2
    NW=2
    HD=0.0
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=256
    NW=2
    HD=0.1
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


PACKING_SEQ_LENGTH=$1
PACKING_BUFFER_SIZE=$2
SEQ_LENGTH=$3
DECODER_SEQ_LENGTH=$4
MAX_POSITION_EMBEDDINGS=$5

OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --use-te \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LENGTH} \
    --decoder-seq-length ${DECODER_SEQ_LENGTH} \
    --max-position-embeddings ${DECODER_SEQ_LENGTH} \
    --ffn-hidden-size 14336 \
    --train-iters 20000 \
    --micro-batch-size 1 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format mistral \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 1000 \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --freeze-LM \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 224 \
    --img-w 224 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=mistral_7b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --ckpt-format torch
"


    # --packing-seq-length 8192 \
    # --packing-buffer-size 100 \

    # --packing-seq-length 8192 \
    # --packing-buffer-size 500 \
    # --packing-seq-length 4096 \
    # --packing-buffer-size 100 \

if [ ${PACKING_SEQ_LENGTH} -gt 0 ]; then
    OPTIONS_PACKING_SEQUENCE=" \
    --packing-seq-length ${PACKING_SEQ_LENGTH} \
    --packing-buffer-size ${PACKING_BUFFER_SIZE}"
else
    OPTIONS_PACKING_SEQUENCE=" "
fi

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

LOG_NAME="sequence_packing_disable_unfreeze_vit_llm"

if [ ${PACKING_SEQ_LENGTH} -gt 0 ]; then
    LOG_NAME="sequence_packing_enable_unfreeze_vit_llm_packing_seq_length_${PACKING_SEQ_LENGTH}_buffer_size_${PACKING_BUFFER_SIZE}_mbs${BATCH_SIZE}gbs${BZ}l${NUM_LAYERS}_tp${TP}pp${PP}"
else
    LOG_NAME="sequence_packing_disable_unfreeze_vit_llm_mbs${BATCH_SIZE}gbs${BZ}l${NUM_LAYERS}_tp${TP}pp${PP}"
fi

# OPTIONS_PROFILE=" \
#     --profile \
#     --profile-ranks 0 1 2 3 4 5 6 7 8\
#     --profile-step-start=50 \
#     --profile-step-end=51 \
# "

OPTIONS_PROFILE=" \
    --profile \
    --profile-ranks 0 1 \
    --profile-step-start=100 \
    --profile-step-end=101 \
"

# LOG_NAME="h20_qwen2_vl_big_vit_uneven_pp_mbs${BATCH_SIZE}gbs${GLOBAL_BATCH_SIZE}l${NUM_LAYERS}_tp${TP}pp${PP}_f${FIRST_PP_LAYERS}_l${LAST_PP_LAYERS}_unfreeze_vit_llm"

TEXT_NAME=$6

NSYS_OUTPUT=freeze/nsys_output/${TEXT_NAME}
LOG_OUTPUT=freeze/packing_sql/${TEXT_NAME}

mkdir -p ${NSYS_OUTPUT}
mkdir -p ${LOG_OUTPUT}

# nsys profile -s none -t nvtx,cuda --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o $NSYS_OUTPUT/${LOG_NAME} \
# torchrun ${DISTRIBUTED_ARGS} examples/multimodal/train.py ${OPTIONS} ${OPTIONS_PACKING_SEQUENCE} ${OPTIONS_PROFILE} 2>&1|tee ${LOG_OUTPUT}/nsys_${LOG_NAME}.log

# TAP_WARMUP_STEPS=5 TAP_ACTIVE_STEPS=1 TAP_MODE=auto TAP_BACKWARD_NVTX=true TAP_SAVE_DIR=./tap_result/m1g32gpu2-no-packing \
# torchrun ${DISTRIBUTED_ARGS} examples/multimodal/train.py ${OPTIONS} ${OPTIONS_PACKING_SEQUENCE} ${OPTIONS_PROFILE}

# torchrun ${DISTRIBUTED_ARGS} examples/multimodal/train.py ${OPTIONS} ${OPTIONS_PACKING_SEQUENCE} 2>&1|tee ${LOG_OUTPUT}/nsys_${LOG_NAME}.log

torchrun ${DISTRIBUTED_ARGS} examples/multimodal/train.py ${OPTIONS} ${OPTIONS_PACKING_SEQUENCE}