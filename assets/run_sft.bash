#!/bin/bash

# ==================================== Pre-defined variables =====================================

TRUE=true
FALSE=false
NULL=""
MODEL_PWD="/media/ExtremeSSD/models/"
MODEL_CHAT_PWD="/media/ExtremeSSD/models/models_chat_template/"
DATA_PWD="/media/ExtremeSSD/llm_data/test_data/"
LOG_PWD="/media/ExtremeSSD/logs/megatron_test"
BF16="bfloat16"
FP16="float16"
FP32="float32"
ADAM="torch.optim.Adam"
ADAMW="torch.optim.AdamW"
SGD="torch.optim.SGD"
PROMPT_DATASET="prompt_response_dataset"
OPENAI_DATASET="openai_format"
OPEN_ASSISTANT_DATAST="open_assistant"
OPEN_MATH_INSTRUCT_DATASET="openmathinstruct2"

# ==================================== User-defined variables ====================================

# ==== Training configuration ====
MAX_EPOCHS=6
MAX_STEPS=1000
VAL_PERIOD=10
VAL_BATCHES=8
VAL_GLOBAL_BATCH_SIZE=64
VAL_MICRO_BATCH_SIZE=4
VAL_AT_START=$TRUE
SEED=42

# ==== Checkpoint configuration ====
ENABLED=$TRUE
CHECKPOINT_DIR="/home/zeyad/nemo-rl/results/sft"
KEEP_TOP_K=1
SAVE_PERIOD=10

# ==== Model configuration ====
MODEL=$MODEL_PWD"qwen2.5-0.5b-instruct"
TOKENIZER_NAME=$MODEL
TEMPLATE_PATH=$MODEL_CHAT_PWD"qwen2.5-0.5b-instruct_chat_template.txt"
TRAIN_GLOBAL_BATCH_SIZE=128
TRAIN_MICRO_BATCH_SIZE=8
MAX_TOTAL_TOKENS=1024
PRECISION=$BF16

# ==== DTensor configuration ====
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1

# ==== Optimizer configuration ====
MAX_GRAD_NORM=1.0
OPTIMIZER=$ADAMW
LR="5.0e-6"
WEIGHT_DECAY="0.1"
BETAS="[0.9,0.98]"
EPS="1e-5"

# ==== Data configuration ====
TRAIN_VAL_RATIO=0.8 # Ratio of training to validation data
MAX_INPUT_SEQ_LENGTH=$MAX_TOTAL_TOKENS
DATASET=$DATA_PWD"arabic_legal/arabic_train_multiturn.json"

# ==== Logging configuration ====
GPU_MONITOR_ENABLE=$TRUE
GPU_MONITOR_COLLECT_INTERVAL=10
GPU_MONITOR_FLUSH_INTERVAL=10

# ==== Cluster(s) configuration ====
GPUS_PER_NODE=4
NUM_NODES=1

# ==== Megatron configuration ====
MEGATRON_ENABLED=$FALSE
if [ "$MEGATRON_ENABLED" = $TRUE ]; then
    echo "Megatron is enabled."
else
    echo "Megatron is not enabled."
fi

MEGATRON_EMPTY_UNUSED_MEMORY_LEVEL=1
MEGATRON_ACTIVATION_CHECKPOINTING=$FALSE
M_TENSOR_PARALLEL_SIZE=2
M_PIPELINE_PARALLEL_SIZE=2
M_CONTEXT_PARALLEL_SIZE=1
PIPELINE_DTYPE=$PRECISION
NUM_LAYERS_FIRST=$NULL
NUM_LAYERS_LAST=$NULL
SEQ_PARALLEL=$FALSE
ROPE_FUSION=$TRUE

# ==== Megatron Optimizer ====
MEGATRON_OPTIMIZER=$ADAM
MEGATRON_LR="5.0e-6"
MEGATRON_MIN_LR="4.9999e-6"
MEGATRON_WEIGHT_DECAY="0.1"
MEGATRON_BF16=$FALSE
MEGATRON_FP16=$FALSE
MEGATRON_PARAMS_DTYPE=$FP32
ADAM_BETA1="0.9"
ADAM_BETA2="0.98"
ADAM_EPS="1e-5"
SGD_MOMENTUM="0.9"
DISTRIBUTED_OPTIMIZER=$TRUE
PRECISION_OPTIMIZER=$TRUE
CLIP_GRADIENTS=$MAX_GRAD_NORM

# ==== Megatron Shceduler ====
LR_WARMUP_ITERS=50
LR_WARMUP_INIT="4.9999e-6"
LR_DECAY_STYLE="constant"
WEIGHT_DECAY_STYLE="constant"
LR_DECAY_ITERS=$NULL
START_WEIGHT_DECAY=$WEIGHT_DECAY
END_WEIGHT_DECAY=$WEIGHT_DECAY

# ==== Megatron DDP ====
GRAD_REDUCE_FP32=$FALSE
OVERLAP_GRAD_REDUCE=$TRUE
OVERLAP_PARAM_GATHER=$TRUE
AVERAGE_IN_COLLECTIVE=$TRUE
DP_SHARDING_STRATEGY="optim_grads_params"

# ==== Logic argument for data format fitting ====
DATASET_TYPE=$OPENAI_DATASET # Options: prompt_response_dataset, openai_format, open_assistant, openmathinstruct2

# ==== Prepare data splitting for training and validation ====
# Exit immediately on error
set -e

# Check if files exist
if [ ! -f "$DATASET" ]; then
  echo "❌ Dataset not found: $DATASET"
  exit 1
fi

readarray -t SPLIT_PATHS < <(python split_dataset.py "$DATASET" "$TRAIN_VAL_RATIO")
TRAIN_PATH="${SPLIT_PATHS[0]}"
VAL_PATH="${SPLIT_PATHS[1]}"

OPTIM=""

if [ "$OPTIMIZER" = $ADAMW ]; then
    OPTIM+=" policy.optimizer.name=$ADAMW"
    OPTIM+=" +policy.optimizer.kwargs.betas=$BETAS"
    OPTIM+=" +policy.optimizer.kwargs.eps=$EPS"
elif [ "$OPTIMIZER" = $ADAM ]; then
    OPTIM+=" policy.optimizer.name=$ADAM"
    OPTIM+=" +policy.optimizer.kwargs.betas=$BETAS"
    OPTIM+=" +policy.optimizer.kwargs.eps=$EPS"
elif [ "$OPTIMIZER" = $SGD ]; then
    OPTIM+=" policy.optimizer.name=$SGD"
fi

ARGS=""

if [ "$DATASET_TYPE" = "prompt_response_dataset" ]; then
    ARGS+=" data.dataset_name=prompt_response_dataset"
    ARGS+=" +data.train_data_path=$TRAIN_PATH"
    ARGS+=" +data.val_data_path=$VAL_PATH"
    ARGS+=" +data.input_key=prompt"
    ARGS+=" +data.output_key=response"

elif [ "$DATASET_TYPE" = "openai_format" ]; then
    ARGS+=" data.dataset_name=openai_format"
    ARGS+=" +data.train_data_path=$TRAIN_PATH"
    ARGS+=" +data.val_data_path=$VAL_PATH"
    ARGS+=" +data.chat_key=messages"
    ARGS+=" +data.system_key=null"
    ARGS+=" +data.system_prompt=null"

elif [ "$DATASET_TYPE" = "open_assistant" ]; then
    ARGS+=" data.dataset_name=open_assistant"
    # No train/val split needed, dataset loads internally

elif [ "$DATASET_TYPE" = "openmathinstruct2" ]; then
    ARGS+=" data.dataset_name=openmathinstruct2"
    ARGS+=" +data.split=train"
    ARGS+=" +data.output_key=output"
    ARGS+=" +data.prompt_file=/path/to/openmathinstruct2/prompts.json"
fi

# ==== Run python script with config overrides ====
RAY_ADDRESS=auto uv run python examples/run_sft.py \
    sft.max_num_epochs=${MAX_EPOCHS} \
    sft.max_num_steps=${MAX_STEPS} \
    sft.val_period=${VAL_PERIOD} \
    sft.val_batches=${VAL_BATCHES} \
    sft.val_global_batch_size=${VAL_GLOBAL_BATCH_SIZE} \
    sft.val_micro_batch_size=${VAL_MICRO_BATCH_SIZE} \
    sft.val_at_start=${VAL_AT_START} \
    sft.seed=${SEED} \
    checkpointing.enabled=$ENABLED \
    checkpointing.checkpoint_dir=$CHECKPOINT_DIR \
    checkpointing.keep_top_k=${KEEP_TOP_K} \
    checkpointing.save_period=${SAVE_PERIOD} \
    policy.model_name=$MODEL \
    policy.tokenizer.name=$TOKENIZER_NAME \
    policy.tokenizer.chat_template=$TEMPLATE_PATH \
    policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
    policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
    policy.max_total_sequence_length=${MAX_TOTAL_TOKENS} \
    policy.precision=$PRECISION \
    policy.dtensor_cfg.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    policy.dtensor_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE} \
    policy.max_grad_norm=${MAX_GRAD_NORM} \
    $OPTIM \
    policy.optimizer.kwargs.lr=$LR \
    policy.optimizer.kwargs.weight_decay=$WEIGHT_DECAY \
    data.max_input_seq_length=${MAX_INPUT_SEQ_LENGTH} \
    $ARGS \
    logger.log_dir=$LOG_PWD \
    logger.monitor_gpus=$GPU_MONITOR_ENABLE \
    logger.gpu_monitoring.collection_interval=${GPU_MONITOR_COLLECT_INTERVAL} \
    logger.gpu_monitoring.flush_interval=${GPU_MONITOR_FLUSH_INTERVAL} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    cluster.num_nodes=${NUM_NODES} \
    policy.megatron_cfg.enabled=$MEGATRON_ENABLED \
    policy.megatron_cfg.empty_unused_memory_level=${MEGATRON_EMPTY_UNUSED_MEM_LEVEL} \
    policy.megatron_cfg.activation_checkpointing=$MEGATRON_ACTIVATION_CHECKPOINTING \
    policy.megatron_cfg.tensor_model_parallel_size=${M_TENSOR_PARALLEL_SIZE} \
    policy.megatron_cfg.pipeline_model_parallel_size=${M_PIPELINE_PARALLEL_SIZE} \
    policy.megatron_cfg.context_parallel_size=${M_CONTEXT_PARALLEL_SIZE} \
    policy.megatron_cfg.pipeline_dtype=$PIPELINE_DTYPE \
    +policy.megatron_cfg.num_layers_in_first_pipeline_stage=${NUM_LAYERS_FIRST:-null} \
    +policy.megatron_cfg.num_layers_in_last_pipeline_stage=${NUM_LAYERS_LAST:-null} \
    policy.megatron_cfg.sequence_parallel=$SEQ_PARALLEL \
    policy.megatron_cfg.apply_rope_fusion=$ROPE_FUSION \
    policy.megatron_cfg.optimizer.optimizer=$MEGATRON_OPTIMIZER \
    policy.megatron_cfg.optimizer.lr=$MEGATRON_LR \
    policy.megatron_cfg.optimizer.min_lr=$MEGATRON_MIN_LR \
    policy.megatron_cfg.optimizer.weight_decay=$WEIGHT_DECAY \
    policy.megatron_cfg.optimizer.bf16=$MEGATRON_BF16 \
    policy.megatron_cfg.optimizer.fp16=$MEGATRON_FP16 \
    policy.megatron_cfg.optimizer.params_dtype=$MEGATRON_PARAMS_DTYPE \
    policy.megatron_cfg.optimizer.adam_beta1=$ADAM_BETA1 \
    policy.megatron_cfg.optimizer.adam_beta2=$ADAM_BETA2 \
    policy.megatron_cfg.optimizer.adam_eps=$EPS \
    policy.megatron_cfg.optimizer.sgd_momentum=$SGD_MOMENTUM \
    policy.megatron_cfg.optimizer.use_distributed_optimizer=$DISTRIBUTED_OPTIMIZER \
    policy.megatron_cfg.optimizer.use_precision_aware_optimizer=$PRECISION_OPTIMIZER \
    policy.megatron_cfg.optimizer.clip_grad=${CLIP_GRADIENTS} \
    policy.megatron_cfg.scheduler.start_weight_decay=$WEIGHT_DECAY \
    policy.megatron_cfg.scheduler.end_weight_decay=$WEIGHT_DECAY \
    policy.megatron_cfg.scheduler.weight_decay_incr_style=$WEIGHT_DECAY_STYLE \
    policy.megatron_cfg.scheduler.lr_decay_style=$LR_DECAY_STYLE \
    +policy.megatron_cfg.scheduler.lr_decay_iters=$LR_DECAY_ITERS \
    policy.megatron_cfg.scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    policy.megatron_cfg.scheduler.lr_warmup_init=$LR_WARMUP_INIT \
    policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=$GRAD_REDUCE_FP32 \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=$OVERLAP_GRAD_REDUCE \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=$OVERLAP_PARAM_GATHER \
    policy.megatron_cfg.distributed_data_parallel_config.average_in_collective=$AVERAGE_IN_COLLECTIVE \
    policy.megatron_cfg.distributed_data_parallel_config.data_parallel_sharding_strategy=$DP_SHARDING_STRATEGY 
    