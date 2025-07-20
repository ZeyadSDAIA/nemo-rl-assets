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

## Total number of steps to train will equal
## min((max_num_epochs * len(train_dataloader)), max_num_steps)
MAX_EPOCHS=1
MAX_STEPS=1000
VAL_PERIOD=10 # How often to run validation
VAL_BATCHES=8 # Number of batches to use for validation
VAL_GLOBAL_BATCH_SIZE=64 # Global batch size for validation
VAL_MICRO_BATCH_SIZE=4 # Number of batches per GPU
VAL_AT_START=$TRUE # Run validation before training
SEED=42 # Random seed for reproducibility

# ==== Checkpoint configuration ====
ENABLED=$TRUE # Enable checkpointing
#If enabled and checkpoint exists in same directory, will load current checkpoint
CHECKPOINT_DIR="/home/zeyad/nemo-rl/results/sft/test" # Checkpoint directory
KEEP_TOP_K=1 # Number of top checkpoints to keep
SAVE_PERIOD=10 # How often to save checkpoints

# ==== Model configuration ====
MODEL=$MODEL_PWD"qwen2.5-0.5b-instruct" # Model to use
TOKENIZER_NAME=$MODEL
TEMPLATE_PATH=$MODEL_CHAT_PWD"qwen2.5-0.5b-instruct_chat_template.txt" # Model chat template to use
TRAIN_GLOBAL_BATCH_SIZE=128 # Global batch size for training
TRAIN_MICRO_BATCH_SIZE=8 # Number of batches per GPU
MAX_TOTAL_TOKENS=1024 # Maximum total tokens in a sequence
PRECISION=$BF16 # Precision to use for training (bfloat16, float16, float32)

# ==== DTensor configuration ====
TENSOR_PARALLEL_SIZE=2 # Tensor parallel size
CONTEXT_PARALLEL_SIZE=1 # Context parallel size

# ==== Optimizer configuration ====
MAX_GRAD_NORM=1.0 # Maximum gradient norm for clipping, used for stability
OPTIMIZER=$ADAM # Optimizer to use (ADAMW, ADAM, SGD)
LR="5.0e-6" # Learning rate
WEIGHT_DECAY="0.1" # Weight decay for regularization
BETAS="[0.9,0.98]" # Adam(W) optimizer betas
EPS="1e-5" # Adam(W) optimizer epsilon for numerical stability

# ==== Data configuration ====
TRAIN_VAL_RATIO=0.8 # Ratio of training to validation data
MAX_INPUT_SEQ_LENGTH=$MAX_TOTAL_TOKENS
DATASET=$DATA_PWD"arabic_legal/arabic_train_multiturn.json"

# ==== Logging configuration ====
GPU_MONITOR_ENABLE=$TRUE # Enable GPU monitoring
GPU_MONITOR_COLLECT_INTERVAL=10 # Interval for collecting GPU metrics in seconds
GPU_MONITOR_FLUSH_INTERVAL=10 # Interval for flushing GPU metrics to disk in seconds

# ==== Cluster(s) configuration ====
GPUS_PER_NODE=4 # Number of GPUs per node
NUM_NODES=1 # Number of nodes in the cluster

# ==== Megatron configuration ====
MEGATRON_ENABLED=$FALSE # Enable Megatron features
if [ "$MEGATRON_ENABLED" = $TRUE ]; then
    echo "Megatron is enabled."
else
    echo "Megatron is NOT enabled."
fi

MEGATRON_EMPTY_UNUSED_MEMORY_LEVEL=1 # Controls how aggressively unused memory is freed
MEGATRON_ACTIVATION_CHECKPOINTING=$FALSE # Enable activation checkpointing to save memory
M_TENSOR_PARALLEL_SIZE=2 # Tensor parallel size for Megatron
M_PIPELINE_PARALLEL_SIZE=2 # Pipeline parallel size for Megatron
M_CONTEXT_PARALLEL_SIZE=1 # Context parallel size for Megatron
PIPELINE_DTYPE=$PRECISION # Data type for pipeline parallelism (bfloat16, float16, float32)
NUM_LAYERS_FIRST=$NULL # Number of layers in the first pipeline stage, null for automatic calculation
NUM_LAYERS_LAST=$NULL # Number of layers in the last pipeline stage, null for automatic calculation
SEQ_PARALLEL=$FALSE # Enable sequence parallelism (Enabling providesds ~20% performance boost)
ROPE_FUSION=$TRUE # Enable RoPE fusion for improved performance

# ==== Megatron Optimizer ====
MEGATRON_MIN_LR="4.9999e-6" # Minimum learning rate for Megatron optimizer
MEGATRON_PARAMS_DTYPE=$FP32 # Data type for model parameters in Megatron optimizer
ADAM_BETA1="0.9" # Adam optimizer beta1
ADAM_BETA2="0.98" # Adam optimizer beta2
ADAM_EPS="1e-5" # Adam optimizer epsilon for numerical stability
SGD_MOMENTUM="0.9" # SGD optimizer momentum
DISTRIBUTED_OPTIMIZER=$TRUE # Uses various optimizer logic to enhance training
PRECISION_OPTIMIZER=$TRUE # Adapts optimizer logic to chosen precision
CLIP_GRADIENTS=$MAX_GRAD_NORM # Enable gradient clipping to prevent exploding gradients

# ==== Megatron Shceduler ====
LR_WARMUP_ITERS=50 # Number of iterations for learning rate warmup
LR_WARMUP_INIT=$MEGATRON_MIN_LR # Initial learning rate for warmup
LR_DECAY_STYLE="constant" # Learning rate decay style (constant, linear, cosine)
WEIGHT_DECAY_STYLE="constant" # Weight decay style (constant, linear, cosine)
LR_DECAY_ITERS=$NULL # Number of iterations for learning rate decay, null for automatic calculation
START_WEIGHT_DECAY=$WEIGHT_DECAY # Initial weight decay for the scheduler
END_WEIGHT_DECAY=$WEIGHT_DECAY # Final weight decay for the scheduler

# ==== Megatron DDP ====
GRAD_REDUCE_FP32=$FALSE # Use FP32 for gradient reduction to improve numerical stability
OVERLAP_GRAD_REDUCE=$TRUE # Start reducing gradients while computing forward pass
OVERLAP_PARAM_GATHER=$TRUE # Start gathering parameters while computing backward pass
AVERAGE_IN_COLLECTIVE=$TRUE # Average gradients in collective communication
DP_SHARDING_STRATEGY="optim_grads_params" # Data parallel sharding strategy (optim_grads_params, optim_grads, optim_params, none)

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

# ========== Ray Start and Auto-Cleanup ==========

export RAY_TMP_DIR="/tmp/ray_session_$RANDOM"
mkdir -p "$RAY_TMP_DIR"

# Trap Ctrl+C or script termination to stop Ray cleanly
cleanup_ray() {
  echo "🛑 Caught interruption. Stopping Ray..."
  uv run ray stop --force || true
  rm -rf "$RAY_TMP_DIR" || true
}
trap cleanup_ray INT TERM EXIT

# Start Ray head
uv run ray start --head --temp-dir="$RAY_TMP_DIR" --disable-usage-stats

# Extract Ray address
RAY_ADDRESS=$(find "$RAY_TMP_DIR" -type f -name "ray_start.log" -exec grep -oP '(?<=--address=)\S+' {} \; | head -n 1)
if [ -z "$RAY_ADDRESS" ]; then
  echo "⚠️ Could not detect Ray address automatically. Using fallback."
  RAY_ADDRESS="127.0.0.1:6379"
fi
export RAY_ADDRESS

echo "✅ Ray head started at $RAY_ADDRESS"
echo "📂 Logs will be in $RAY_TMP_DIR"

# ==== Run python script with config overrides ====
uv run python examples/run_sft.py \
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
    policy.megatron_cfg.empty_unused_memory_level=${MEGATRON_EMPTY_UNUSED_MEMORY_LEVEL} \
    policy.megatron_cfg.activation_checkpointing=$MEGATRON_ACTIVATION_CHECKPOINTING \
    policy.megatron_cfg.tensor_model_parallel_size=${M_TENSOR_PARALLEL_SIZE} \
    policy.megatron_cfg.pipeline_model_parallel_size=${M_PIPELINE_PARALLEL_SIZE} \
    policy.megatron_cfg.context_parallel_size=${M_CONTEXT_PARALLEL_SIZE} \
    policy.megatron_cfg.pipeline_dtype=$PIPELINE_DTYPE \
    +policy.megatron_cfg.num_layers_in_first_pipeline_stage=${NUM_LAYERS_FIRST:-null} \
    +policy.megatron_cfg.num_layers_in_last_pipeline_stage=${NUM_LAYERS_LAST:-null} \
    policy.megatron_cfg.sequence_parallel=$SEQ_PARALLEL \
    policy.megatron_cfg.apply_rope_fusion=$ROPE_FUSION \
    policy.megatron_cfg.optimizer.optimizer=$OPTIMIZER \
    policy.megatron_cfg.optimizer.lr=$LR \
    policy.megatron_cfg.optimizer.min_lr=$MEGATRON_MIN_LR \
    policy.megatron_cfg.optimizer.weight_decay=$WEIGHT_DECAY \
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
    