#!/bin/bash
# ==================================== SFT Training Script ====================================

# Exit on error, undefined variable, pipe failure
set -euo pipefail

# =============================== Logging Functions ===============================
log_info()  { printf "ℹ️  %s %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }
log_warn()  { printf "⚠️  %s %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }
log_error() { printf "❌ %s %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }

# =============================== TRUE CONSTANTS ===============================
declare -r TRUE=true
declare -r FALSE=false
declare -r TRUE_STR="true"
declare -r FALSE_STR="false"
declare -r NULL="null"
declare -r BF16="bfloat16"
declare -r FP16="float16"
declare -r FP32="float32"
declare -r ADAM="torch.optim.Adam"
declare -r ADAMW="torch.optim.AdamW"
declare -r ADAM_FUSED="torch.optim.AdamW"
declare -r SGD="torch.optim.SGD"
declare -r PROMPT_DATASET="prompt_response_dataset"
declare -r OPENAI_DATASET="openai_format"
declare -r OPEN_ASSISTANT_DATAST="open_assistant"
declare -r OPEN_MATH_INSTRUCT_DATASET="openmathinstruct2"
declare -r MODEL_PWD="/media/ExtremeSSD/models/"
declare -r MODEL_CHAT_PWD="/media/ExtremeSSD/models/models_chat_template/"
declare -r DATA_PWD="/media/ExtremeSSD/llm_data/"
declare -r LOG_PWD="/media/ExtremeSSD/logs/megatron_test"

# =============================== User Variables (Mutable) ===============================
# ==== Training configuration ====
MAX_EPOCHS=2
MAX_STEPS=9000
VAL_PERIOD=100
VAL_BATCHES=10
VAL_GLOBAL_BATCH_SIZE=64
VAL_MICRO_BATCH_SIZE=8
VAL_AT_START=$TRUE
SEED=42
CHECKPOINT=$TRUE

# ==== Checkpoint configuration ====
ENABLED=$TRUE
CHECKPOINT_DIR="/home/zeyad/nemo-rl/results/sft"
KEEP_TOP_K=1
SAVE_PERIOD=100

# ==== Model configuration ====
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_NAME=$MODEL
TEMPLATE_PATH="${MODEL_CHAT_PWD}qwen2.5-1.5b-instruct_chat_template.txt"
TRAIN_GLOBAL_BATCH_SIZE=64
TRAIN_MICRO_BATCH_SIZE=8
MAX_TOTAL_TOKENS=1024
PRECISION=$BF16

# ==== DTensor configuration ====
TENSOR_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

# ==== Optimizer configuration ====
MAX_GRAD_NORM=1.0
OPTIMIZER=$ADAMW
LR="1.0e-5"
WEIGHT_DECAY="0.01"
BETAS="[0.9,0.98]"
EPS="1e-8"

# ==== Data configuration ====
PRE_TOKENIZE=$FALSE
TRAIN_VAL_RATIO=0.8
MAX_INPUT_SEQ_LENGTH=$MAX_TOTAL_TOKENS
DATASET="/media/ExtremeSSD/llm_data/llm_sft_althubaity/Processed/sft/nemo-rl-test/7m_sampled/processed_1m_sample_oai.jsonl"
TMP_WORKDIR="/home/zeyad/nemo-rl/tmp_data"

# ==== Logging configuration ====
GPU_MONITOR_ENABLE=$TRUE
GPU_MONITOR_COLLECT_INTERVAL=10
GPU_MONITOR_FLUSH_INTERVAL=10

# ==== Cluster(s) configuration ====
GPUS_PER_NODE=4
NUM_NODES=1

# ==== Megatron configuration ====
MEGATRON_ENABLED=$TRUE
MEGATRON_EMPTY_UNUSED_MEMORY_LEVEL=1
MEGATRON_ACTIVATION_CHECKPOINTING=$TRUE
M_TENSOR_PARALLEL_SIZE=1
M_PIPELINE_PARALLEL_SIZE=1
M_CONTEXT_PARALLEL_SIZE=1
PIPELINE_DTYPE=$PRECISION
NUM_LAYERS_FIRST=$NULL
NUM_LAYERS_LAST=$NULL
SEQ_PARALLEL=$FALSE
ROPE_FUSION=$FALSE

# ==== Megatron Optimizer ====
MEGATRON_MIN_LR="1e-6"
MEGATRON_PARAMS_DTYPE=$PRECISION
ADAM_BETA1="0.9"
ADAM_BETA2="0.98"
ADAM_EPS="1e-5"
SGD_MOMENTUM="0.9"
DISTRIBUTED_OPTIMIZER=$FALSE
PRECISION_OPTIMIZER=$FALSE_STR
CLIP_GRADIENTS=$MAX_GRAD_NORM

# ==== Megatron Scheduler ====
LR_WARMUP_ITERS=300
LR_WARMUP_INIT=$MEGATRON_MIN_LR
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY_STYLE="constant"
LR_DECAY_ITERS=$NULL
START_WEIGHT_DECAY=$WEIGHT_DECAY
END_WEIGHT_DECAY=$WEIGHT_DECAY

# ==== Megatron DDP ====
GRAD_REDUCE_FP32=$FALSE
OVERLAP_GRAD_REDUCE=$FALSE
OVERLAP_PARAM_GATHER=$FALSE
AVERAGE_IN_COLLECTIVE=$TRUE
DP_SHARDING_STRATEGY="optim_grads_params"

# ==== Dataset Type ====
DATASET_TYPE=$OPENAI_DATASET

# =============================== Running script: DO NOT TOUCH ===============================

# =============================== Data Splitting ===============================
log_info "🚀 Script started"
if [[ $MEGATRON_ENABLED == $TRUE ]]; then
    log_info "🔧 Megatron mode enabled"
else
    log_info "🔧 DTensor mode enabled"
fi
log_info "🔄 Splitting dataset to local fast storage..."

# Use mktemp -d for RAY_TMP_DIR (Optim 1)
RAY_TMP_DIR=$(mktemp -d /tmp/ray_session_XXXXXX)
mkdir -p "$RAY_TMP_DIR"

# Increase file limits (Optim 11)
ulimit -n 65536 || log_warn "Could not increase file descriptor limit"

# Validate dataset
if [[ ! -f "$DATASET" ]]; then
    log_error "❌ Dataset not found: $DATASET"
    exit 1
fi

# Create temp file for capturing output
TMP_PATHS=$(mktemp)

# Run split and capture all output (logs + paths)
log_info "📦 Running split_dataset.py..."
python_output=$(python split_dataset.py "$DATASET" "$TRAIN_VAL_RATIO" "$SEED" "$TMP_WORKDIR" 2>&1 | tee /dev/tty)

# Save full output for inspection (optional)
echo "$python_output" > "$TMP_PATHS"

# === Extract SPLIT_DIR from stderr ===
SPLIT_DIR=$(echo "$python_output" | grep "📌 SPLIT_DIR=" | sed 's|📌 SPLIT_DIR=||' | head -n1 | xargs)
if [[ -n "$SPLIT_DIR" && -d "$SPLIT_DIR" ]]; then
    log_info "📁 Temporary split directory: $SPLIT_DIR"
else
    log_warn "⚠️ Could not detect valid SPLIT_DIR"
    SPLIT_DIR=""  # Ensure empty if invalid
fi

# === Extract ONLY the two .jsonl file paths from stdout ===
# We look for lines that are absolute paths and end with .jsonl
mapfile -t PATH_LINES < <(echo "$python_output" | grep -E "^/" | grep "\.jsonl$" | xargs -I{} echo {} | head -2)

if [[ ${#PATH_LINES[@]} -lt 2 ]]; then
    log_error "❌ Failed to extract train/val paths from split_dataset.py output"
    log_error "Expected 2 .jsonl paths, found ${#PATH_LINES[@]}"
    rm -f "$TMP_PATHS"
    exit 1
fi

TRAIN_PATH="${PATH_LINES[0]}"
VAL_PATH="${PATH_LINES[1]}"

# Validate that files exist
if [[ ! -f "$TRAIN_PATH" ]]; then
    log_error "❌ Train split file does not exist: $TRAIN_PATH"
    rm -f "$TMP_PATHS"
    exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
    log_error "❌ Validation split file does not exist: $VAL_PATH"
    rm -f "$TMP_PATHS"
    exit 1
fi

# Log final paths
log_info "✅ Dataset split successfully"
log_info "🧪 TRAIN_PATH (local): $TRAIN_PATH"
log_info "🧪 VAL_PATH (local):   $VAL_PATH"

# Clean up temp file
rm -f "$TMP_PATHS"

# =============================== Build Arguments ===============================
# Megatron prefix shortcut 
M="policy.megatron_cfg"

# ==== Ensure only one parallelism backend is active ====
if [[ "$MEGATRON_ENABLED" == "$TRUE" ]]; then
    PARALLELISM_OVERRIDE=" policy.dtensor_cfg.enabled=false"
else
    PARALLELISM_OVERRIDE=" policy.megatron_cfg.enabled=false"
fi

# ==== Optimizer args ====
OPTIM=""

if [[ "$MEGATRON_ENABLED" == "$TRUE" ]]; then
    # Use Megatron-native optimizer (not raw PyTorch)
    case "$OPTIMIZER" in
        "$ADAMW"|"adamw"|"torch.optim.AdamW")
            MEGATRON_OPTIM="adam"  
            ;;
        "$ADAM"|"adam")
            MEGATRON_OPTIM="adam"
            ;;
        "$SGD"|"sgd")
            MEGATRON_OPTIM="sgd"
            ;;
        *)
            log_error "Unsupported optimizer for Megatron: $OPTIMIZER"
            exit 1
            ;;
    esac

    OPTIM+=" policy.megatron_cfg.optimizer.optimizer=$MEGATRON_OPTIM"
    OPTIM+=" policy.megatron_cfg.optimizer.adam_beta1=$ADAM_BETA1"
    OPTIM+=" policy.megatron_cfg.optimizer.adam_beta2=$ADAM_BETA2"
    OPTIM+=" policy.megatron_cfg.optimizer.adam_eps=$ADAM_EPS"
    OPTIM+=" policy.megatron_cfg.optimizer.weight_decay=$WEIGHT_DECAY"
    OPTIM+=" policy.megatron_cfg.optimizer.lr=$LR"
else
    # DTensor uses standard policy.optimizer
    OPTIM+=" policy.optimizer.name=$OPTIMIZER"
    OPTIM+=" policy.optimizer.kwargs.betas=$BETAS"
    OPTIM+=" policy.optimizer.kwargs.eps=$EPS"
    OPTIM+=" policy.optimizer.kwargs.lr=$LR"
    OPTIM+=" policy.optimizer.kwargs.weight_decay=$WEIGHT_DECAY"
fi

# Dataset args function
build_dataset_args() {
    case "$DATASET_TYPE" in
        "$PROMPT_DATASET")
            echo " data.dataset_name=prompt_response_dataset"
            echo " +data.train_data_path=$TRAIN_PATH"
            echo " +data.val_data_path=$VAL_PATH"
            echo " +data.input_key=prompt"
            echo " +data.output_key=response"
            ;;
        "$OPENAI_DATASET")
            echo " data.dataset_name=openai_format"
            echo " +data.train_data_path=$TRAIN_PATH"
            echo " +data.val_data_path=$VAL_PATH"
            echo " +data.chat_key=messages"
            echo " +data.system_key=null"
            echo " +data.system_prompt=null"
            ;;
        "$OPEN_ASSISTANT_DATAST")
            echo " data.dataset_name=open_assistant"
            ;;
        "$OPEN_MATH_INSTRUCT_DATASET")
            echo " data.dataset_name=openmathinstruct2"
            echo " +data.split=train"
            echo " +data.output_key=output"
            echo " +data.prompt_file=/path/to/openmathinstruct2/prompts.json"
            ;;
        *)
            log_error "Unknown DATASET_TYPE: $DATASET_TYPE"
            exit 1
            ;;
    esac
}

ARGS="+data.tokenized=$PRE_TOKENIZE $(build_dataset_args)"

# =============================== NCCL Environment Setup ===============================
# Add this section BEFORE the Ray startup

# NCCL Communication Settings
export NCCL_DEBUG=INFO                    # Enable detailed NCCL logging
export NCCL_TIMEOUT_SEC=120              # 30 minutes timeout
export NCCL_IB_DISABLE=1                  # Disable InfiniBand (often problematic)
export NCCL_SOCKET_IFNAME=^docker,lo      # Exclude problematic interfaces
export NCCL_P2P_DISABLE=1                 # Disable P2P if causing issues
export NCCL_SHM_DISABLE=1                 # Disable shared memory if needed
export NCCL_NET_GDR_LEVEL=0               # Disable GPU Direct RDMA
export NCCL_NET_GDR_READ=0                # Disable GPU Direct RDMA reads

# PyTorch Distributed Settings
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000 # Enable NCCL trace buffer
export TORCH_NCCL_DUMP_ON_TIMEOUT=1       # Dump debug info on timeout
export TORCH_DISTRIBUTED_DEBUG=INFO       # PyTorch distributed debugging

# CUDA Settings
export CUDA_VISIBLE_DEVICES=0,1,2,3       # Explicit GPU assignment
export CUDA_DEVICE_ORDER=PCI_BUS_ID       # Consistent GPU ordering

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# =============================== Setup Cleanup Trap ===============================
# =============================== Setup Cleanup Trap ===============================
# Cleanup function
cleanup() {
    log_info "🧹 Cleaning up (signal received)..."
    
    # Stop Ray
    if command -v ray &> /dev/null; then
        uv run ray stop --force > /dev/null 2>&1 || true
    fi
    
    # Kill any Ray processes
    pkill -f "ray::" > /dev/null 2>&1 || true
    
    # Clean up temp directories
    [[ -n "$SPLIT_DIR" && -d "$SPLIT_DIR" ]] && rm -rf "$SPLIT_DIR" || true
    [[ -n "$RAY_TMP_DIR" && -d "$RAY_TMP_DIR" ]] && rm -rf "$RAY_TMP_DIR" || true
    
    log_info "✅ Cleanup completed"
}

# Set up signal traps for Ctrl+C (INT), kill (TERM), and script exit (EXIT)
trap cleanup EXIT INT TERM

log_info "🛡️ Signal handlers installed"

# ========== Ray Start ==========
log_info "🚀 Starting Ray cluster..."

# Clean up any zombie processes
uv run ray stop --force > /dev/null 2>&1 || true
pkill -f "ray::" > /dev/null 2>&1 || true
sleep 2

# Start Ray with error handling
log_info "▶️ Starting Ray head node..."
if uv run ray start --head --disable-usage-stats --port=6379 --temp-dir="$RAY_TMP_DIR"; then
    log_info "✅ Ray head started successfully"
else
    log_error "❌ Ray head failed to start"
    exit 1
fi

# Wait a moment for Ray to fully initialize
sleep 5

# Verify Ray is running
log_info "🔍 Verifying Ray status..."
if uv run ray status --address=127.0.0.1:6379; then
    log_info "✅ Ray cluster is active and ready"
    export RAY_ADDRESS="127.0.0.1:6379"
else
    log_error "❌ Ray cluster verification failed"
    log_info "🔧 Attempting to connect to default address..."
    
    # Try to find Ray's actual address
    if uv run ray status; then
        log_info "✅ Found Ray at default address"
        # Don't export RAY_ADDRESS, let Ray auto-detect
    else
        log_error "❌ Cannot connect to Ray cluster"
        log_error "🔧 Try running: uv run ray start --head --disable-usage-stats"
        exit 1
    fi
fi

log_info "🎯 Ray initialization complete"

# ========== Ray Start ==========

# Clean up any zombie processes
uv run ray stop --force > /dev/null 2>&1 || true
pkill -f "ray::" > /dev/null 2>&1 || true
sleep 2

log_info "🎯 Ray initialization complete"

# =============================== Run Training ===============================
log_info "▶️ Starting SFT training..."

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
    policy.activation_checkpointing_enabled=$CHECKPOINT \
    policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
    policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
    policy.max_total_sequence_length=${MAX_TOTAL_TOKENS} \
    policy.precision=$PRECISION \
    policy.dtensor_cfg.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    policy.dtensor_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE} \
    policy.max_grad_norm=${MAX_GRAD_NORM} \
    $OPTIM \
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
    policy.megatron_cfg.sequence_parallel=$SEQ_PARALLEL \
    policy.megatron_cfg.apply_rope_fusion=$ROPE_FUSION \
    policy.megatron_cfg.optimizer.min_lr=$MEGATRON_MIN_LR \
    policy.megatron_cfg.optimizer.params_dtype=$MEGATRON_PARAMS_DTYPE \
    policy.megatron_cfg.optimizer.sgd_momentum=$SGD_MOMENTUM \
    policy.megatron_cfg.optimizer.use_distributed_optimizer=$DISTRIBUTED_OPTIMIZER \
    policy.megatron_cfg.optimizer.use_precision_aware_optimizer="$PRECISION_OPTIMIZER" \
    policy.megatron_cfg.optimizer.clip_grad=${CLIP_GRADIENTS} \
    policy.megatron_cfg.scheduler.weight_decay_incr_style=$WEIGHT_DECAY_STYLE \
    policy.megatron_cfg.scheduler.lr_decay_style=$LR_DECAY_STYLE \
    policy.megatron_cfg.scheduler.lr_decay_iters="$LR_DECAY_ITERS" \
    policy.megatron_cfg.scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    policy.megatron_cfg.scheduler.lr_warmup_init=$LR_WARMUP_INIT \
    policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=$GRAD_REDUCE_FP32 \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=$OVERLAP_GRAD_REDUCE \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=$OVERLAP_PARAM_GATHER \
    policy.megatron_cfg.distributed_data_parallel_config.average_in_collective=$AVERAGE_IN_COLLECTIVE \
    policy.megatron_cfg.distributed_data_parallel_config.data_parallel_sharding_strategy=$DP_SHARDING_STRATEGY \
    $PARALLELISM_OVERRIDE 

# Check training result
TRAINING_EXIT_CODE=$?
if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    log_info "🎉 Training completed successfully"
else
    log_error "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
fi

# Exit (cleanup trap will run automatically)
exit $TRAINING_EXIT_CODE