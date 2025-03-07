#!/bin/bash
set -ex
# MASTER_NODE=10.0.1.11 MASTER_PORT=12345 NODE_RANK=0 NNODES=4 bash 8b.sh
source ~/miniconda3/bin/activate vcpm
RUN_NAME="MiniCPM-26o-GRPO-GUI-FGEVAL-Staged"

LOCAL_CODEBASE_PATH="/data3/workhome/luyaxi/VCPM-R1/src/open-r1-multimodal"
echo "Code Base: $LOCAL_CODEBASE_PATH"

# Get master node IP address
echo "MASTER_NODE: $MASTER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES: $NNODES"

# Create log directory for each node
LOG_DIR="/data3/workhome/luyaxi/VCPM-R1/src/open-r1-multimodal/logs"
mkdir -p $LOG_DIR

host=$(hostname -i)

echo "Launching training on $host..."
LOG_FILE="$LOG_DIR/${host}_rank${NODE_RANK}.log"
echo "Launching on node $host with rank $NODE_RANK, logging to $LOG_FILE"
cd $LOCAL_CODEBASE_PATH


# NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1

CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" WANDB_PROJECT=CPM-RFT torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_NODE \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/ds_zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  /data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-o-2_6-hg \
    --dataset_name /data3/workhome/luyaxi/VCPM-R1/GUIData/mb_data/tasks.jsonl \
    --image_root  /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data \
    --max_prompt_length 2048 \
    --max_completion_length 256 \
    --num_generations 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --reward_funcs "schema" "type" "args" "point"
