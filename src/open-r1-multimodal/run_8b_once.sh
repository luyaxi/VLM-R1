#!/bin/bash
set -e

RUN_NAME="MiniCPM-26o-GRPO-GUI-FGEVAL-Staged"

node_ips=("10.0.1.11" "10.0.1.12" "10.0.1.13" "10.0.1.15")
# node_ips=("100.96.1.27" "100.96.1.23" "100.96.1.28" "100.96.1.5")
# node_ips=("10.0.1.13" "10.0.1.15")

LOCAL_CODEBASE_PATH="/data3/workhome/luyaxi/VCPM-R1/src/open-r1-multimodal"

master_ip=${node_ips[0]}
master_port=12329

echo "Code Base: $LOCAL_CODEBASE_PATH"

# Get master node IP address
echo "MASTER_NODE: $master_ip"


# Create log directory for each node
LOG_DIR="/data3/workhome/luyaxi/VCPM-R1/src/open-r1-multimodal/logs"
mkdir -p $LOG_DIR


# Launch training on each node
NODE_RANK=0
# NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \

for host in "${node_ips[@]}"; do
    echo "Launching training on $host..."
    LOG_FILE="$LOG_DIR/${host}_rank${NODE_RANK}.log"
    echo "Launching on node $host with rank $NODE_RANK, logging to $LOG_FILE"
    ssh $host "cd $LOCAL_CODEBASE_PATH && \
    source ~/miniconda3/bin/activate vcpm && pwd && \
    NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1  NCCL_DEBUG=INFO WANDB_PROJECT=CPM-RFT torchrun --nproc_per_node=8 \
    --nnodes="${#node_ips[@]}" \
    --node_rank=$NODE_RANK \
    --master_addr=$master_ip \
    --master_port=$master_port \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/ds_zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  /data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-o-2_6-hg \
    --dataset_name /data3/workhome/luyaxi/VCPM-R1/GUIData/mb_data/tasks.jsonl \
    --image_root  /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data \
    --max_prompt_length 2048 \
    --max_completion_length 256 \
    --num_generations 8 \
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
    " > "$LOG_FILE" 2>&1 &
    
    NODE_RANK=$((NODE_RANK+1))

done

echo "Jobs launched. To monitor the logs, you can:"
echo "1. Use 'tail -f $LOG_DIR/*.log' to watch all logs"
echo "2. Use 'tail -f $LOG_DIR/<node_name>_rank<N>.log' to watch a specific node"

wait