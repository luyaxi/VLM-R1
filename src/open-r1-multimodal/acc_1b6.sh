#!/bin/bash

cd `dirname $0`

export DEBUG_MODE="true"

RUN_NAME="MiniCPM-V-1B6-GRPO12-GUI"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export NCCL_P2P_LEVEL=NVL


WANDB_PROJECT=CPM-RFT CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" accelerate launch \
    --config_file acc_zero3.yaml \
    src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  /data3/workhome/luyaxi/VCPM-R1/models/MiniCPM3-V-1_6B-hg \
    --dataset_name /data3/workhome/luyaxi/VCPM-R1/GUIData/mb_data/tasks.jsonl \
    --image_root  /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data \
    --max_prompt_length 2048 \
    --max_completion_length 128 \
    --num_generations 12 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --reward_funcs "schema" "type" "args" "point"
    # --attn_implementation flash_attention_2 \
