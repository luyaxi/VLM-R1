#!/bin/bash

source ~/miniconda3/bin/activate vcpm
cd `dirname $0`

RUN_NAME="MiniCPM-26o-GRPO-GUI-thought-cIoU-diverse"

set -ex
CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" WANDB_PROJECT=CPM-RFT accelerate launch \
    --config_file 4nodes_zero3.yaml \
    src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  /data3/workhome/luyaxi/VCPM-R1/models/MiniCPM-o-2_6-hg \
    --dataset_name /data3/workhome/luyaxi/VCPM-R1/GUIData/bboxdata/tasks.jsonl \
    --image_root  /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data \
    --max_prompt_length 2048 \
    --max_completion_length 128 \
    --num_generations 8 \
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
    --reward_funcs "type" "args"
