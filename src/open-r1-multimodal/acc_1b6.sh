#!/bin/bash

cd `dirname $0`

export DEBUG_MODE="true"

RUN_NAME="MiniCPM-V-1B6-GRPO-1120px-256s"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export NCCL_P2P_LEVEL=NVL


WANDB_PROJECT=CPM-RFT CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" accelerate launch \
    --config_file debug.yml \
    src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  /share_data/data1/models/MiniCPM3-V-1_6B-hg \
    --dataset_name /share_data/data1/GUIData/bboxdata/tasks.jsonl \
    --image_root  /share_data/data1/GUIData/bboxdata/ \
    --max_prompt_length 2048 \
    --max_completion_length 64 \
    --max_line_res 1120 \
    --num_generations 256 \
    --num_iterations 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --learning_rate 1e-6 \
    --warmup_steps 10 \
    --weight_decay 0.1 \
    --adam_beta2 0.99 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --save_only_model true \
    --reward_funcs "args"
    # --attn_implementation flash_attention_2 \
