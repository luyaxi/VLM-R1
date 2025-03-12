#!/bin/bash

source ~/miniconda3/bin/activate vcpm
cd `dirname $0`

# RUN_NAME="MiniCPM-26o-GRPO-1120px-abs-IoU-KL"

RUN_NAME="MiniCPMV-HW-7B-GRPO-1120px"

set -ex
TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" WANDB_PROJECT=CPM-RFT accelerate launch \
    --config_file 4nodes_zero3.yaml \
    src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /share_data/data1/models/MiniCPM-V-HW-7B-hg \
    --dataset_name /share_data/data1/GUIData/bboxdata/tasks.jsonl \
    --image_root  /share_data/data1/GUIData/bboxdata/ \
    --max_prompt_length 2048 \
    --max_completion_length 128 \
    --max_line_res 1120 \
    --num_generations 128 \
    --num_iterations 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 10.0 \
    --logging_steps 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.99 \
    --lr_scheduler_type "constant" \
    --tune_vision true \
    --gather_deepspeed3_params true \
    --bf16 \
    --beta 0.1 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --reward_funcs "args"
