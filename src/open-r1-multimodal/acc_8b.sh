#!/bin/bash

source ~/miniconda3/bin/activate vcpm
cd `dirname $0`

# RUN_NAME="MiniCPM-26o-SFT-2epochs-GRPO-1120px-8s-lr"

# RUN_NAME="MiniCPMV-HW-THOUGHT-7B-GRPO-1120px-8s-lr"
RUN_NAME="MiniCPMV-HW-E-THOUGHT-7B-GRPO-1120px-8s-lr"


set -ex
TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1 UCX_NET_DEVICES=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6" WANDB_PROJECT=CPM-RFT accelerate launch \
    --config_file debug.yml \
    src/open_r1/grpo_rec.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /share_data/data1/models/MiniCPM-V-HW-THOUGHT-EP1/checkpoint-1100 \
    --dataset_name /share_data/data1/GUIData/unique_aitw_mb_ac.jsonl \
    --max_prompt_length 2048 \
    --max_completion_length 160 \
    --max_line_res 1120 \
    --num_generations 8 \
    --num_iterations 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataloader_prefetch_factor 12 --dataloader_num_workers 32 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --learning_rate 3e-6 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --adam_beta2 0.98 \
    --global_var false \
    --lr_scheduler_type "cosine" \
    --tune_vision true \
    --gather_deepspeed3_params true \
    --bf16 \
    --beta 0.04 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --reward_funcs "args" "schema"
    # --attn_implementation flash_attention_2 \
