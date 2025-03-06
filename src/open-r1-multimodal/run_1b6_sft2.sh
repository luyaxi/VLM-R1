#!/bin/bash

cd `dirname $0`

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="MiniCPM-V-1B6-SFT3000-GRPO-GUI"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export NCCL_P2P_LEVEL=NVL


torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /data3/workhome/luyaxi/VCPM-R1/models/MiniCPM3-V-1_6B-SFT3000 \
    --dataset_name /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data/tasks.jsonl \
    --image_root  /data3/workhome/luyaxi/VCPM-R1/GUIData/new_mb_data \
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
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
    --reward_funcs "gui"
    # --attn_implementation flash_attention_2 \
