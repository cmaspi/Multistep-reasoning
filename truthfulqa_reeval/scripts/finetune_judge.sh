MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

LR=1e-5
for JUDGE_TYPE in "truth"; do
    MODEL_NAME=llama2_${MODEL_SIZE}_${JUDGE_TYPE}_judge_final
    OUTPUT_DIR=output/${MODEL_NAME}/

    echo "Training LLaMa ${MODEL_SIZE} for ${JUDGE_TYPE} prediction."
    echo "Using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

    export WANDB_NAME=${MODEL_NAME}
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        src/finetune_llama.py \
        --model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
        --use_slow_tokenizer \
        --train_file data/ARC+world_tree.jsonl \
        --max_seq_length 256 \
        --preprocessing_num_workers 64 \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate 1e-5 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0. \
        --num_train_epochs 10 \
        --output_dir ${OUTPUT_DIR} \
        --with_tracking \
        --report_to wandb \
        --logging_steps 1 \
        --use_lora
done