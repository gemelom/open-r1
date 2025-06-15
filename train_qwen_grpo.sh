WANDB_MODE=offline \
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/custom_zero3.yaml \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/custom_config.yaml \
    --vllm_mode colocate