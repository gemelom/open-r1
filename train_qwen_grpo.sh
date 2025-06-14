ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/custom_zero3.yaml \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/custom_config.yaml \
    --vllm_mode colocate