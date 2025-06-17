HF_HUB_ENABLE_HF_TRANSFER=0 \
lighteval accelerate \
    "vllm_eval_config.yaml" \
    "community|trajectory_prediction_evals|0|0" \
    --custom-tasks trajectory_prediction_evals.py \