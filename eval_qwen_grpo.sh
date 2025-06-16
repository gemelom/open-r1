lighteval accelerate \
    "model_name=Qwen/Qwen2.5-1.5B-Instruct" \
    "community|trajectory_prediction_evals|0|0" \
    --custom-tasks trajectory_prediction_evals.py