import json
from datasets import Dataset
from huggingface_hub import HfApi

# Load the JSON data
with open("data/trajectory-prediction-v1/train.json", "r") as f:
    data = json.load(f)

# Convert to list of examples
examples = []
for item in data:
    example = {
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "distribution": item["distribution"],
    }
    examples.append(example)

# Create Hugging Face dataset
dataset = Dataset.from_list(examples)

# Add dataset metadata
dataset.info.description = "A dataset for trajectory prediction tasks"
dataset.info.license = "MIT"
dataset.info.features = {
    "question": "string",
    "ground_truth": "string",
    "distribution": "string",
}

# Push to Hugging Face
api = HfApi()
dataset.push_to_hub("trajectory-prediction-v1", private=True)
