import re
from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model and tokenizer
model_name = "gemelom/Qwen2.5-1.5B-Open-R1-GRPO"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
)

PROMPT_TEMPLATE = """You are an expert in mathematical sequences and patterns. Your task is to analyze a given sequence of two-dimensional coordinates, identify the underlying uniform linear motion, and predict the subsequent coordinates.

Here is the input sequence of coordinates:
{COORDINATE_SEQUENCE}

Think step-by-step to determine the rule governing this sequence.
1. Calculate the difference between consecutive coordinates for both the x and y components to find the constant velocity vector (Δx, Δy).
2. Verify if this velocity vector (Δx, Δy) is consistent across all given pairs.
3. Once the velocity vector is confirmed, apply this velocity vector to generate the next 10 coordinates. Each new coordinate (xn+1, yn+1) will be calculated as (xn + Δx, yn + Δy). Ensure that none of the given coordinates are repeated in the output.

Your final answer should only be the predicted 10 coordinates, starting from the last coordinate in the input seqeunce. The answer should be formatted as a comma-separated list of (x, y) pairs enclosed in curly braces, like: {{(x1, y1), (x2, y2), ..., (x10, y10)}}.
"""


def plot_trajectory(input_coords, predicted_coords):
    """Create a plot showing input and predicted coordinates."""
    plt.figure(figsize=(10, 6))

    # Plot input points
    input_x = [x for x, y in input_coords]
    input_y = [y for x, y in input_coords]
    plt.scatter(input_x, input_y, c="blue", label="Input Points", s=100)

    # Plot predicted points
    pred_x = [x for x, y in predicted_coords]
    pred_y = [y for x, y in predicted_coords]
    plt.scatter(pred_x, pred_y, c="red", label="Predicted Points", s=100)

    # Connect points with lines
    plt.plot(input_x + pred_x, input_y + pred_y, "k--", alpha=0.3)

    # Add labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot to bytes and convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def extract_answer(content: str):
    # Extract content between <answer></answer> tags
    pattern = r"<answer>\n(.*?)\n</answer>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def parse_coordinate(coord_str: str):
    # Remove whitespace and check if it's in (x,y) format
    coord_str = coord_str.strip()
    if not (coord_str.startswith("(") and coord_str.endswith(")")):
        return None

    # Remove parentheses and split by comma
    try:
        x, y = coord_str[1:-1].split(",")
        return (float(x.strip()), float(y.strip()))
    except (ValueError, IndexError):
        return None


def extract_coordinates(text: str) -> list:
    # Use regex to extract all (x,y) pairs
    return re.findall(r"\([^()]+,[^()]+\)", text)


def generate_prediction(coordinate_sequence):
    # Format the prompt
    system_prompt = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"""
    user_prompt = PROMPT_TEMPLATE.format(COORDINATE_SEQUENCE=coordinate_sequence)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Generate response
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[full_output.find("assistant") :]
    print("response:", response)

    # Extract the answer part
    answer = extract_answer(response)

    # Parse coordinates
    try:
        coords = extract_coordinates(answer)
        input_coords = extract_coordinates(coordinate_sequence)
        input_coords = [parse_coordinate(coord) for coord in input_coords]
        predicted_coords = [parse_coordinate(coord) for coord in coords]

        if not coords:
            raise ValueError("No coordinates could be parsed from the answer")

        # Create visualization
        plot_buf = plot_trajectory(input_coords, predicted_coords)

        return response, answer, plot_buf
    except Exception as e:
        print("Error during coordinate parsing:", str(e))
        return response, answer, None


# Create Gradio interface
demo = gr.Interface(
    fn=generate_prediction,
    inputs=gr.Textbox(
        label="Input Coordinate Sequence",
        placeholder="Enter a sequence of coordinates in the format: {(x1,y1), (x2,y2), ...}",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="Model Response", lines=10),
        gr.Textbox(label="Predicted Coordinates", lines=10),
        gr.Image(label="Trajectory Visualization", type="pil"),
    ],
    title="Trajectory Prediction Demo",
    description="Enter a sequence of 2D coordinates and the model will predict the next 10 coordinates based on the pattern.",
    examples=[
        ["{(1,2), (3,4), (5,6), (7,8), (9,10)}"],
        ["{(0,0), (2,2), (4,4), (6,6), (8,8)}"],
        ["{(1,1), (2,3), (3,5), (4,7), (5,9)}"],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
