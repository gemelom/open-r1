# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import re
from typing import Optional

import numpy as np

from lighteval.metrics.metrics import SampleLevelMetric, SampleLevelMetricGrouping
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


PROMPT_2D_TEMPLATE = """You are an expert in mathematical sequences and patterns. Your task is to analyze a given sequence of two-dimensional coordinates, identify the underlying uniform linear motion, and predict the subsequent coordinates.

Here is the input sequence of coordinates:
{{{COORDINATE_SEQUENCE}}}

Think step-by-step to determine the rule governing this sequence.
1. Calculate the difference between consecutive coordinates for both the x and y components to find the constant velocity vector (Δx, Δy).
2. Verify if this velocity vector (Δx, Δy) is consistent across all given pairs.
3. Once the velocity vector is confirmed, apply this velocity vector to generate the next 10 coordinates. Each new coordinate (xn+1, yn+1) will be calculated as (xn + Δx, yn + Δy). Ensure that none of the given coordinates are repeated in the output.

Your final answer should only be the predicted 10 coordinates, starting from the last coordinate in the input seqeunce. The answer should be formatted as a comma-separated list of (x, y) pairs enclosed in curly braces, like: {{(x1, y1), (x2, y2), ..., (x10, y10)}}.
"""


def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query=PROMPT_2D_TEMPLATE.format(COORDINATE_SEQUENCE=line["question"]),
        choices=[line["ground_truth"]],
        gold_index=0,
        instruction="",
    )


def accuracy_reward_metric(
    predictions: list[str], formatted_doc: Doc, **kwargs
) -> dict:
    def extract_answer(content: str) -> Optional[str]:
        # Extract content between <answer></answer> tags
        pattern = r"<answer>\n(.*?)\n</answer>"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def parse_coordinate(coord_str: str) -> Optional[tuple[float, float]]:
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

    def validate_model_output(text: str) -> bool:
        text = text.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return False
        coords = extract_coordinates(text)
        if len(coords) != 10:
            return False
        for coord in coords:
            if parse_coordinate(coord) is None:
                return False
        return True

    def validate_ground_truth(text: str) -> bool:
        coords = extract_coordinates(text)
        if len(coords) != 10:
            return False
        for coord in coords:
            if parse_coordinate(coord) is None:
                return False
        return True

    def is_close(
        a: float, b: float, rel_tol: float = 1e-5, abs_tol: float = 1e-2
    ) -> bool:
        """Check if two floats are close to each other within the given tolerance."""
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def compare_coordinates(
        coord1: tuple[float, float], coord2: tuple[float, float]
    ) -> bool:
        """Compare two coordinates with tolerance."""
        return is_close(coord1[0], coord2[0]) and is_close(coord1[1], coord2[1])

    response = predictions[0]
    answer = extract_answer(response)

    print("response", response)
    print("answer", answer)

    if answer is None:
        return {"accuracy": False, "score": 0}

    # Validate format of answer and solution
    if not validate_model_output(answer) or not validate_ground_truth(
        formatted_doc.choices[0]
    ):
        return {"accuracy": False, "score": 0}

    # Compare the coordinates
    try:
        # Parse coordinates into list of (x,y) tuples
        answer_coords = [
            parse_coordinate(coord) for coord in extract_coordinates(answer)
        ]
        sol_coords = [
            parse_coordinate(coord)
            for coord in extract_coordinates(formatted_doc.choices[0])
        ]

        # Count correct coordinates
        correct_count = sum(
            1 for a, s in zip(answer_coords, sol_coords) if compare_coordinates(a, s)
        )

        # Calculate reward based on the number of correct coordinates
        # Each correct coordinate contributes 0.1 to the reward
        reward = correct_count * 0.1
    except Exception as e:
        reward = 0

    return {"accuracy": reward == 1, "score": reward}


trajectory_prediction_evals = SampleLevelMetricGrouping(
    metric_name="trajectory_prediction_evals",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=accuracy_reward_metric,
    corpus_level_fn={"accuracy": np.mean, "score": np.mean},
)

task = LightevalTaskConfig(
    name="trajectory_prediction_evals",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="gemelom/trajectory-prediction-v1",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[trajectory_prediction_evals],
)

TASKS_TABLE = [task]
