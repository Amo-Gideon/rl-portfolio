"""
Verifiable math task generator for GRPO training.

Tasks are programmatically generated arithmetic word problems with exact answers,
so rewards need no learned model — just string/numeric matching.
"""

import random
import re
from typing import Dict, List


PROMPT_TEMPLATE = (
    "Solve the following problem. Reason step by step, then write your final "
    "answer on the last line in the form 'ANSWER: <number>'.\n\nProblem: {problem}"
)


def generate_tasks(num_tasks: int = 200, seed: int = 42, max_number: int = 99) -> List[Dict]:
    """Generate a mix of single- and two-step arithmetic problems."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(num_tasks):
        kind = rng.choice(["add", "sub", "mul", "add_mul", "mul_add"])
        a = rng.randint(2, max_number)
        b = rng.randint(2, max_number)
        c = rng.randint(2, 30)

        if kind == "add":
            problem = f"What is {a} + {b}?"
            answer = a + b
        elif kind == "sub":
            hi, lo = max(a, b), min(a, b)
            problem = f"What is {hi} - {lo}?"
            answer = hi - lo
        elif kind == "mul":
            problem = f"What is {a} * {b}?"
            answer = a * b
        elif kind == "add_mul":
            problem = f"What is {a} + {b} * {c}?"
            answer = a + b * c
        else:  # mul_add
            problem = f"What is {a} * {b} + {c}?"
            answer = a * b + c

        tasks.append({
            "prompt": PROMPT_TEMPLATE.format(problem=problem),
            "problem": problem,
            "answer": str(answer),
        })
    return tasks


def extract_answer(text: str) -> str:
    """Extract the final answer from model output, looking for 'ANSWER: X'."""
    matches = re.findall(r"ANSWER:\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


def reward(response: str, ground_truth: str) -> Dict[str, float]:
    """Binary exact-match reward plus small format bonus for using 'ANSWER:'."""
    pred = extract_answer(response)
    gt = ground_truth.strip()
    try:
        correct = abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        correct = pred == gt

    format_ok = "ANSWER:" in response
    answer_r = 1.0 if correct else 0.0
    format_r = 1.0 if format_ok else 0.0
    return {
        "answer_reward": answer_r,
        "format_reward": format_r,
        "total_reward": 0.8 * answer_r + 0.2 * format_r,
    }
