"""
Verifiable Reward Decomposition for Agentic RL.

Implements three components:
  1. format_reward: JSON structure compliance
  2. tool_validity_reward: Tool existence and argument validity
  3. answer_accuracy_reward: Exact or numeric match to ground truth

References:
  - Agent-RLVR: Verifiable rewards for code agents
  - Dr. GRPO: Token-level penalization to prevent reward hacking
"""

import json
import re
from typing import Dict, Any, Optional


class VerifiableReward:
    """
    Decomposed reward function with verifiable components.
    No learned reward model needed — all signals are programmatic.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.weights = self.config.get("components", {
            "format_reward": 0.2,
            "tool_validity_reward": 0.3,
            "answer_accuracy_reward": 0.5,
        })
        self.numeric_tolerance = self.config.get("numeric_tolerance", 0.05)
        self.available_tools = {"search", "calculate", "run_code"}

    def compute(self, trajectory: Dict[str, Any], ground_truth: str) -> Dict[str, float]:
        """
        Compute decomposed rewards for a full trajectory.

        Args:
            trajectory: dict with keys 'actions' (list of parsed JSONs), 
                       'infos' (list of env info dicts), 'final_answer' (str)
            ground_truth: correct answer string

        Returns:
            dict with individual components and total reward
        """
        actions = trajectory.get("actions", [])
        infos = trajectory.get("infos", [])
        final_answer = trajectory.get("final_answer", "").strip()

        format_r = self._format_reward(actions)
        tool_r = self._tool_validity_reward(infos)
        answer_r = self._answer_accuracy_reward(final_answer, ground_truth)

        total = (
            self.weights.get("format_reward", 0.2) * format_r +
            self.weights.get("tool_validity_reward", 0.3) * tool_r +
            self.weights.get("answer_accuracy_reward", 0.5) * answer_r
        )

        return {
            "format_reward": format_r,
            "tool_validity_reward": tool_r,
            "answer_accuracy_reward": answer_r,
            "total_reward": total,
        }

    def _format_reward(self, actions: list) -> float:
        """
        Reward valid JSON structure with required keys.

        For tool calls: requires 'thought', 'tool_name', 'tool_input'
        For final answers: requires 'thought', 'final_answer'
        """
        if not actions:
            return 0.0

        valid_count = 0
        for action in actions:
            if action is None:
                continue
            if "final_answer" in action:
                if "thought" in action and "final_answer" in action:
                    valid_count += 1
            elif "tool_name" in action and "tool_input" in action:
                if "thought" in action:
                    valid_count += 1

        return valid_count / len(actions) if actions else 0.0

    def _tool_validity_reward(self, infos: list) -> float:
        """
        Reward correct tool selection and execution.
        """
        if not infos:
            return 0.0

        valid_count = 0
        for info in infos:
            if info.get("format_valid") and info.get("tool_valid"):
                valid_count += 1

        return valid_count / len(infos) if infos else 0.0

    def _answer_accuracy_reward(self, predicted: str, ground_truth: str) -> float:
        """
        Reward answer correctness.
        Supports exact string match or numeric tolerance.
        """
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()

        if pred == gt:
            return 1.0

        pred_num = self._extract_number(pred)
        gt_num = self._extract_number(gt)

        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                return 1.0 if abs(pred_num) < 1e-6 else 0.0
            relative_error = abs(pred_num - gt_num) / abs(gt_num)
            if relative_error <= self.numeric_tolerance:
                return 1.0 - (relative_error / self.numeric_tolerance) * 0.5
            else:
                return 0.0

        if pred in gt or gt in pred:
            return 0.3

        return 0.0

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract first numeric value from text."""
        text = text.replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None