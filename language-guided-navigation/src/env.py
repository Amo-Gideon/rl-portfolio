"""
Language-guided navigation environment.

A tiny embodied-AI benchmark: an agent moves on a grid, follows a
natural-language instruction, and must emit a final answer.
Designed to run on CPU / small-GPU instances.
"""

import json
import random
import re
from typing import Dict, List, Tuple, Optional


class LangNavEnv:
    """
    Grid-world environment with landmarks and natural-language goals.

    Observation format (text):
        "Instruction: <instruction>\nYou are at (x,y) facing <dir>.\n"
        "Visible: <description>\nLast action result: ..."

    Action format (JSON):
        {"thought": "...", "action": "move_forward|turn_left|turn_right|look|answer"}
        {"thought": "...", "final_answer": "..."}
    """

    DIRS = ["N", "E", "S", "W"]
    DIR_VECTORS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
    ACTIONS = ["move_forward", "turn_left", "turn_right", "look"]

    def __init__(self, size: int = 5, max_steps: int = 15, seed: Optional[int] = None):
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.current_task: Optional[Dict] = None
        self.position: Tuple[int, int] = (0, 0)
        self.facing: str = "N"
        self.step_count: int = 0
        self.done: bool = False
        self.last_result: str = "None"

    def reset(self, task: Optional[Dict] = None) -> str:
        """Reset with a given task or a random generated one."""
        if task is None:
            task = self._sample_task()
        self.current_task = task
        self.position = tuple(task["start"])
        self.facing = task["start_dir"]
        self.step_count = 0
        self.done = False
        self.last_result = "Start."
        return self._build_obs()

    def step(self, action_text: str) -> Tuple[str, Dict, bool]:
        """Execute one JSON action."""
        self.step_count += 1
        info = {
            "raw": action_text,
            "parsed": None,
            "format_valid": False,
            "action_valid": False,
            "result": "",
        }

        parsed = self._parse_action(action_text)
        info["parsed"] = parsed

        if parsed is None:
            self.last_result = "Error: invalid JSON."
            self._maybe_done()
            return self._build_obs(), info, self.done

        info["format_valid"] = True

        if "final_answer" in parsed:
            self.done = True
            info["result"] = f"final_answer={parsed['final_answer']}"
            return self._build_obs(), info, self.done

        action = parsed.get("action", "")
        if action not in self.ACTIONS:
            self.last_result = f"Error: unknown action '{action}'."
            self._maybe_done()
            return self._build_obs(), info, self.done

        info["action_valid"] = True
        self._execute_action(action)
        info["result"] = self.last_result
        self._maybe_done()
        return self._build_obs(), info, self.done

    def _execute_action(self, action: str):
        if action == "move_forward":
            dx, dy = self.DIR_VECTORS[self.facing]
            x, y = self.position
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.position = (nx, ny)
                self.last_result = f"Moved to {self.position}."
            else:
                self.last_result = "Bump into wall."
        elif action == "turn_left":
            self.facing = self.DIRS[(self.DIRS.index(self.facing) - 1) % 4]
            self.last_result = f"Now facing {self.facing}."
        elif action == "turn_right":
            self.facing = self.DIRS[(self.DIRS.index(self.facing) + 1) % 4]
            self.last_result = f"Now facing {self.facing}."
        elif action == "look":
            self.last_result = self._describe_visible()

    def _maybe_done(self):
        if self.step_count >= self.max_steps:
            self.done = True

    def _build_obs(self) -> str:
        task = self.current_task
        return (
            f"Instruction: {task['instruction']}\n"
            f"You are at {self.position} facing {self.facing}.\n"
            f"Visible: {self._describe_visible()}\n"
            f"Last result: {self.last_result}\n"
            f"Choose one action: {self.ACTIONS} or answer with {{'final_answer': '...'}}"
        )

    def _describe_visible(self) -> str:
        """Describe landmarks in the same row/column or immediate neighbors."""
        if not self.current_task:
            return "nothing"
        visible = []
        for name, (x, y) in self.current_task["objects"].items():
            if (x, y) == self.position:
                visible.append(f"{name} here")
            elif self._in_front((x, y)):
                dist = self._distance((x, y))
                visible.append(f"{name} {dist} step(s) ahead")
            elif x == self.position[0] or y == self.position[1]:
                dir_name = self._relative_dir((x, y))
                visible.append(f"{name} to the {dir_name}")
        return ", ".join(visible) if visible else "empty surroundings"

    def _in_front(self, pos: Tuple[int, int]) -> bool:
        dx, dy = self.DIR_VECTORS[self.facing]
        x, y = self.position
        fx, fy = x + dx, y + dy
        return pos == (fx, fy)

    def _distance(self, pos: Tuple[int, int]) -> int:
        return abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])

    def _relative_dir(self, pos: Tuple[int, int]) -> str:
        dx = pos[0] - self.position[0]
        dy = pos[1] - self.position[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "front" if dy > 0 else "back"

    def _sample_task(self) -> Dict:
        """Generate a random navigation + QA task."""
        objects = self.rng.sample(
            [("red house", "red"), ("blue car", "blue"), ("green tree", "green"),
             ("yellow bench", "yellow"), ("mailbox", "metal")],
            k=3,
        )
        positions = []
        while len(positions) < len(objects):
            p = (self.rng.randint(0, self.size - 1), self.rng.randint(0, self.size - 1))
            if p not in positions and p != (0, 0):
                positions.append(p)
        obj_map = {name: pos for (name, _), pos in zip(objects, positions)}
        target_name, target_attr = self.rng.choice(objects)
        instruction = (
            f"Go to the {target_name} and report its color."
        )
        return {
            "instruction": instruction,
            "answer": target_attr,
            "start": [0, 0],
            "start_dir": "N",
            "objects": obj_map,
            "target": target_name,
        }

    def get_answer(self) -> str:
        return str(self.current_task.get("answer", "")).strip().lower()

    def _parse_action(self, text: str) -> Optional[Dict]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[-1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
        return None


def reward_from_trajectory(trajectory: Dict, env: LangNavEnv) -> Dict[str, float]:
    """
    Compute a simple decomposed reward.
    trajectory: dict with keys actions, infos, final_answer.
    """
    final_answer = trajectory.get("final_answer", "").strip().lower()
    gt = env.get_answer()
    answer_r = 1.0 if final_answer == gt else 0.0

    format_r = sum(1 for a in trajectory.get("actions", []) if a is not None) / max(1, len(trajectory.get("actions", [])))
    action_valid_r = sum(
        1 for i in trajectory.get("infos", []) if i.get("format_valid") and i.get("action_valid")
    ) / max(1, len(trajectory.get("infos", [])))

    total = 0.5 * answer_r + 0.2 * format_r + 0.3 * action_valid_r
    return {
        "answer_accuracy_reward": answer_r,
        "format_reward": format_r,
        "action_validity_reward": action_valid_r,
        "total_reward": total,
    }
