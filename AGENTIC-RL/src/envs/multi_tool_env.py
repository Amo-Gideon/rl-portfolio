"""
Multi-Tool Agentic Environment for RL Training.

The agent interacts with a task by generating structured JSON tool calls.
Tools: search, calculate, run_code.
Tasks are programmatically verifiable.
"""

import json
import random
import re
from typing import Dict, List, Tuple, Any, Optional


class ToolExecutor:
    """Simulated tool execution for training."""

    def __init__(self):
        self.knowledge_base = {
            "population of france": "68 million",
            "capital of japan": "Tokyo",
            "gdp of usa 2023": "27.36 trillion USD",
            "speed of light": "299792458 m/s",
            "pi": "3.14159265359",
            "atomic number of oxygen": "8",
            "height of eiffel tower": "330 meters",
            "area of china": "9.6 million square kilometers",
        }

    def search(self, query: str) -> str:
        """Simulated search: fuzzy match against knowledge base."""
        query_lower = query.lower().strip()
        best_match = None
        best_score = 0
        for key, value in self.knowledge_base.items():
            score = self._overlap_score(query_lower, key)
            if score > best_score:
                best_score = score
                best_match = value
        if best_match and best_score > 0.3:
            return f"Search result: {best_match}"
        return "Search result: No relevant information found."

    def calculate(self, expression: str) -> str:
        """Safe math evaluation."""
        try:
            # Whitelist allowed characters
            if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
                return "Error: Invalid characters in expression"
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_code(self, code: str) -> str:
        """Simulated code execution (restricted)."""
        try:
            # Only allow basic math operations
            if any(bad in code for bad in ["import", "open", "exec", "eval", "__"]):
                return "Error: Forbidden operation in code"
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            output = local_vars.get("result", "No output")
            return f"Output: {output}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _overlap_score(self, q: str, key: str) -> float:
        q_words = set(q.split())
        k_words = set(key.split())
        if not q_words:
            return 0.0
        return len(q_words & k_words) / len(q_words)


class MultiToolEnv:
    """
    Agentic environment where an LLM solves tasks via tool use.

    Action space: structured JSON with either:
      - tool call: {"thought": str, "tool_name": str, "tool_input": str}
      - final answer: {"thought": str, "final_answer": str}

    Observation: execution result or task description.
    """

    def __init__(self, tasks: Optional[List[Dict]] = None, max_steps: int = 5):
        self.executor = ToolExecutor()
        self.max_steps = max_steps
        self.tasks = tasks or self._default_tasks()
        self.current_task = None
        self.history = []
        self.step_count = 0
        self.done = False

    def _default_tasks(self) -> List[Dict]:
        """Built-in verifiable tasks."""
        return [
            {
                "question": "What is the population of France divided by 2?",
                "answer": "34000000",
                "required_tools": ["search", "calculate"],
                "hint": "Search for population of France, then divide by 2"
            },
            {
                "question": "Calculate the area of a circle with radius 5.",
                "answer": "78.54",
                "required_tools": ["run_code"],
                "hint": "Use pi * r^2"
            },
            {
                "question": "What is 15 multiplied by 23 plus 7?",
                "answer": "352",
                "required_tools": ["calculate"],
                "hint": "Use calculate tool"
            },
            {
                "question": "What is the capital of Japan?",
                "answer": "Tokyo",
                "required_tools": ["search"],
                "hint": "Search for capital of Japan"
            },
            {
                "question": "If the speed of light is 299792458 m/s, what is it divided by 2?",
                "answer": "149896229",
                "required_tools": ["search", "calculate"],
                "hint": "Search for speed of light, then divide by 2"
            },
            {
                "question": "What is the atomic number of oxygen plus 10?",
                "answer": "18",
                "required_tools": ["search", "calculate"],
                "hint": "Search atomic number of oxygen, then add 10"
            },
            {
                "question": "Calculate 100 factorial divided by 99 factorial.",
                "answer": "100",
                "required_tools": ["calculate"],
                "hint": "Simplify: 100! / 99! = 100"
            },
            {
                "question": "What is the height of the Eiffel Tower in meters multiplied by 2?",
                "answer": "660",
                "required_tools": ["search", "calculate"],
                "hint": "Search height, then multiply by 2"
            },
        ]

    def reset(self, task_idx: Optional[int] = None) -> str:
        """Reset environment with a new task."""
        if task_idx is not None:
            self.current_task = self.tasks[task_idx]
        else:
            self.current_task = random.choice(self.tasks)
        self.history = []
        self.step_count = 0
        self.done = False
        return self._build_observation()

    def _build_observation(self) -> str:
        """Build the initial prompt/observation."""
        task = self.current_task
        obs = f"""Task: {task['question']}

You have access to these tools:
- search(query): Search for factual information
- calculate(expression): Evaluate a mathematical expression
- run_code(code): Execute Python code and return the result

Respond in JSON format.
For tool calls: {{"thought": "...", "tool_name": "...", "tool_input": "..."}}
For final answer: {{"thought": "...", "final_answer": "..."}}
"""
        return obs

    def step(self, action_text: str) -> Tuple[str, Dict[str, Any], bool]:
        """
        Execute one step.

        Returns:
            observation: result of tool execution or feedback
            info: dict with parsed action, tool result, etc.
            done: whether episode is finished
        """
        self.step_count += 1
        info = {
            "raw_action": action_text,
            "parsed_action": None,
            "tool_result": None,
            "format_valid": False,
            "tool_valid": False,
        }

        # Parse JSON action
        parsed = self._parse_action(action_text)
        info["parsed_action"] = parsed

        if parsed is None:
            obs = "Error: Invalid JSON format. Please respond with valid JSON."
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, info, self.done

        info["format_valid"] = True

        # Check if final answer
        if "final_answer" in parsed:
            self.done = True
            info["final_answer"] = str(parsed["final_answer"]).strip()
            obs = f"Final answer received: {info['final_answer']}"
            return obs, info, self.done

        # Execute tool
        tool_name = parsed.get("tool_name", "")
        tool_input = parsed.get("tool_input", "")

        if tool_name not in ["search", "calculate", "run_code"]:
            obs = f"Error: Unknown tool '{tool_name}'. Available: search, calculate, run_code"
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, info, self.done

        info["tool_valid"] = True

        if tool_name == "search":
            result = self.executor.search(tool_input)
        elif tool_name == "calculate":
            result = self.executor.calculate(tool_input)
        elif tool_name == "run_code":
            result = self.executor.run_code(tool_input)

        info["tool_result"] = result
        obs = f"Tool '{tool_name}' result: {result}"

        if self.step_count >= self.max_steps:
            self.done = True
            obs += "\n[Max steps reached]"

        return obs, info, self.done

    def _parse_action(self, text: str) -> Optional[Dict]:
        """Extract JSON from agent response."""
        # Try to find JSON block
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[-1].split("```")[0].strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

        return None

    def get_task_answer(self) -> str:
        """Get the ground-truth answer for current task."""
        return str(self.current_task.get("answer", "")).strip()

    def get_current_task(self) -> Dict:
        """Get current task metadata."""
        return self.current_task
