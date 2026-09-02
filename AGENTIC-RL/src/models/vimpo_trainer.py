"""
Value-Implicit Policy Optimization (VIMPO) Trainer for Agentic RL.

Extends GRPO with per-token credit assignment via backward dynamic programming.
Instead of assigning the same flat advantage to all tokens in a response (GRPO),
VIMPO computes a policy-implied value function backward from the terminal reward,
giving each token its own advantage based on its position and the full policy distribution.

References:
  - VIMPO: Value-Implicit Policy Optimization for LLMs (UC Berkeley, 2026)
    https://github.com/backprop07/VIMPO
  - DeepSeek-R1: GRPO for reasoning (2025)
"""