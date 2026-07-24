import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # FORCE CPU

import sys
print(f"[1] Python started: {sys.version}", flush=True)

import torch
print(f"[2] PyTorch loaded. CUDA visible: {torch.cuda.is_available()}", flush=True)

from transformers import AutoTokenizer
print("[3] Transformers loaded", flush=True)

from data_utils import load_mock_data
print("[4] Data utils loaded", flush=True)

# Load tokenizer (no GPU memory needed)
print("[5] Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct", trust_remote_code=True)
print("[6] Tokenizer OK", flush=True)

# Load mock data
print("[7] Loading mock data...", flush=True)
raw = load_mock_data(20)
print(f"[8] Mock data: {len(raw)} samples", flush=True)

# Try model loading on CPU
print("[9] Loading model on CPU... (this takes ~30 seconds)", flush=True)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map=None,  # CPU
    trust_remote_code=True,
)
print(f"[10] Model loaded on CPU. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

print("\n[SUCCESS] Your environment works. The silent failure was GPU OOM.")
print("Run full training on CPU (slow) or use Google Colab / AutoDL for GPU.")