# debug_env.py
import sys
import torch
import transformers
import trl

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"Transformers: {transformers.__version__}")
print(f"TRL: {trl.__version__}")

# Test model loading
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    print("[OK] Tokenizer loads from local")
except Exception as e:
    print(f"[FAIL] Tokenizer: {e}")

# Test dataset loading
try:
    from data_utils import load_preference_dataset
    train, eval = load_preference_dataset(n_train=10, n_test=5)
    print(f"[OK] Dataset loads: {len(train)} train, {len(eval)} eval")
except Exception as e:
    print(f"[FAIL] Dataset: {e}")