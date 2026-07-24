import json, os

# Test 1: Can we load mock data?
print("[1] Loading mock data...")
from data_utils import load_mock_data
raw = load_mock_data(5)
print(f"[OK] {len(raw)} samples loaded")
print(f"  Sample prompt: {raw[0]['prompt'][:50]}...")

# Test 2: Can we build the JSON format DPO expects?
print("\n[2] Building DPO-format dataset...")
dpo_format = {
    "prompt": [d["prompt"] for d in raw],
    "chosen": [d["chosen"] for d in raw],
    "rejected": [d["rejected"] for d in raw],
}
print(f"[OK] Dataset dict ready: {len(dpo_format['prompt'])} items")

# Test 3: Does the dataset save/load?
print("\n[3] Saving to JSON...")
os.makedirs("output", exist_ok=True)
with open("output/test_data.json", "w") as f:
    json.dump(raw, f, indent=2)
print("[OK] Saved to output/test_data.json")

print("\n[SUCCESS] Data pipeline works. Ready for AutoDL training.")