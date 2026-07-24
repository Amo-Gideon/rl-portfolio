import os
from pathlib import Path

LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def download_model():
    if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        print(f"[OK] Model found locally at {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    print(f"Downloading {MODEL_ID} ...")

    # Try HuggingFace first (standard for GitHub/international)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_MODEL_DIR, local_dir_use_symlinks=False)
        print(f"[OK] Downloaded from HuggingFace to {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR
    except Exception as e:
        print(f"HuggingFace failed ({e}), trying ModelScope...")

    # Fallback to ModelScope (better for China)
    try:
        from modelscope import snapshot_download
        snapshot_download(MODEL_ID, local_dir=LOCAL_MODEL_DIR)
        print(f"[OK] Downloaded from ModelScope to {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR
    except Exception as e:
        print(f"ModelScope also failed: {e}")
        raise

if __name__ == "__main__":
    download_model()