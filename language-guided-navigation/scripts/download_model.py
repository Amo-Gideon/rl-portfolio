#!/usr/bin/env python3
"""
Download the base model from ModelScope (preferred in PAI-DSW) or a HuggingFace mirror.

Usage:
    python scripts/download_model.py
    # then set model.name in configs/sft.yaml and configs/rl.yaml to the printed path
"""

import argparse
import os
from pathlib import Path


def modelscope_download(model_id: str, cache_root: str) -> str:
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(model_id, cache_dir=cache_root)
    # ModelScope stores under cache_root/<organization>/<model>
    local_path = Path(cache_root) / model_id.replace("/", os.sep)
    return str(local_path.resolve())


def hf_mirror_download(model_id: str, local_dir: str, endpoint: str = "https://hf-mirror.com") -> str:
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    from huggingface_hub import snapshot_download
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    return str(Path(local_dir).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache_root", type=str, default="models")
    parser.add_argument("--source", type=str, choices=["modelscope", "hf-mirror"], default="modelscope")
    args = parser.parse_args()

    if args.source == "modelscope":
        local_path = modelscope_download(args.model_id, args.cache_root)
    else:
        local_path = hf_mirror_download(args.model_id, f"{args.cache_root}/{args.model_id.split('/')[-1]}")

    print(f"\nDownloaded model to:\n{local_path}")
    print(f"\nSet model.name to:\n{local_path}")
