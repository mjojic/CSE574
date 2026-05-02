#!/usr/bin/env python3
"""Download Qwen/Qwen3-32B-FP8 into a Hugging Face Hub cache directory.

Install dependency::

    pip install huggingface_hub

If download fails with 401/403, set a token (accept the model license on Hugging Face first if prompted)::

    export HF_TOKEN=hf_...

Usage::

    python scripts/download_qwen3_32b_fp8.py
"""

from __future__ import annotations

import os
import sys

REPO_ID = "Qwen/Qwen3-32B-FP8"
HUB_CACHE_DIR = "/scratch/mjojic/huggingface/hub"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing package: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    os.makedirs(HUB_CACHE_DIR, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    path = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=HUB_CACHE_DIR,
        token=token,
    )
    print(path)


if __name__ == "__main__":
    main()
