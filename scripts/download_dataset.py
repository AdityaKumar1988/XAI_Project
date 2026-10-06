"""
Dataset Downloader & Benchmark Preparer for Document Forgery Detection
Supports:
1. Academic Transcripts Forensics Benchmark (with pixel-level masks and zero-leakage master IDs)
2. Find it again! (ICDAR 2021)
3. AIForge-Doc (Hugging Face gated benchmark)
"""

import os
import sys
import argparse
import logging
import json
import urllib.request
import zipfile
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_dataset")

def download_aiforge_hf(token=None, output_dir="data/raw/aiforge"):
    """Downloads AIForge-Doc from Hugging Face if token is provided."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN environment variable not set. AIForge-Doc requires authentication.\n"
            "Please refer to DATASET_SETUP.md for setup instructions."
        )
        return False
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Downloading AIForge-Doc v1 to {output_dir}...")
        snapshot_download(
            repo_id="Scam-AI/AIForge-Doc-v1",
            repo_type="dataset",
            token=token,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        logger.info("Successfully downloaded AIForge-Doc dataset.")
        return True
    except Exception as e:
        logger.error(f"Failed to download AIForge-Doc from Hugging Face: {e}")
        return False

def setup_find_it_again(raw_dir="data/raw/find_it_again"):
    """Validates or prepares Find it again! benchmark directory."""
    os.makedirs(raw_dir, exist_ok=True)
    kaggle_notice = (
        "Find it again! (ICDAR 2021) can be downloaded from:\n"
        "https://www.kaggle.com/datasets/beatrizmartneztorns/find-it-again-dataset\n"
        f"Place the extracted folders into: {raw_dir}/"
    )
    logger.info(kaggle_notice)
    return os.path.exists(os.path.join(raw_dir, "images"))

def main():
    parser = argparse.ArgumentParser(description="Download and prepare document forgery datasets.")
    parser.add_argument("--dataset", type=str, default="academic_transcripts",
                        choices=["academic_transcripts", "find_it_again", "aiforge", "all"],
                        help="Dataset to download/prepare")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()

    if args.dataset in ["aiforge", "all"]:
        download_aiforge_hf(token=args.hf_token)
    if args.dataset in ["find_it_again", "all"]:
        setup_find_it_again()
    if args.dataset in ["academic_transcripts", "all"]:
        logger.info("Academic Transcripts suite can be prepared via: python scripts/prepare_dataset.py")

if __name__ == "__main__":
    main()
