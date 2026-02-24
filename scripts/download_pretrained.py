"""
Download Pretrained Weights
============================

Downloads NeuralAlpha pretrained checkpoints from the public release.

Usage:
    python scripts/download_pretrained.py
    python scripts/download_pretrained.py --model full --out checkpoints/
"""

import argparse
import os
import urllib.request
from pathlib import Path
import json

PRETRAINED_REGISTRY = {
    "full": {
        "description": "Full NeuralAlpha pipeline (encoder + transformer + synthesizer)",
        "files": ["model.pt", "config.json"],
        "base_url": "https://github.com/sourabh-sharma/NeuralAlpha/releases/download/v0.1.0/",
        "output_dir": "checkpoints/full/best/",
    },
    "encoder": {
        "description": "Pretrained Market Encoder only",
        "files": ["encoder.pt"],
        "base_url": "https://github.com/sourabh-sharma/NeuralAlpha/releases/download/v0.1.0/",
        "output_dir": "checkpoints/encoder/",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=list(PRETRAINED_REGISTRY.keys()),
        help="Which pretrained model to download",
    )
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def download_file(url: str, dest: str) -> None:
    """Download a file with a progress bar."""
    print(f"  Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  ✓ Saved {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        print(f"    Please download manually from the GitHub releases page.")


def main():
    args = parse_args()
    config = PRETRAINED_REGISTRY[args.model]

    out_dir = Path(args.out or config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {config['description']}")
    print(f"Destination: {out_dir}\n")

    for filename in config["files"]:
        url = config["base_url"] + filename
        dest = out_dir / filename
        download_file(url, str(dest))

    print(f"\n✓ Done. Load with:")
    print(f"  from neural_alpha import NeuralAlphaPipeline")
    print(f"  pipeline = NeuralAlphaPipeline.from_pretrained('{out_dir}')\n")


if __name__ == "__main__":
    main()
