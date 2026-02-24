"""
Train Market Encoder
=====================

Pretrain the Market Encoder with a self-supervised objective:
predict the 5-day forward log return from the current feature window.

Usage:
    python train_encoder.py --config configs/encoder_base.yaml --data data/processed/
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

from neural_alpha.encoder import MarketEncoder
from neural_alpha.utils import get_logger

logger = get_logger("train_encoder", log_file="logs/train_encoder.log")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/encoder_base.yaml")
    parser.add_argument("--data", default="data/processed/")
    parser.add_argument("--output", default="checkpoints/encoder/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    encoder = MarketEncoder(**model_cfg).to(device)

    # Projection head for pretraining: predict forward return
    proj = nn.Sequential(
        nn.Linear(model_cfg.get("latent_dim", 256), 64),
        nn.GELU(),
        nn.Linear(64, 1),
    ).to(device)

    data_dir = Path(args.data)
    try:
        X = torch.load(data_dir / "X_train.pt")
        y = torch.load(data_dir / "y_train.pt")
    except FileNotFoundError:
        logger.error("Training data not found. Run scripts/prepare_data.py first.")
        return

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=train_cfg.get("batch_size", 128),
        shuffle=True,
    )

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    best_loss = float("inf")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg.get("epochs", 30) + 1):
        encoder.train()
        proj.train()
        total_loss = 0.0

        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            h = encoder(features)         # (B, T, latent_dim)
            pred = proj(h[:, -1, :]).squeeze(-1)  # predict from last timestep
            loss = nn.functional.mse_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch:3d} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), out_dir / "best.pt")
            logger.info(f"  ↑ New best encoder checkpoint saved.")

    torch.save(encoder.state_dict(), out_dir / "latest.pt")
    logger.info("Encoder training complete.")


if __name__ == "__main__":
    main()
