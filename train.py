"""
NeuralAlpha — Main Training Script
====================================

End-to-end training for the full NeuralAlpha pipeline.
Trains MarketEncoder + AlphaTransformer + SignalSynthesizer jointly
using a multi-task loss:

    L = λ1 * L_direction  +  λ2 * L_ranking  +  λ3 * L_confidence

    L_direction:   Binary cross-entropy (did we get the direction right?)
    L_ranking:     ListNet ranking loss (did we rank assets correctly?)
    L_confidence:  Calibration loss (is confidence well-calibrated?)

Usage:
    python train.py --config configs/full_pipeline.yaml

    python train.py \
        --config configs/full_pipeline.yaml \
        --encoder checkpoints/encoder/best.pt \
        --causal checkpoints/causal/best.pt \
        --resume checkpoints/full/latest.pt
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import logging

from neural_alpha import NeuralAlphaPipeline
from neural_alpha.utils import BacktestMetrics, get_logger

logger = get_logger("train", log_file="logs/train.log")


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuralAlpha full pipeline")
    parser.add_argument("--config", type=str, default="configs/full_pipeline.yaml")
    parser.add_argument("--encoder", type=str, default=None, help="Pretrained encoder checkpoint")
    parser.add_argument("--causal", type=str, default=None, help="Pretrained causal checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def direction_loss(alpha_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """BCE loss on direction prediction."""
    labels = (returns > 0).float()
    probs = (alpha_pred + 1) / 2  # map (-1,1) → (0,1)
    return nn.functional.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), labels)


def ranking_loss(alpha_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    ListNet ranking loss.
    Encourages the model's rank ordering of alpha scores to match
    the rank ordering of actual forward returns.
    """
    # Normalize to probability distributions via softmax
    p_pred = torch.softmax(alpha_pred, dim=-1)
    p_true = torch.softmax(returns / returns.std().clamp(min=1e-6), dim=-1)
    # Cross-entropy between distributions
    return -(p_true * torch.log(p_pred.clamp(min=1e-9))).sum()


def confidence_calibration_loss(
    confidence: torch.Tensor,
    alpha_pred: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Confidence should correlate with correctness.
    Penalize high confidence on wrong predictions.
    """
    correct = ((alpha_pred > 0) == (returns > 0)).float()
    return nn.functional.mse_loss(confidence, correct)


def train_epoch(
    model: NeuralAlphaPipeline,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: dict,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    batches = 0

    lam1 = config.get("lambda_direction", 1.0)
    lam2 = config.get("lambda_ranking", 0.5)
    lam3 = config.get("lambda_confidence", 0.3)

    for features, returns in loader:
        features = features.to(device)
        returns = returns.to(device)

        optimizer.zero_grad()
        alpha, confidence, magnitude = model(features)

        l1 = direction_loss(alpha, returns)
        l2 = ranking_loss(alpha, returns)
        l3 = confidence_calibration_loss(confidence, alpha, returns)
        loss = lam1 * l1 + lam2 * l2 + lam3 * l3

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return {"loss": total_loss / max(batches, 1)}


@torch.no_grad()
def validate(
    model: NeuralAlphaPipeline,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_alpha = []
    all_returns = []

    for features, returns in loader:
        features = features.to(device)
        alpha, _, _ = model(features)
        all_alpha.append(alpha.cpu().numpy())
        all_returns.append(returns.numpy())

    all_alpha = np.concatenate(all_alpha)
    all_returns = np.concatenate(all_returns)

    metrics = BacktestMetrics()
    import pandas as pd
    ic = metrics.information_coefficient(
        pd.Series(all_alpha), pd.Series(all_returns)
    )
    hit = metrics.hit_rate(pd.Series(all_alpha), pd.Series(all_returns))

    return {"IC": ic, "hit_rate": hit}


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(args.device)
    logger.info(f"Training on device: {device}")

    # Initialize model
    model_cfg = config.get("model", {})
    model = NeuralAlphaPipeline(**model_cfg, device=args.device)

    # Load pretrained encoder if provided
    if args.encoder and Path(args.encoder).exists():
        logger.info(f"Loading pretrained encoder from {args.encoder}")
        sd = torch.load(args.encoder, map_location=args.device)
        model.encoder.load_state_dict(sd, strict=False)

    # Optimizer & scheduler
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    n_epochs = train_cfg.get("epochs", 50)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # Placeholder: real training uses prepared data files
    # See scripts/prepare_data.py to generate actual training tensors
    logger.info("Loading training data...")
    data_cfg = config.get("data", {})
    data_path = Path(data_cfg.get("processed_dir", "data/processed/"))

    try:
        X_train = torch.load(data_path / "X_train.pt")
        y_train = torch.load(data_path / "y_train.pt")
        X_val = torch.load(data_path / "X_val.pt")
        y_val = torch.load(data_path / "y_val.pt")
    except FileNotFoundError:
        logger.error(
            "Training data not found. Run: python scripts/prepare_data.py first."
        )
        return

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # Training loop
    best_ic = -float("inf")
    out_dir = Path(config.get("output_dir", "checkpoints/full/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, train_cfg, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"IC={val_metrics['IC']:.4f} | "
            f"hit={val_metrics['hit_rate']:.3f}"
        )

        # Save best checkpoint
        if val_metrics["IC"] > best_ic:
            best_ic = val_metrics["IC"]
            model.save(str(out_dir / "best"))
            logger.info(f"  ↑ New best IC={best_ic:.4f}. Saved.")

        # Save latest checkpoint
        model.save(str(out_dir / "latest"))

    logger.info(f"Training complete. Best IC: {best_ic:.4f}")


if __name__ == "__main__":
    main()
