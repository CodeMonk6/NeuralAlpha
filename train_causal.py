"""
Train Causal Engine
====================

Runs the NOTEARS causal discovery algorithm on macro/factor data
to learn the structural causal graph.

Usage:
    python train_causal.py --config configs/causal_discovery.yaml --data data/processed/
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import logging

from neural_alpha.causal import CausalEngine, CausalGraphEmbedder
from neural_alpha.utils import get_logger

logger = get_logger("train_causal", log_file="logs/train_causal.log")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_discovery.yaml")
    parser.add_argument("--data", default="data/processed/")
    parser.add_argument("--output", default="checkpoints/causal/")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    causal_cfg = config.get("causal_engine", {})
    embedder_cfg = config.get("graph_embedder", {})
    factor_names = config.get("factor_nodes", [])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load factor data if available, otherwise use placeholder
    data_dir = Path(args.data)
    factor_path = data_dir / "factor_data.npy"

    if factor_path.exists():
        X = np.load(factor_path)
        logger.info(f"Loaded factor data: {X.shape}")
    else:
        logger.warning("Factor data not found. Using synthetic data for demonstration.")
        # Synthetic: generate data with known causal structure for testing
        np.random.seed(42)
        T = 1000
        d = causal_cfg.get("d", 12)
        X = np.random.randn(T, d)
        # Inject some causal structure
        X[:, 1] += 0.7 * X[:, 0]
        X[:, 2] += 0.5 * X[:, 1]
        X[:, 4] += 0.6 * X[:, 3]

    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-9)

    # Run causal discovery
    logger.info("Running NOTEARS causal discovery...")
    engine = CausalEngine(**causal_cfg)
    adj_matrix = engine.fit(X)

    logger.info(engine.summary())

    # Save adjacency matrix
    np.save(out_dir / "adjacency.npy", adj_matrix)
    logger.info(f"Adjacency matrix saved: {out_dir / 'adjacency.npy'}")

    # Save edge list
    edges = engine.get_edge_list()
    import json
    with open(out_dir / "edges.json", "w") as f:
        json.dump(
            {
                "edges": edges,
                "n_edges": len(edges),
                "n_nodes": causal_cfg.get("d", 12),
                "node_names": factor_names,
            },
            f,
            indent=2,
        )

    # Train graph embedder on the discovered graph
    logger.info("Training CausalGraphEmbedder...")
    adj_t = torch.tensor(adj_matrix, dtype=torch.float32)
    embedder = CausalGraphEmbedder(**embedder_cfg)
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)

    for step in range(200):
        emb = embedder(adj_t)
        # Self-supervised: maximize variance of embeddings (avoid collapse)
        loss = -emb.var()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            logger.debug(f"Embedder step {step} | loss={loss.item():.4f}")

    torch.save(embedder.state_dict(), out_dir / "embedder.pt")
    logger.info(f"Embedder saved: {out_dir / 'embedder.pt'}")
    logger.info("Causal engine training complete.")


if __name__ == "__main__":
    main()
