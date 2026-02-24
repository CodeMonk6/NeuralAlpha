"""
Causal Discovery Engine
========================

Learns a directed acyclic graph (DAG) over macro/financial factors
and asset return residuals using a differentiable NOTEARS-style approach.

Algorithm:
    1. Standardize input factor matrix X ∈ ℝ^{T × d}.
    2. Parameterize adjacency matrix W ∈ ℝ^{d × d}.
    3. Optimize:
         min  ½·‖X − XW‖_F²  +  λ₁·‖W‖₁  +  λ₂·h(W)
       where h(W) = tr(e^{W∘W}) − d  is the NOTEARS acyclicity constraint.
    4. Threshold small weights → binary DAG A.

Reference:
    Zheng et al. (2018). DAGs with NO TEARS: Continuous Optimization
    for Structure Learning. NeurIPS 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NOTEARSLayer(nn.Module):
    """
    Learnable adjacency matrix with acyclicity regularization.

    Args:
        d: Number of nodes (variables) in the causal graph.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Initialize W close to zero (no edges)
        self.W_raw = nn.Parameter(torch.zeros(d, d))
        # Mask diagonal to zero (no self-loops)
        mask = torch.ones(d, d) - torch.eye(d)
        self.register_buffer("mask", mask)

    def forward(self) -> torch.Tensor:
        """Returns the masked adjacency matrix W."""
        return self.W_raw * self.mask

    def acyclicity_constraint(self) -> torch.Tensor:
        """
        NOTEARS h(W) = tr(e^{W∘W}) − d ≥ 0, = 0 iff W is a DAG.
        Uses the matrix exponential approximation via eigenvalues.
        """
        W = self.forward()
        W2 = W * W  # element-wise square
        # Matrix exponential via expm approximation (truncated series for efficiency)
        M = W2
        expm_approx = torch.eye(self.d, device=W.device)
        M_power = torch.eye(self.d, device=W.device)
        for k in range(1, 10):  # 10-term Taylor expansion
            M_power = M_power @ M / k
            expm_approx = expm_approx + M_power
        h = expm_approx.trace() - self.d
        return h


class CausalEngine:
    """
    Full causal discovery pipeline for financial factor graphs.

    Usage:
        engine = CausalEngine(d=12, lambda1=0.01, lambda2=1.0)
        adj_matrix = engine.fit(X)           # X: (T, d) factor matrix
        graph_embed = engine.get_embeddings() # (d, d) adjacency for GNN

    Args:
        d:          Number of variables (factors + return residuals).
        lambda1:    L1 sparsity penalty on W.
        lambda2:    Acyclicity penalty coefficient (augmented Lagrangian).
        threshold:  Edge weight below which edges are zeroed in final DAG.
        max_iter:   Maximum optimization iterations.
        lr:         Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        d: int = 12,
        lambda1: float = 0.01,
        lambda2: float = 1.0,
        threshold: float = 0.3,
        max_iter: int = 500,
        lr: float = 1e-2,
    ):
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.threshold = threshold
        self.max_iter = max_iter
        self.lr = lr

        self.model = NOTEARSLayer(d)
        self.adj_matrix_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the causal graph to observed factor data.

        Args:
            X: Factor matrix of shape (T, d).
               Each column is one variable (macro factor / return residual).
               Strongly recommend standardizing before calling fit.

        Returns:
            adj_matrix: Thresholded adjacency matrix (d, d) with float edge weights.
        """
        T, d = X.shape
        assert d == self.d, f"Expected d={self.d} variables, got {d}"

        # Standardize
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True).clip(min=1e-9)
        X_std = (X - mu) / sigma

        X_t = torch.tensor(X_std, dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        rho = 1.0   # augmented Lagrangian penalty
        alpha = 0.0  # dual variable
        prev_h = float("inf")

        for outer_iter in range(10):  # outer AL loop
            for inner_iter in range(self.max_iter):
                optimizer.zero_grad()

                W = self.model()
                # Reconstruction loss: ½·‖X - XW‖_F²
                X_hat = X_t @ W
                recon_loss = 0.5 * ((X_t - X_hat) ** 2).mean()

                # L1 sparsity
                l1_loss = self.lambda1 * W.abs().sum()

                # Acyclicity constraint h(W)
                h = self.model.acyclicity_constraint()
                dag_penalty = alpha * h + 0.5 * rho * h ** 2

                loss = recon_loss + l1_loss + dag_penalty
                loss.backward()
                optimizer.step()

                if inner_iter % 100 == 0:
                    logger.debug(
                        f"[AL outer={outer_iter} inner={inner_iter}] "
                        f"loss={loss.item():.4f} recon={recon_loss.item():.4f} h={h.item():.6f}"
                    )

            # Update AL dual variable
            h_val = self.model.acyclicity_constraint().item()
            alpha += rho * h_val
            if h_val > 0.25 * prev_h:
                rho *= 10.0
            prev_h = h_val

            if abs(h_val) < 1e-8:
                logger.info(f"DAG constraint satisfied at outer iteration {outer_iter}. h={h_val:.2e}")
                break

        with torch.no_grad():
            W_final = self.model().numpy()

        # Threshold small edges
        W_final[np.abs(W_final) < self.threshold] = 0.0
        self.adj_matrix_ = W_final
        self._fitted = True

        n_edges = (W_final != 0).sum()
        logger.info(f"Causal graph fitted: {n_edges} edges discovered over {d} nodes.")
        return W_final

    def get_adjacency(self) -> np.ndarray:
        """Return the fitted adjacency matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_adjacency().")
        return self.adj_matrix_

    def get_edge_list(self) -> list:
        """Return list of (i, j, weight) tuples for non-zero edges."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_edge_list().")
        edges = []
        for i in range(self.d):
            for j in range(self.d):
                w = self.adj_matrix_[i, j]
                if w != 0:
                    edges.append((i, j, float(w)))
        return edges

    def summary(self) -> str:
        if not self._fitted:
            return "CausalEngine (not fitted)"
        n_edges = (self.adj_matrix_ != 0).sum()
        density = n_edges / (self.d * (self.d - 1))
        return (
            f"CausalEngine(d={self.d}, edges={n_edges}, density={density:.3f}, "
            f"lambda1={self.lambda1}, threshold={self.threshold})"
        )
