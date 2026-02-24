"""
Causal Graph Embeddings
========================

Graph Attention Network (GAT) that encodes the learned causal DAG
into node embeddings, which are then used as conditioning signal
in the AlphaTransformer.

Input:  adjacency matrix A ∈ ℝ^{d × d}  +  node features F ∈ ℝ^{d × f}
Output: node embeddings E ∈ ℝ^{d × embed_dim}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention (GAT) layer.

    Computes:
        e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
        α_ij = softmax_j(e_ij)   (over neighbors j)
        h'_i = σ( Σ_j α_ij · Wh_j )

    Args:
        in_features:    Input node feature dimension.
        out_features:   Output node embedding dimension.
        n_heads:        Number of attention heads (multi-head GAT).
        dropout:        Attention coefficient dropout.
        alpha:          LeakyReLU negative slope.
        concat:         If True, concatenate heads; else average.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h:   Node features (n, in_features)
            adj: Adjacency matrix (n, n) — use weighted adj from CausalEngine

        Returns:
            out: Node embeddings (n, out_features * n_heads) if concat else (n, out_features)
        """
        N = h.size(0)

        # Linear transform per head: (n_heads, N, out_features)
        Wh = torch.einsum("hio,ni->hno", self.W, h)

        # Attention logits: e_ij = a^T [Wh_i || Wh_j]
        # Wh_i repeated for all j; Wh_j repeated for all i
        # Simpler approach: pair-wise attention
        Wh_i = Wh.unsqueeze(3).repeat(1, 1, 1, N, 1).view(self.n_heads, N * N, self.out_features)
        Wh_j = Wh.unsqueeze(2).repeat(1, 1, N, 1, 1).view(self.n_heads, N * N, self.out_features)

        pair = torch.cat([Wh_i, Wh_j], dim=-1)  # (H, N*N, 2*F)
        e = self.leaky_relu((pair * self.a.unsqueeze(1)).sum(-1))  # (H, N*N)
        e = e.view(self.n_heads, N, N)  # (H, N, N)

        # Mask non-edges with -inf so softmax ignores them
        adj_mask = (adj == 0).unsqueeze(0)  # (1, N, N)
        e = e.masked_fill(adj_mask, float("-inf"))

        alpha_coef = F.softmax(e, dim=-1)  # (H, N, N)
        alpha_coef = torch.nan_to_num(alpha_coef, nan=0.0)  # handle all-inf rows
        alpha_coef = self.dropout(alpha_coef)

        # Aggregate: h'_i = Σ_j α_ij · Wh_j
        Wh = Wh.view(self.n_heads, N, self.out_features)
        out = torch.bmm(alpha_coef, Wh)  # (H, N, out_features)

        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, self.n_heads * self.out_features)
        else:
            out = out.mean(dim=0)  # (N, out_features)

        return F.elu(out)


class CausalGraphEmbedder(nn.Module):
    """
    Multi-layer GAT over the causal DAG to produce node embeddings
    that encode structural causal context.

    These embeddings are pooled and used as a conditioning signal
    for the AlphaTransformer.

    Args:
        n_nodes:        Number of causal graph nodes (= n_factors).
        node_feat_dim:  Input node feature dimension.
        embed_dim:      Final embedding dimension per node.
        n_layers:       Number of GAT layers.
        n_heads:        Attention heads (per layer).
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        n_nodes: int = 12,
        node_feat_dim: int = 16,
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim

        layers = []
        in_dim = node_feat_dim
        for i in range(n_layers):
            out_dim = embed_dim if i == n_layers - 1 else embed_dim // 2
            concat = i < n_layers - 1
            layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )
            in_dim = out_dim * n_heads if concat else out_dim

        self.gat_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable node feature embeddings (used when external features not provided)
        self.node_emb = nn.Embedding(n_nodes, node_feat_dim)

    def forward(
        self,
        adj: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            adj:           Adjacency matrix (n_nodes, n_nodes).
            node_features: Optional external node features (n_nodes, node_feat_dim).
                           If None, uses learnable embeddings.

        Returns:
            graph_embed: Global graph embedding (embed_dim,) via mean pooling.
                         Or (n_nodes, embed_dim) if per-node embedding needed.
        """
        if node_features is None:
            idx = torch.arange(self.n_nodes, device=adj.device)
            h = self.node_emb(idx)  # (n_nodes, node_feat_dim)
        else:
            h = node_features

        for layer in self.gat_layers:
            h = layer(h, adj)

        h = self.norm(h)
        # Global mean pooling → single graph embedding
        return h.mean(dim=0)  # (embed_dim,)

    @property
    def output_dim(self) -> int:
        return self.embed_dim
