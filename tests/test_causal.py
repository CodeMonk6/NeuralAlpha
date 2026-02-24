"""Tests for the Causal Engine module."""

import pytest
import numpy as np
import torch
from neural_alpha.causal import CausalEngine, CausalGraphEmbedder
from neural_alpha.causal.intervention import CausalIntervention


class TestCausalEngine:
    @pytest.fixture
    def simple_data(self):
        """Generate data from a known causal structure: X1 → X2 → X3."""
        np.random.seed(0)
        T = 500
        X1 = np.random.randn(T)
        X2 = 0.8 * X1 + 0.2 * np.random.randn(T)
        X3 = 0.7 * X2 + 0.3 * np.random.randn(T)
        # Add noise variables
        X4 = np.random.randn(T)
        X5 = np.random.randn(T)
        return np.column_stack([X1, X2, X3, X4, X5])

    def test_fit_returns_matrix(self, simple_data):
        engine = CausalEngine(d=5, max_iter=50, threshold=0.1)
        adj = engine.fit(simple_data)
        assert adj.shape == (5, 5)

    def test_diagonal_is_zero(self, simple_data):
        engine = CausalEngine(d=5, max_iter=50, threshold=0.1)
        adj = engine.fit(simple_data)
        assert np.diag(adj).sum() == 0.0, "Self-loops detected in adjacency matrix"

    def test_get_adjacency_before_fit_raises(self):
        engine = CausalEngine(d=5)
        with pytest.raises(RuntimeError):
            engine.get_adjacency()

    def test_edge_list(self, simple_data):
        engine = CausalEngine(d=5, max_iter=50, threshold=0.1)
        engine.fit(simple_data)
        edges = engine.get_edge_list()
        assert isinstance(edges, list)
        for edge in edges:
            assert len(edge) == 3  # (i, j, weight)

    def test_summary(self, simple_data):
        engine = CausalEngine(d=5, max_iter=50, threshold=0.1)
        engine.fit(simple_data)
        summary = engine.summary()
        assert "CausalEngine" in summary


class TestCausalGraphEmbedder:
    @pytest.fixture
    def embedder(self):
        return CausalGraphEmbedder(n_nodes=5, node_feat_dim=8, embed_dim=32)

    def test_output_shape(self, embedder):
        adj = torch.rand(5, 5)
        emb = embedder(adj)
        assert emb.shape == (32,), f"Expected (32,), got {emb.shape}"

    def test_output_dim_property(self, embedder):
        assert embedder.output_dim == 32

    def test_with_node_features(self, embedder):
        adj = torch.rand(5, 5)
        feats = torch.randn(5, 8)
        emb = embedder(adj, node_features=feats)
        assert emb.shape == (32,)

    def test_no_nan(self, embedder):
        adj = torch.rand(5, 5)
        emb = embedder(adj)
        assert not torch.isnan(emb).any()


class TestCausalIntervention:
    @pytest.fixture
    def intervention_engine(self):
        # X1 → X2 → X3  (simple chain)
        adj = np.zeros((3, 3))
        adj[0, 1] = 0.8   # X1 → X2
        adj[1, 2] = 0.7   # X2 → X3
        return CausalIntervention(adj, node_names=["X1", "X2", "X3"])

    def test_no_intervention(self, intervention_engine):
        X_obs = np.array([1.0, 0.8, 0.56])
        X_int = intervention_engine.intervene(X_obs, {})
        np.testing.assert_array_almost_equal(X_obs, X_int)

    def test_intervention_on_root(self, intervention_engine):
        X_obs = np.array([1.0, 0.8, 0.56])
        X_int = intervention_engine.intervene(X_obs, {0: 2.0})
        # X1 forced to 2.0
        assert X_int[0] == 2.0
        # X2 should be updated: 0.8 * 2.0 = 1.6
        assert abs(X_int[1] - 1.6) < 0.01

    def test_counterfactual_return_impact(self, intervention_engine):
        X_obs = np.array([1.0, 0.8, 0.56])
        obs_ret, cf_ret = intervention_engine.counterfactual_return_impact(
            X_obs, interventions={0: 3.0}, return_node=2
        )
        assert isinstance(obs_ret, float)
        assert isinstance(cf_ret, float)

    def test_causal_paths(self, intervention_engine):
        paths = intervention_engine.get_causal_paths(0, 2)
        assert len(paths) >= 1
        assert [0, 1, 2] in paths
