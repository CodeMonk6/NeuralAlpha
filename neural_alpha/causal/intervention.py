"""
Causal Interventions (Do-Calculus)
====================================

Implements do-calculus style interventions on the learned causal graph.
Useful for counterfactual scenario analysis:
  "If the Fed hikes 50bps unexpectedly (do(rate_shock=+2σ)),
   what is the expected return impact on growth equities?"

Approach:
    Given the fitted structural equation model (linear SEM):
        X = X·W + ε
    An intervention do(X_i = v) sets node i to value v and removes
    all incoming edges to i (graph mutilation).

    We then propagate the intervention forward through the DAG
    using topological ordering.
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple


class CausalIntervention:
    """
    Do-calculus intervention engine operating on a linear SEM.

    Args:
        adj_matrix:    Fitted adjacency matrix W (d, d) from CausalEngine.
        node_names:    List of variable names (length d).
    """

    def __init__(self, adj_matrix: np.ndarray, node_names: Optional[list] = None):
        self.W = adj_matrix.copy()
        self.d = adj_matrix.shape[0]
        self.node_names = node_names or [f"X{i}" for i in range(self.d)]

        # Build networkx DiGraph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.d))
        for i in range(self.d):
            for j in range(self.d):
                if self.W[i, j] != 0:
                    self.G.add_edge(i, j, weight=float(self.W[i, j]))

        # Topological order (for propagation)
        if nx.is_directed_acyclic_graph(self.G):
            self.topo_order = list(nx.topological_sort(self.G))
        else:
            # Fallback: use approximate order (shouldn't happen with fitted DAG)
            self.topo_order = list(range(self.d))

    def intervene(
        self,
        X_obs: np.ndarray,
        interventions: Dict[int, float],
    ) -> np.ndarray:
        """
        Apply hard interventions do(X_i = v) to observed data.

        Args:
            X_obs:         Observed factor values (d,).
            interventions: Dict {node_index: intervention_value}.

        Returns:
            X_int: Post-intervention factor values (d,).
        """
        X_int = X_obs.copy()

        # Graph mutilation: set intervened nodes
        for node, val in interventions.items():
            X_int[node] = val

        # Propagate forward through topological order
        W_mutilated = self.W.copy()
        for node in interventions:
            W_mutilated[:, node] = 0.0  # remove incoming edges

        for j in self.topo_order:
            if j in interventions:
                continue  # fixed by intervention
            # Structural equation: X_j = Σ_i W_ij * X_i  (exogenous noise ε ≈ residual)
            parents = np.nonzero(W_mutilated[:, j])[0]
            if len(parents) > 0:
                X_int[j] = sum(W_mutilated[p, j] * X_int[p] for p in parents)

        return X_int

    def counterfactual_return_impact(
        self,
        X_obs: np.ndarray,
        interventions: Dict[int, float],
        return_node: int,
    ) -> Tuple[float, float]:
        """
        Estimate the counterfactual return impact of an intervention.

        Returns:
            (observed_return, counterfactual_return)
        """
        X_int = self.intervene(X_obs, interventions)
        return float(X_obs[return_node]), float(X_int[return_node])

    def sensitivity_analysis(
        self,
        X_obs: np.ndarray,
        target_node: int,
        shock_range: np.ndarray,
    ) -> np.ndarray:
        """
        Sweep a range of shock values for each parent of target_node
        and report the resulting change in target_node.

        Args:
            X_obs:       Observed factor values (d,).
            target_node: Index of the variable to track.
            shock_range: Array of shock magnitudes (in std units) to test.

        Returns:
            impact_matrix: (d, len(shock_range)) — impact on target_node per parent per shock.
        """
        parents = list(self.G.predecessors(target_node))
        impact = np.zeros((self.d, len(shock_range)))

        for p in parents:
            base = float(X_obs[target_node])
            for k, shock in enumerate(shock_range):
                X_int = self.intervene(X_obs, {p: X_obs[p] + shock})
                impact[p, k] = X_int[target_node] - base

        return impact

    def get_causal_paths(self, source: int, target: int) -> list:
        """Return all directed paths from source to target in the causal DAG."""
        try:
            return list(nx.all_simple_paths(self.G, source, target))
        except nx.NetworkXNoPath:
            return []

    def __repr__(self) -> str:
        return (
            f"CausalIntervention(d={self.d}, "
            f"edges={self.G.number_of_edges()}, "
            f"nodes={self.node_names[:3]}...)"
        )
