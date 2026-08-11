"""Tests for operational goal-geometry measurements."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch

    from goal_geometry import (
        collect_geometry_diagnostics,
        jensen_shannon_distance,
        matrix_rank_correlation,
        pairwise_euclidean,
        pca_projection,
    )
    from multi_goal_gru import MultiGoalGRUAgent
    from quantum_environments import QuantumEnvironment
    from quantum_rl_common import GoalSpec
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class GeometryTests(unittest.TestCase):
    def test_distance_utilities(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        self.assertEqual(jensen_shannon_distance(p, p), 0.0)
        self.assertAlmostEqual(jensen_shannon_distance(p, q), jensen_shannon_distance(q, p))
        rows = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        distances = pairwise_euclidean(rows)
        self.assertEqual(distances[0, 1], 5.0)
        projection, explained = pca_projection(rows)
        self.assertEqual(projection.shape, (3, 2))
        self.assertAlmostEqual(float(explained.sum()), 1.0)
        self.assertAlmostEqual(matrix_rank_correlation(distances, distances), 1.0)

    def test_collector_returns_all_three_geometries(self) -> None:
        env = QuantumEnvironment(initial_state="one", seed=4)
        goals = (GoalSpec("Z1", ((0, 1),)), GoalSpec("X0", ((1, 0),)))
        agent = MultiGoalGRUAgent(
            n_actions=env.n_actions,
            n_outcomes=env.n_outcomes,
            action_outcome_counts=env.action_outcome_counts,
            goals=goals,
            hidden_dim=8,
            goal_dim=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=2,
            device="cpu",
        )
        diagnostics = collect_geometry_diagnostics(
            agent,
            env,
            goals,
            episodes_per_goal=2,
            max_steps=3,
        )
        for key in ("embedding_distance", "strategy_distance", "trajectory_distance"):
            matrix = np.asarray(diagnostics[key])
            self.assertEqual(matrix.shape, (2, 2))
            self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertEqual(len(diagnostics["reachability"]), 2)
        self.assertEqual(len(diagnostics["reachability_curves"]), 6)
        # With two goals there is only one off-diagonal distance, so rank
        # correlation is correctly undefined rather than spuriously perfect.
        self.assertTrue(math.isnan(float(diagnostics["embedding_strategy_spearman"])))


if __name__ == "__main__":
    unittest.main()
