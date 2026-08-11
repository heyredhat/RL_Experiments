"""Tests for inverse-designed two-dimensional hodological geometry."""

from __future__ import annotations

import unittest

import numpy as np

from spatial_hodology import (
    COORDINATES,
    canonical_actions,
    classical_mds,
    exact_movement_costs,
    geometry_metrics,
    normalized_stress,
    PlaceQAgent,
    procrustes_r2,
    search_diagonal_instrument,
)


class SpatialHodologyTests(unittest.TestCase):
    def test_classical_mds_recovers_euclidean_grid(self) -> None:
        delta = COORDINATES[:, None, :] - COORDINATES[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=-1))
        embedded, eigenvalues = classical_mds(distances, 2)
        self.assertLess(normalized_stress(distances, embedded), 1e-10)
        self.assertGreater(procrustes_r2(embedded, COORDINATES), 1.0 - 1e-10)
        self.assertLess(abs(eigenvalues[2]), 1e-9)

    def test_cost_matched_diagonals_are_more_euclidean_than_cardinal_moves(self) -> None:
        optimized = geometry_metrics(exact_movement_costs(1 / np.sqrt(2)))
        cardinal = geometry_metrics(exact_movement_costs(cardinal_only=True))
        self.assertLess(optimized["stress_2d"], 0.06)
        self.assertLess(optimized["stress_2d"], cardinal["stress_2d"])
        self.assertGreater(optimized["stress_1d"], 0.35)
        self.assertLess(optimized["stress_2d"], 0.2 * optimized["stress_1d"])

    def test_outer_search_finds_sqrt_two_cost_neighborhood(self) -> None:
        _, best = search_diagonal_instrument(np.linspace(0.65, 0.76, 23))
        self.assertLess(abs(best["diagonal_success"] - 1 / np.sqrt(2)), 0.03)

    def test_canonical_histories_end_at_requested_sites(self) -> None:
        for site in range(9):
            x = y = 1
            for action in canonical_actions(site):
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
                x, y = x + dx, y + dy
            self.assertEqual(y * 3 + x, site)

    def test_place_agent_discards_incoming_action_but_retains_place(self) -> None:
        agent = PlaceQAgent(n_actions=9, epsilon_start=0.0, epsilon_end=0.0)
        agent.reset_episode(0, 1, False)
        agent.act(0, 0, False)
        agent.observe(0, 0, 2, 7, 0.0, False, 0, False)
        first = agent._state(0, 0)
        agent.act(0, 0, False)
        agent.observe(0, 0, 5, 7, 0.0, False, 0, False)
        self.assertEqual(first, agent._state(0, 0))


if __name__ == "__main__":
    unittest.main()
