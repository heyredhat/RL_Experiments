"""Tests for the self-contained low-dimensional search."""

from __future__ import annotations

import unittest

import numpy as np

import search_low_dimensional as search


class TestLowDimensionalSearch(unittest.TestCase):
    def test_hesse_sic_is_complete_and_equiangular(self) -> None:
        states, _, _ = search.hesse_sic()
        completeness = sum(search.ket_density(state) / 3 for state in states)
        overlaps = np.abs(states.conj() @ states.T) ** 2
        self.assertTrue(np.allclose(completeness, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(search.upper_values(overlaps), 0.25, atol=1e-12))

    def test_weyl_controls_close_on_nine_states(self) -> None:
        states, x, z = search.hesse_sic()
        transitions = search.hesse_transitions(states, [x, x.conj().T, z, z.conj().T])
        self.assertEqual(transitions.shape, (4, 9))
        self.assertTrue(all(len(set(row.tolist())) == 9 for row in transitions))
        self.assertTrue(np.array_equal(search.shortest_control_distance(transitions), search.torus_manhattan()))

    def test_sic_hitting_cost_respects_torus_distance_classes(self) -> None:
        states, x, z = search.hesse_sic()
        transitions = search.hesse_transitions(states, [x, x.conj().T, z, z.conj().T])
        costs, _, _ = search.solve_sic_hitting_cost(search.sic_probabilities(states), transitions)
        excess = costs - np.diag(costs)[None, :]
        distance = search.torus_manhattan()
        for level in (0, 1, 2):
            values = excess[distance == level]
            self.assertLess(np.ptp(values), 1e-9)
        self.assertGreater(excess[distance == 1][0], 0.0)
        self.assertGreater(excess[distance == 2][0], excess[distance == 1][0])

    def test_null_qubit_has_automaton_but_no_state_geometry(self) -> None:
        states, defect = search.qubit_patch(0.0)
        self.assertTrue(np.allclose(search.pairwise_trace_distance(states), 0.0))
        self.assertEqual(defect, 0.0)
        self.assertGreater(np.max(search.open_manhattan()), 0.0)

    def test_qubit_patch_becomes_noncommutative(self) -> None:
        states, defect = search.qubit_patch(0.4)
        self.assertGreater(np.min(search.upper_values(search.pairwise_trace_distance(states))), 0.0)
        self.assertGreater(defect, 0.0)
        mean_defect, max_defect = search.qubit_translation_defects(0.4)
        self.assertGreater(mean_defect, 0.0)
        self.assertGreaterEqual(max_defect, mean_defect)

    def test_qutrit_phase_grid_is_a_covariant_povm(self) -> None:
        states, u, v = search.qutrit_phase_grid()
        completeness = sum(search.ket_density(state) / 3 for state in states)
        self.assertTrue(np.allclose(completeness, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(u @ v, v @ u, atol=1e-12))
        transitions = search.hesse_transitions(
            states, [u, u.conj().T, v, v.conj().T]
        )
        self.assertTrue(
            np.array_equal(search.shortest_control_distance(transitions), search.torus_manhattan())
        )

    def test_confirmation_sequences_retain_phase_grid_order(self) -> None:
        states, u, v = search.qutrit_phase_grid()
        transitions = search.hesse_transitions(states, [u, u.conj().T, v, v.conj().T])
        costs, _ = search.solve_confirmation_goal(
            search.sic_probabilities(states), transitions, confirmations=2
        )
        excess = costs - np.diag(costs)[None, :]
        correlation = search.pearson(
            search.upper_values(search.torus_manhattan()),
            search.upper_values(0.5 * (excess + excess.T)),
        )
        self.assertGreater(correlation, 0.95)

    def test_quantum_counter_backaction_breaks_coin_additivity(self) -> None:
        coin, qubit = search.counter_surfaces()
        coin_residual = coin - coin[:, :1] - coin[:1, :]
        qubit_residual = qubit - qubit[:, :1] - qubit[:1, :]
        self.assertTrue(np.allclose(coin_residual, 0.0, atol=1e-9))
        self.assertGreater(np.max(np.abs(qubit_residual)), 0.1)


if __name__ == "__main__":
    unittest.main()
