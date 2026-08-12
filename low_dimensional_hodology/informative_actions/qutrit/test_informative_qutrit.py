"""Tests for operationally informative qutrit actions."""

from __future__ import annotations

import unittest

import numpy as np

import informative_qutrit as model


class TestInformativeQutrit(unittest.TestCase):
    def test_axis_instruments_are_complete_and_have_exact_likelihoods(self) -> None:
        eta = 0.6
        effects, kraus = model.axis_instruments(eta)
        for action in range(2):
            self.assertTrue(np.allclose(effects[action].sum(axis=0), np.eye(3)))
            self.assertTrue(np.allclose(sum(k.conj().T @ k for k in kraus[action]), np.eye(3)))
        table = model.likelihoods(effects)
        matched, unmatched = (3 + 2 * eta) / 9, (3 - eta) / 9
        for state, (x, y) in enumerate(model.COORDS):
            self.assertAlmostEqual(table[state, 0, x], matched)
            self.assertAlmostEqual(table[state, 1, y], matched)
            self.assertTrue(all(np.isclose(table[state, 0, o], unmatched) for o in range(3) if o != x))

    def test_two_axis_signatures_are_injective_but_null_is_not(self) -> None:
        informative = model.likelihoods(model.axis_instruments(0.3)[0]).reshape(9, -1)
        null = model.likelihoods(model.axis_instruments(0.0)[0]).reshape(9, -1)
        self.assertEqual(len(np.unique(np.round(informative, 12), axis=0)), 9)
        self.assertEqual(len(np.unique(np.round(null, 12), axis=0)), 1)

    def test_hesse_measure_prepare_kernel_is_exact(self) -> None:
        kernel = model.hesse_kernel()
        self.assertTrue(np.allclose(kernel.sum(axis=1), 1))
        self.assertTrue(np.allclose(np.diag(kernel), 1 / 3))
        self.assertTrue(np.allclose(kernel[~np.eye(9, dtype=bool)], 1 / 12))
        self.assertEqual(len(np.unique(np.round(kernel, 12), axis=0)), 9)

    def test_integrated_kraus_operators_are_complete_and_prepare_outcome(self) -> None:
        states, effects, x, _ = model.hesse_system()
        operators = [model.rho(state) @ x / np.sqrt(3) for state in states]
        self.assertTrue(np.allclose(sum(k.conj().T @ k for k in operators), np.eye(3)))
        source = model.rho(states[4])
        for outcome, operator in enumerate(operators):
            branch = operator @ source @ operator.conj().T
            if np.trace(branch).real > 1e-14:
                branch /= np.trace(branch)
                self.assertTrue(np.allclose(branch, model.rho(states[outcome]), atol=1e-12))

    def test_opaque_outcome_kernels_recover_z3_square(self) -> None:
        rows, summary = model.opaque_hesse_rows(seed=7, trials=3000)
        self.assertTrue(all(row["permutation_accuracy"] == 1 for row in rows))
        self.assertEqual(summary["learned_group_order"], 9)
        self.assertEqual(summary["orbit_size_from_token_zero"], 9)
        self.assertTrue(summary["all_generators_commute"])
        self.assertEqual(summary["identity_action_inferred_by_order_one"], 1)

    def test_integrated_bellman_has_exact_plateau_and_diagonal(self) -> None:
        costs, _ = model.integrated_hesse_bellman()
        _, _, x, z = model.hesse_system()
        permutations = [model.infer_kernel_permutation(model.hesse_kernel(u)) for u in (x, x.conj().T, z, z.conj().T)]
        distance = model.graph_distance(permutations)
        self.assertTrue(np.allclose(costs[distance <= 1], 4.0, atol=1e-10))
        self.assertTrue(np.allclose(costs[distance == 2], 5.0, atol=1e-10))

    def test_weak_integrated_family_always_collapses_self_and_edge(self) -> None:
        _, _, x, z = model.hesse_system()
        permutations = [model.infer_kernel_permutation(model.hesse_kernel(u)) for u in (x, x.conj().T, z, z.conj().T)]
        distance = model.graph_distance(permutations)
        for eta in (0.0, 0.2, 0.6, 1.0):
            costs = model.integrated_weak_bellman(eta)
            self.assertAlmostEqual(
                float(np.mean(np.diag(costs))),
                float(np.mean(costs[distance == 1])),
                places=10,
            )

    def test_separated_bellman_recovers_word_distance_exactly(self) -> None:
        costs, _ = model.bellman_hesse()
        _, _, x, z = model.hesse_system()
        permutations = [model.infer_kernel_permutation(model.hesse_kernel(u)) for u in (x, x.conj().T, z, z.conj().T)]
        distance = model.graph_distance(permutations)
        excess = costs - np.diag(costs)[None, :]
        self.assertTrue(np.allclose(excess, distance, atol=1e-10))

    def test_action_and_outcome_permutations_are_gauge(self) -> None:
        controls, meanings, _ = model.hidden_controls(seed=41)
        effects = model.axis_instruments(0.6)[0]
        first = model.infer_action_meanings(controls, effects)
        order = (2, 0, 3, 1)
        second = model.infer_action_meanings([controls[i] for i in order], effects)
        expected = [(first[i]["inferred_dx"], first[i]["inferred_dy"]) for i in order]
        observed = [(row["inferred_dx"], row["inferred_dy"]) for row in second]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
