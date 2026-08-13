import unittest

import numpy as np

from sic_anchored_square import (
    DISCOUNT_POLICY_THRESHOLD,
    I2,
    LABELS,
    action_values,
    all_bellman_values,
    analytic_discounted_shell_values,
    bellman_values,
    analytic_value_matrix,
    sic_kernel,
    sic_kraus,
    square_distance,
    tetrahedral_states,
)


class SicAnchoredSquareTests(unittest.TestCase):
    def test_sic_is_complete(self):
        completeness = sum(k.conj().T @ k for k in sic_kraus())
        self.assertTrue(np.allclose(completeness, I2, atol=1e-14))

    def test_sic_outcome_prepares_its_own_orbit_state(self):
        states = tetrahedral_states()
        for source in range(4):
            for outcome, kraus in enumerate(sic_kraus()):
                branch = kraus @ states[source] @ kraus.conj().T
                probability = np.trace(branch).real
                self.assertGreater(probability, 0)
                self.assertTrue(np.allclose(branch / probability, states[outcome], atol=1e-14))

    def test_sic_kernel_has_tetrahedral_response(self):
        kernel = sic_kernel()
        self.assertTrue(np.allclose(np.diag(kernel), 0.5))
        self.assertTrue(np.allclose(kernel[~np.eye(4, dtype=bool)], 1 / 6))
        self.assertTrue(np.allclose(kernel.sum(axis=0), 1))

    def test_reported_goal_bellman_values_are_baseline_plus_distance(self):
        values, _ = all_bellman_values()
        self.assertTrue(np.allclose(values, analytic_value_matrix(), atol=2e-12))
        operational = values - np.diag(values)[None, :]
        self.assertTrue(np.allclose(operational, square_distance(), atol=2e-12))

    def test_policy_reports_at_goal_and_moves_elsewhere(self):
        _, policies = all_bellman_values()
        for goal in range(4):
            self.assertEqual(policies[goal, goal], 3)
            for source in range(4):
                if source != goal:
                    self.assertIn(policies[source, goal], (0, 1, 2))

    def test_all_undiscounted_action_values_and_strict_optima(self):
        values, policy = bellman_values(0)
        q = action_values(values, 0)
        c = (8 + np.sqrt(2)) / 3
        expected = np.array(
            [
                [c + 2, c + 2, c + 2, c],
                [c + 1, c + 1 + np.sqrt(2), c + 2, c + 1 + (2 + np.sqrt(2)) / 9],
                [c + 1 + np.sqrt(2), c + 1, c + 2, c + 1 + (2 + np.sqrt(2)) / 9],
                [c + 2, c + 2, c + np.sqrt(2), c + np.sqrt(2) + (8 - 5 * np.sqrt(2)) / 9],
            ]
        )
        self.assertTrue(np.allclose(q, expected, atol=2e-12))
        self.assertTrue(np.array_equal(policy, np.array([3, 0, 1, 2])))

    def test_discounted_closed_forms_match_value_iteration(self):
        for gamma in (0.0, 0.1, 0.5, 0.7, DISCOUNT_POLICY_THRESHOLD, 0.9, 0.99, 1.0):
            shell, _ = analytic_discounted_shell_values(gamma)
            values, _ = bellman_values(0, discount=gamma)
            self.assertTrue(np.allclose(values, shell[[0, 1, 1, 2]], atol=3e-12))

    def test_discounting_changes_policy_and_geometry(self):
        _, low_policy = bellman_values(0, discount=0.25)
        _, middle_policy = bellman_values(0, discount=0.75)
        _, high_policy = bellman_values(0, discount=0.9)
        self.assertTrue(np.array_equal(low_policy, np.array([3, 3, 3, 3])))
        self.assertTrue(np.array_equal(middle_policy, np.array([3, 0, 1, 3])))
        self.assertTrue(np.array_equal(high_policy, np.array([3, 0, 1, 2])))

        shell, _ = analytic_discounted_shell_values(0.5)
        self.assertAlmostEqual(shell[1] - shell[0], shell[2] - shell[0])
        shell, _ = analytic_discounted_shell_values(1.0)
        self.assertAlmostEqual(shell[1] - shell[0], 1.0)
        self.assertAlmostEqual(shell[2] - shell[0], np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
