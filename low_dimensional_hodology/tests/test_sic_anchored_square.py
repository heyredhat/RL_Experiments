import unittest

import numpy as np

from sic_anchored_square import (
    I2,
    LABELS,
    all_bellman_values,
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


if __name__ == "__main__":
    unittest.main()
