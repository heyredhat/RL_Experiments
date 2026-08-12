import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_control import (  # noqa: E402
    LocalQutritModel,
    bellman_distances,
    bellman_residual,
    costs_from_parameters,
    euclidean,
    grid_displacements,
    kraus_completeness_residual,
    optimize_class_costs,
    repertoire,
    schoenberg_diagnostics,
    translation_covariance_residual,
)


class LocalControlTests(unittest.TestCase):
    def setUp(self):
        self.model = LocalQutritModel()

    def test_repertoire_sizes_and_locality(self):
        self.assertEqual([len(repertoire(name)) for name in ("D4", "D8", "D16", "D32")], [4, 8, 16, 32])
        self.assertLessEqual(max(euclidean(a) for a in repertoire("D32")), np.sqrt(13))

    def test_qutrit_translation_covariance(self):
        residual = translation_covariance_residual(self.model, ((0, 0), (2, -3)), repertoire("D16"))
        self.assertLess(residual, 1e-13)

    def test_random_unitary_instruments_are_complete(self):
        actions = repertoire("D16")
        costs = {action: max(1.0, euclidean(action)) for action in actions}
        self.assertLess(kraus_completeness_residual(self.model, actions, costs), 1e-13)

    def test_d4_bellman_solution_is_manhattan(self):
        actions = repertoire("D4")
        costs = {action: 1.0 for action in actions}
        distances = bellman_distances(actions, costs, 8)
        for point in grid_displacements(8):
            self.assertAlmostEqual(distances[point], abs(point[0]) + abs(point[1]), places=12)
        self.assertLess(bellman_residual(distances, actions, costs, 8), 1e-13)

    def test_optimizer_reduces_training_loss(self):
        actions = repertoire("D16")
        parameters, history = optimize_class_costs(actions, grid_displacements(4), euclidean, iterations=18)
        self.assertLessEqual(history[-1], history[0])
        self.assertTrue(all(value >= 1.0 for value in parameters.values()))
        costs = costs_from_parameters(actions, parameters)
        self.assertEqual(set(costs), set(actions))

    def test_schoenberg_recognizes_planar_euclidean_distances(self):
        points = np.array([(x, y) for x in range(3) for y in range(3)], dtype=float)
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        diagnostic = schoenberg_diagnostics(distance)
        self.assertLess(diagnostic["negative_eigenmass_fraction"], 1e-12)
        self.assertEqual(diagnostic["positive_dimension"], 2)
        self.assertLess(diagnostic["mds_2d_stress"], 1e-12)

    def test_fubini_study_metric_is_locally_isotropic(self):
        scale = 1e-4
        axial = self.model.scaled_fubini_study_displacement((scale, 0.0))
        diagonal = self.model.scaled_fubini_study_displacement((scale/np.sqrt(2), scale/np.sqrt(2)))
        self.assertAlmostEqual(axial, scale, delta=2e-7)
        self.assertAlmostEqual(diagonal, scale, delta=2e-7)


if __name__ == "__main__":
    unittest.main()
