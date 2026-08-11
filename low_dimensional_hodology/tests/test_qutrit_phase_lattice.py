"""Tests for the exact qutrit phase-manifold construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qutrit_phase_lattice import (  # noqa: E402
    QutritPhaseLattice,
    bellman_residual,
    distance_matrix,
    schoenberg_gram,
)


class QutritPhaseLatticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = QutritPhaseLattice(order=11)

    def test_generators_form_a_faithful_finite_phase_torus(self) -> None:
        u, v = self.model.unitaries
        np.testing.assert_allclose(u @ v, v @ u, atol=1e-12)
        np.testing.assert_allclose(np.linalg.matrix_power(u, 11), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.matrix_power(v, 11), np.eye(3), atol=1e-12)
        states = [self.model.density(coordinate) for coordinate in self.model.coordinates()]
        rounded = {tuple(np.round(state.flatten(), 10)) for state in states}
        self.assertEqual(len(rounded), 121)

    def test_phase_metric_is_exactly_isotropic(self) -> None:
        np.testing.assert_allclose(
            self.model.fubini_study_metric(), (3 / 16) * np.eye(2), atol=1e-15
        )

    def test_retry_kraus_operators_are_complete_and_move_exactly(self) -> None:
        source = (1, 7)
        for displacement in ((1, 0), (1, 1), (-2, 1), (2, -2)):
            success, failure = self.model.retry_kraus(displacement)
            np.testing.assert_allclose(
                success.conj().T @ success + failure.conj().T @ failure,
                np.eye(3),
                atol=1e-12,
            )
            branch = success @ self.model.density(source) @ success.conj().T
            branch /= np.trace(branch)
            target = (
                (source[0] + displacement[0]) % 11,
                (source[1] + displacement[1]) % 11,
            )
            np.testing.assert_allclose(branch, self.model.density(target), atol=1e-12)

    def test_open_patch_has_exact_euclidean_distance_and_rank_two(self) -> None:
        patch = self.model.patch_coordinates()
        matrix = distance_matrix(self.model, patch)
        coordinates = np.array(patch, dtype=float)
        exact = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        np.testing.assert_allclose(matrix, exact, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(schoenberg_gram(matrix))
        self.assertEqual(int(np.sum(eigenvalues > 1e-10)), 2)
        self.assertGreaterEqual(float(np.min(eigenvalues)), -1e-10)

    def test_analytic_distance_satisfies_every_bellman_equation(self) -> None:
        self.assertLess(bellman_residual(self.model, (0, 0)), 1e-12)


if __name__ == "__main__":
    unittest.main()

