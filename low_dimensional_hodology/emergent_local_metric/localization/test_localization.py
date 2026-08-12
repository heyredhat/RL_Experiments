"""Tests for qutrit weak-measurement localization."""

from __future__ import annotations

import unittest

import numpy as np

import localization_experiment as loc


class TestLocalization(unittest.TestCase):
    def test_phase_frame_and_weak_povm_are_complete(self) -> None:
        kets, _, _ = loc.phase_grid()
        self.assertTrue(np.allclose(sum(loc.density(ket) / 3 for ket in kets), np.eye(3)))
        for eta in (0.0, 0.2, 0.8, 1.0):
            effects, kraus = loc.weak_instrument(eta)
            self.assertTrue(np.allclose(effects.sum(axis=0), np.eye(3), atol=1e-12))
            self.assertTrue(
                np.allclose(sum(operator.conj().T @ operator for operator in kraus), np.eye(3), atol=1e-12)
            )

    def test_null_observation_changes_neither_weights_nor_states(self) -> None:
        effects, kraus = loc.weak_instrument(0.0)
        belief = loc.QuantumLabelFilter.uniform()
        before_weights = belief.weights.copy()
        before_states = belief.branch_states.copy()
        belief.observe(4, effects, kraus)
        self.assertTrue(np.allclose(belief.weights, before_weights))
        self.assertTrue(np.allclose(belief.branch_states, before_states))

    def test_covariance_of_outcome_probabilities(self) -> None:
        kets, _, _ = loc.phase_grid()
        effects, _ = loc.weak_instrument(0.63)
        for source in range(9):
            moved = loc.translate_index(source, 1, -1)
            unitary = loc.move_unitary(1, -1)
            moved_state = unitary @ loc.density(kets[source]) @ unitary.conj().T
            self.assertTrue(np.allclose(moved_state, loc.density(kets[moved]), atol=1e-12))
            original = np.array([loc.born_probability(loc.density(kets[source]), effect) for effect in effects])
            translated = np.array([loc.born_probability(moved_state, effect) for effect in effects])
            expected = np.zeros(9)
            for outcome in range(9):
                expected[loc.translate_index(outcome, 1, -1)] = original[outcome]
            self.assertTrue(np.allclose(translated, expected, atol=1e-12))

    def test_one_shot_formula(self) -> None:
        for eta in (0.0, 0.2, 0.5, 1.0):
            label, operational = loc.analytic_one_shot(eta)
            self.assertAlmostEqual(label, (1 + 2 * eta) / 9)
            self.assertAlmostEqual(operational, (1 + 2 * eta) / 3)

    def test_stable_argmax_breaks_numerical_ties_by_label(self) -> None:
        self.assertEqual(loc.stable_argmax(np.array([1.0, 1.0 + 2e-14, 0.5])), 0)

    def test_known_start_navigation_is_exact(self) -> None:
        rng = np.random.default_rng(2)
        strategy = loc.Strategy("known", fixed_senses=0)
        for _ in range(40):
            record = loc.run_episode(0.5, strategy, rng, known_start=True)
            self.assertEqual(record["label_success"], 1)
            self.assertAlmostEqual(float(record["operational_score"]), 1.0, places=12)
            self.assertEqual(record["state_label_success"], 1)
            self.assertAlmostEqual(float(record["state_operational_score"]), 1.0, places=12)

    def test_filter_remains_normalized_and_physical(self) -> None:
        rng = np.random.default_rng(9)
        effects, kraus = loc.weak_instrument(0.7)
        belief = loc.QuantumLabelFilter.uniform()
        actual = loc.density(loc.phase_grid()[0][5])
        for _ in range(8):
            outcome = loc.sample_outcome(actual, effects, rng)
            actual = loc.normalize_density(kraus[outcome] @ actual @ kraus[outcome])
            belief.observe(outcome, effects, kraus)
            self.assertAlmostEqual(float(belief.weights.sum()), 1.0)
            self.assertTrue(np.all(belief.weights >= 0.0))
            for state in belief.branch_states:
                self.assertAlmostEqual(float(np.real(np.trace(state))), 1.0)
                self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(state))), -1e-12)

    def test_seeded_episode_is_reproducible(self) -> None:
        strategy = loc.Strategy("fixed", fixed_senses=3)
        first = loc.run_episode(0.8, strategy, np.random.default_rng(123))
        second = loc.run_episode(0.8, strategy, np.random.default_rng(123))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
