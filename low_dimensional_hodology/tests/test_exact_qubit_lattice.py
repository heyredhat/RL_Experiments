import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact_qubit_lattice import (  # noqa: E402
    PhaseLattice,
    apply_channel,
    bounded_sequence_goal_accepts,
    canonical_word,
    dephasing_kraus,
    distance_matrix,
    exact_euclidean_macro_cost,
    exact_word_distance,
    euclidean_success_probability,
    euclidean_waiting_instrument,
    goals_3x3,
    kraus_gram,
    macro_kraus,
    noisy_goal_fidelity,
    qutrit_torus_distance,
    qutrit_weyl_operators,
)


class ExactQubitLatticeTests(unittest.TestCase):
    def setUp(self):
        self.model = PhaseLattice()

    def test_all_action_and_verifier_kraus_operators_are_complete(self):
        identity = np.eye(2)
        for kraus in self.model.actions.values():
            np.testing.assert_allclose(kraus_gram(kraus), identity, atol=1e-12)
        for goal in goals_3x3():
            np.testing.assert_allclose(kraus_gram(tuple(self.model.verifier(goal).values())), identity, atol=1e-12)

    def test_every_canonical_word_reaches_its_goal_exactly(self):
        for source in goals_3x3():
            for target in goals_3x3():
                reached = self.model.apply_word(self.model.density(source), canonical_word(source, target))
                np.testing.assert_allclose(reached, self.model.density(target), atol=2e-14)

    def test_word_metric_is_the_open_grid_manhattan_metric(self):
        matrix = distance_matrix(exact_word_distance)
        self.assertEqual(matrix.shape, (9, 9))
        self.assertEqual(matrix[0, 8], 4)
        self.assertEqual(matrix[2, 6], 4)
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_shortest_words_are_members_of_bounded_regular_goals(self):
        for source in goals_3x3():
            for target in goals_3x3():
                word = canonical_word(source, target)
                self.assertTrue(bounded_sequence_goal_accepts(word, source, target))
                self.assertEqual(len(word), exact_word_distance(source, target))
        self.assertFalse(bounded_sequence_goal_accepts(("E",) * 5, (0, 0), (2, 0)))

    def test_norm_priced_macro_actions_give_exact_euclidean_metric(self):
        for source in goals_3x3():
            for target in goals_3x3():
                delta = (target[0] - source[0], target[1] - source[1])
                self.assertAlmostEqual(exact_euclidean_macro_cost(source, target), np.hypot(*delta))
                np.testing.assert_allclose(kraus_gram(macro_kraus(self.model, delta)), np.eye(2), atol=1e-12)

    def test_unit_cost_waiting_instruments_give_euclidean_expected_cost(self):
        for delta in ((1, 0), (1, 1), (2, 1), (2, 2), (-2, 1)):
            instrument = euclidean_waiting_instrument(self.model, delta)
            np.testing.assert_allclose(kraus_gram(tuple(instrument.values())), np.eye(2), atol=1e-12)
            probability = euclidean_success_probability(delta)
            self.assertAlmostEqual(1.0 / probability, np.hypot(*delta), places=12)
            failure = instrument["failure"]
            state = self.model.density((0, 0))
            if probability < 1.0:
                conditional_failure = failure @ state @ failure.conj().T / (1.0 - probability)
                np.testing.assert_allclose(conditional_failure, state, atol=1e-12)
            else:
                np.testing.assert_allclose(failure, np.zeros((2, 2)), atol=1e-12)

    def test_qutrit_weyl_candidate_wraps_to_a_torus(self):
        x, z = qutrit_weyl_operators()
        np.testing.assert_allclose(np.linalg.matrix_power(x, 3), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.matrix_power(z, 3), np.eye(3), atol=1e-12)
        self.assertEqual(exact_word_distance((0, 0), (2, 0)), 2)
        self.assertEqual(qutrit_torus_distance((0, 0), (2, 0)), 1)

    def test_dephasing_closed_form(self):
        eta = 0.87
        state = self.model.density((0, 0))
        channel = dephasing_kraus(eta)
        for length in range(5):
            if length:
                state = apply_channel(state, channel)
            acceptance = float(np.real(np.trace(state @ self.model.density((0, 0)))))
            self.assertAlmostEqual(acceptance, noisy_goal_fidelity(length, eta), places=12)


if __name__ == "__main__":
    unittest.main()
