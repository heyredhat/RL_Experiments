import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from informative_qubit import (  # noqa: E402
    BUTTONS, InformativeQubit, charts_equivalent_under_d4, deterministic_search,
    infer_coordinate_chart, manhattan_goal_matrix, reconstruct_phases,
    reconstruct_state_from_probes, sequence_displacement, trace_distance,
    word_equivalence_audit,
)


class InformativeQubitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        best = deterministic_search()[0]
        cls.model = InformativeQubit(best["alpha"], best["beta"], best["strength"])

    def test_instruments_are_complete_and_genuinely_nonunitary(self):
        for button in BUTTONS:
            kraus = self.model.instrument(button)
            gram = sum((k.conj().T @ k for k in kraus.values()), np.zeros((2,2), complex))
            np.testing.assert_allclose(gram, np.eye(2), atol=1e-12)
            self.assertGreater(np.linalg.norm(kraus[1].conj().T @ kraus[1] - .5*np.eye(2)), .05)

    def test_action_outcomes_are_state_dependent(self):
        reset = self.model.reset_state()
        probabilities = [self.model.apply(reset, button, 1)[0] for button in BUTTONS]
        self.assertGreater(max(probabilities) - min(probabilities), .05)

    def test_exact_common_probe_signatures_recover_chart_without_labels(self):
        signatures = {button: self.model.exact_signature(button) for button in BUTTONS}
        phases = reconstruct_phases(signatures, self.model.coherence_retention)
        chart = infer_coordinate_chart(phases)
        self.assertTrue(charts_equivalent_under_d4(chart, self.model.hidden_coordinates))

    def test_predictive_tomography_reconstructs_exact_state(self):
        state = self.model.reset_state()
        for button, outcome in zip(("amber","crimson","blue"), (1,-1,1)):
            _, state = self.model.apply(state, button, outcome)
        probabilities = {axis: self.model.probe_probability(state, axis) for axis in ("X","Y","Z")}
        reconstructed = reconstruct_state_from_probes(probabilities)
        self.assertLess(trace_distance(state, reconstructed), 1e-12)

    def test_sequence_goals_have_exact_square_chart(self):
        chart = self.model.hidden_coordinates
        self.assertEqual(sequence_displacement(("amber","crimson"), chart), (1,1))
        matrix = manhattan_goal_matrix()
        self.assertEqual(matrix.shape, (9,9))
        self.assertEqual(matrix[0,-1], 4)

    def test_search_is_deterministic_and_has_positive_margins(self):
        first = deterministic_search()[0]
        second = deterministic_search()[0]
        self.assertEqual(first, second)
        self.assertGreater(first["goal_separation"], .1)
        self.assertGreater(first["information_margin"], .1)
        self.assertGreater(first["alias_margin_h8"], .01)

    def test_nonselective_word_goals_fail_predictive_path_independence(self):
        audit = word_equivalence_audit(self.model)
        repeated = [row for row in audit if row["path_count"] > 1]
        self.assertTrue(repeated)
        self.assertTrue(any(not row["equivalent_at_1e-10"] for row in repeated))


if __name__ == "__main__":
    unittest.main()
