import sys
import unittest
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covariant_memory import (  # noqa: E402
    MemoryInstrument, analytic_distance, analytic_shell_values,
    branch_rank_diagnostics, covariance_residual, exhaustive_bellman_verification,
    generate_observable_action_probe_records, geodesic_path_closure,
    kraus_residual, learn_action_maps_from_observable_probes,
    metric_diagnostics, observable_action_probe_audit, schoenberg,
    state_hitting_bellman, torus_diagnostics,
)


class CovariantMemoryTests(unittest.TestCase):
    def test_corrected_hesse_state_hitting_baseline_is_zero_four_five(self):
        distance, _ = state_hitting_bellman(MemoryInstrument(0.0, 1.0))
        self.assertEqual(set(np.round(distance, 10).flat), {0.0, 4.0, 5.0})
        self.assertTrue(metric_diagnostics(distance)["is_metric"])

    def test_closed_form_matches_full_bellman_solver(self):
        for model in (MemoryInstrument(.2,1), MemoryInstrument(.8,1), MemoryInstrument(.8,.5)):
            numerical, _ = state_hitting_bellman(model)
            np.testing.assert_allclose(numerical, analytic_distance(model), atol=2e-12)

    def test_every_searched_nontrivial_candidate_has_strict_shells(self):
        for memory in (0.0,.2,.5,.8,.95,1.0):
            edge, diagonal = analytic_shell_values(MemoryInstrument(memory,.4))
            self.assertGreater(edge, 0.0); self.assertGreater(diagonal, edge)

    def test_exact_local_square_ratio(self):
        memory = (4*sqrt(2)-5)/3
        edge, diagonal = analytic_shell_values(MemoryInstrument(memory,1))
        self.assertAlmostEqual(diagonal/edge, sqrt(2), places=12)
        distance = analytic_distance(MemoryInstrument(memory,1))
        cell = distance[np.ix_((0,1,3,4), (0,1,3,4))]
        diagnostic = schoenberg(cell)
        self.assertEqual(diagnostic["positive_dimension"], 2)
        self.assertLess(diagnostic["mds_2d_stress"], 1e-12)

    def test_kraus_completeness_and_covariance(self):
        model = MemoryInstrument(.8,.7)
        self.assertLess(kraus_residual(model), 1e-12)
        self.assertLess(covariance_residual(model), 1e-12)

    def test_equal_length_paths_close_but_detours_retain_noise_age(self):
        result = geodesic_path_closure(MemoryInstrument(.8,1))
        self.assertLess(result["equal_length_path_closure_residual"], 1e-12)
        self.assertGreater(result["all_length_path_closure_residual"], .1)

    def test_memory_only_is_exact_torus_word_metric_but_uninformative(self):
        model = MemoryInstrument(1,1)
        distance = analytic_distance(model)
        self.assertAlmostEqual(torus_diagnostics(distance)["relative_rmse_after_scale"], 0.0)
        self.assertAlmostEqual(model.immediate_information_bits(), 0.0)

    def test_observable_probe_recovers_action_group_without_successor_labels(self):
        model = MemoryInstrument(.8, 1)
        records, private = generate_observable_action_probe_records(model, 500, 20260812)
        self.assertTrue(records)
        self.assertTrue(all(len(record) == 3 for record in records))
        self.assertEqual(
            "opaque_anchor_token,opaque_action_token,opaque_future_probe_token",
            observable_action_probe_audit(model, 500, 20260812)[0]["learner_fields"],
        )
        inferred, laws = learn_action_maps_from_observable_probes(records)
        self.assertEqual(inferred.shape, (5, 9))
        self.assertEqual(laws.shape, (5, 9, 9))
        summary, _ = observable_action_probe_audit(model, 500, 20260812)
        self.assertEqual(summary["offline_mean_map_accuracy"], 1.0)
        self.assertTrue(summary["valid_permutation_action_set"])
        self.assertTrue(summary["commuting"])
        self.assertEqual(summary["learned_group_order"], 9)
        self.assertEqual(summary["orbit_size"], 9)
        self.assertNotIn("next_state", summary["learner_fields"])
        self.assertEqual(private["accepted_memory_events"], len(records))

    def test_branch_choi_rank_and_operator_rank_are_not_conflated(self):
        selected = branch_rank_diagnostics(MemoryInstrument(.8, 1))
        self.assertEqual(selected["memory_branch_choi_rank"], 1)
        self.assertEqual(selected["memory_branch_maximum_operator_rank"], 3)
        self.assertEqual(selected["reset_branch_minimum_choi_rank"], 1)
        self.assertEqual(selected["reset_branch_maximum_choi_rank"], 1)
        weak = branch_rank_diagnostics(MemoryInstrument(.8, .5))
        self.assertEqual(weak["reset_branch_minimum_choi_rank"], 3)
        self.assertEqual(weak["reset_branch_maximum_choi_rank"], 3)
        report_only = branch_rank_diagnostics(MemoryInstrument(0, 1))
        self.assertEqual(report_only["memory_branch_choi_rank"], 0)
        memory_only = branch_rank_diagnostics(MemoryInstrument(1, 1))
        self.assertEqual(memory_only["reset_branch_maximum_choi_rank"], 0)

    def test_exhaustive_grid_closed_form_matches_full_bellman(self):
        rows = exhaustive_bellman_verification()
        self.assertEqual(len(rows), 380)
        self.assertLess(max(row["analytic_bellman_max_error"] for row in rows), 1e-10)
        self.assertTrue(all(row["is_metric"] for row in rows))


if __name__ == "__main__":
    unittest.main()
