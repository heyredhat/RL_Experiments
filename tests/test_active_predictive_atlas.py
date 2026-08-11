"""Tests for equal-cost active sensing and reversible spatial motion."""

from __future__ import annotations

import unittest

import numpy as np

from active_predictive_atlas import (
    _restricted_matrix,
    choose_policy_partition_action,
    exact_beacon_model,
    exact_joint_model,
    policy_partition,
    update_beacon_belief,
)
from predictive_atlas import planning_values
from quantum_environments import environment_definition
from spatial_hodology import exact_movement_costs


ENVIRONMENT = "qudit-grid-3x3-reversible-beacons"


class ActivePredictiveAtlasTests(unittest.TestCase):
    def test_random_unitary_moves_preserve_uniform_uncertainty(self) -> None:
        joint = exact_joint_model(ENVIRONMENT)
        transitions = joint.sum(axis=1)
        uniform = np.full(9, 1 / 9)
        for action in range(8):
            self.assertTrue(np.allclose(uniform @ transitions[action], uniform))
            self.assertTrue(np.allclose(transitions[action].sum(axis=0), 1.0))

    def test_reversible_move_kraus_operators_are_random_unitaries(self) -> None:
        definition = environment_definition(ENVIRONMENT, weak_q=0.715)
        for action, measurement in enumerate(definition.measurements[:8]):
            success = measurement.outcome_kraus[0][0]
            gram = success.conj().T @ success
            expected = 1.0 if action < 4 else 0.715
            self.assertTrue(np.allclose(gram, expected * np.eye(9)))

    def test_reversible_cost_geometry_matches_optimized_grid(self) -> None:
        values, _ = planning_values(exact_joint_model(ENVIRONMENT))
        self.assertTrue(np.allclose(values, exact_movement_costs(0.715), atol=1e-7))

    def test_beacon_update_uses_learned_operational_likelihood(self) -> None:
        likelihoods = exact_beacon_model(ENVIRONMENT)
        posterior = update_beacon_belief(
            np.full(9, 1 / 9), likelihoods, beacon_index=0, outcome=1
        )
        self.assertGreater(posterior[2], posterior[0])
        self.assertAlmostEqual(float(posterior.sum()), 1.0)

    def test_policy_partition_is_goal_relative(self) -> None:
        joint = exact_joint_model(ENVIRONMENT)
        _, q_values = planning_values(joint)
        _, corner_masses = policy_partition(np.full(9, 1 / 9), 0, q_values)
        _, center_masses = policy_partition(np.full(9, 1 / 9), 4, q_values)
        self.assertFalse(np.allclose(corner_masses, center_masses))
        self.assertAlmostEqual(float(corner_masses.sum()), 1.0)

    def test_active_rule_senses_when_needed_and_commits_when_certain(self) -> None:
        joint = exact_joint_model(ENVIRONMENT)
        _, q_values = planning_values(joint)
        likelihoods = exact_beacon_model(ENVIRONMENT)
        kind, _, _ = choose_policy_partition_action(
            np.full(9, 1 / 9),
            0,
            likelihoods,
            q_values,
            decision_error_penalty=100.0,
            sensing_lookahead=3,
        )
        self.assertEqual(kind, "sense")
        kind, _, _ = choose_policy_partition_action(
            np.eye(9)[0],
            0,
            likelihoods,
            q_values,
            decision_error_penalty=100.0,
            sensing_lookahead=3,
        )
        self.assertEqual(kind, "commit")

    def test_sensing_burden_can_be_aggregated_without_failure_censoring(self) -> None:
        records = [
            {"source": source, "goal": goal, "success": True, "senses": 2}
            for source in range(9)
            for goal in range(9)
        ]
        records.append({"source": 0, "goal": 1, "success": False, "senses": 4})
        actual, success = _restricted_matrix(
            records, "senses", 60.0, censor_failures=False
        )
        censored, _ = _restricted_matrix(records, "senses", 60.0)
        self.assertAlmostEqual(actual[0, 1], 3.0)
        self.assertAlmostEqual(censored[0, 1], 31.0)
        self.assertAlmostEqual(success[0, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
