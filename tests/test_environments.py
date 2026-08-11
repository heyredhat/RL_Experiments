"""Physical validity and public-interface tests for every catalog world."""

from __future__ import annotations

import unittest

import numpy as np

from quantum_environments import (
    DEFAULT_GOALS_BY_ENVIRONMENT,
    QuantumEnvironment,
    available_environments,
    environment_definition,
)
from quantum_rl_common import parse_goals, validate_goals


class EnvironmentCatalogTests(unittest.TestCase):
    def test_every_catalog_entry_has_valid_defaults_and_goals(self) -> None:
        self.assertEqual(set(available_environments()), set(DEFAULT_GOALS_BY_ENVIRONMENT))
        for name in available_environments():
            definition = environment_definition(name)
            self.assertIn(definition.default_initial_state, definition.initial_states)
            env = QuantumEnvironment(environment=name, seed=7)
            validate_goals(parse_goals(DEFAULT_GOALS_BY_ENVIRONMENT[name]), env)

    def test_all_instruments_preserve_density_matrix_invariants(self) -> None:
        for name in available_environments():
            env = QuantumEnvironment(environment=name, seed=11)
            for action, outcome_count in enumerate(env.action_outcome_counts):
                env.reset()
                for _ in range(30):
                    outcome = env.step(action)
                    self.assertIn(outcome, range(outcome_count))
                    rho = env._rho  # Privileged access is appropriate only in simulator tests.
                    self.assertIsNotNone(rho)
                    assert rho is not None
                    self.assertTrue(np.allclose(rho, rho.conj().T, atol=1e-9))
                    self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=9)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(rho).min()), -1e-9)

    def test_known_projective_outcome_is_deterministic(self) -> None:
        env = QuantumEnvironment(
            environment="qubit-zx-weak",
            initial_state="one",
            seed=3,
        )
        self.assertEqual([env.step(0) for _ in range(20)], [1] * 20)

    def test_seed_reproduces_outcome_stream(self) -> None:
        first = QuantumEnvironment(environment="qubit-pauli-sic", seed=42)
        second = QuantumEnvironment(environment="qubit-pauli-sic", seed=42)
        actions = [3, 0, 3, 1, 2] * 10
        self.assertEqual(
            [first.step(action) for action in actions],
            [second.step(action) for action in actions],
        )

    def test_spatial_move_reports_destination_without_coordinates(self) -> None:
        env = QuantumEnvironment(environment="qudit-grid-3x3", seed=5)
        # Center is anonymous place symbol 4; north reaches symbol 1.
        self.assertEqual(env.step(0), 1)
        rho = env._rho
        assert rho is not None
        self.assertAlmostEqual(float(rho[1, 1].real), 1.0)

    def test_blind_spatial_boundary_reports_failure(self) -> None:
        env = QuantumEnvironment(environment="qudit-grid-3x3-blind", seed=5)
        self.assertEqual(env.step(0), 0)  # center -> north succeeds
        self.assertEqual(env.step(0), 1)  # open boundary cannot move farther

    def test_weak_beacons_are_qnd_and_overlap(self) -> None:
        definition = environment_definition("qudit-grid-3x3-beacons", weak_q=0.715)
        beacon = definition.measurements[8]
        probabilities = np.diag(
            beacon.outcome_kraus[1][0].conj().T @ beacon.outcome_kraus[1][0]
        ).real
        self.assertTrue(np.allclose(probabilities[:3], (0.05, 0.50, 0.95)))
        self.assertTrue(np.all((probabilities > 0.0) & (probabilities < 1.0)))

        env = QuantumEnvironment(
            environment="qudit-grid-3x3-beacons",
            initial_state="site-0",
            weak_q=0.715,
            seed=13,
        )
        for _ in range(50):
            env.step(8)
            rho = env._rho
            assert rho is not None
            self.assertAlmostEqual(float(rho[0, 0].real), 1.0, places=10)

    def test_null_beacons_are_place_independent(self) -> None:
        definition = environment_definition(
            "qudit-grid-3x3-null-beacons", weak_q=0.715
        )
        for beacon in definition.measurements[8:12]:
            probabilities = np.diag(
                beacon.outcome_kraus[1][0].conj().T
                @ beacon.outcome_kraus[1][0]
            ).real
            self.assertTrue(np.allclose(probabilities, 0.5))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown environment"):
            QuantumEnvironment(environment="not-a-world")
        with self.assertRaisesRegex(ValueError, "weak_q"):
            QuantumEnvironment(weak_q=0.5)
        with self.assertRaisesRegex(ValueError, "initial_state"):
            QuantumEnvironment(environment="qutrit-mub", initial_state="plus-i")
        env = QuantumEnvironment(environment="qutrit-mub")
        with self.assertRaisesRegex(ValueError, "invalid action"):
            env.step(env.n_actions)


if __name__ == "__main__":
    unittest.main()
