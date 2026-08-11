"""Small deterministic smoke/regression tests for each learning backend."""

from __future__ import annotations

import math
import unittest

import numpy as np

from quantum_environments import QuantumEnvironment
from quantum_rl_common import GoalSpec, run_episode
from q_learning import TabularQAgent

try:
    import torch

    from gru_q_learning import GRUQAgent
    from multi_goal_gru import MultiGoalGRUAgent
    from predictive_gru_q_learning import PredictiveGRUQAgent
except ModuleNotFoundError:  # The base shell may intentionally omit PyTorch.
    torch = None


class TabularAgentTests(unittest.TestCase):
    def test_update_and_epsilon_decay(self) -> None:
        env = QuantumEnvironment(initial_state="one", seed=0)
        goal = GoalSpec("Z1", ((0, 1),))
        agent = TabularQAgent(
            n_actions=env.n_actions,
            epsilon_start=0.0,
            epsilon_end=0.0,
            alpha=0.5,
            seed=0,
        )
        agent.Q[(0, 0, ())][0] = 1.0
        before_value = float(agent.Q[(0, 0, ())][0])
        result = run_episode(env, goal, 0, agent, training=True, max_steps=3)
        self.assertTrue(result.success)
        self.assertGreater(float(agent.Q[(0, 0, ())][0]), before_value)
        self.assertGreater(max(values.max() for values in agent.Q.values()), 0.0)


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class RecurrentAgentTests(unittest.TestCase):
    goals = (GoalSpec("Z1", ((0, 1),)), GoalSpec("X0", ((1, 0),)))

    def setUp(self) -> None:
        self.env = QuantumEnvironment(initial_state="one", seed=2)

    def test_plain_gru_trains_one_episode(self) -> None:
        agent = GRUQAgent(
            n_actions=self.env.n_actions,
            n_outcomes=self.env.n_outcomes,
            goals=self.goals,
            hidden_dim=8,
            goal_dim=4,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=1,
            device="cpu",
        )
        run_episode(self.env, self.goals[0], 0, agent, training=True, max_steps=3)
        self.assertTrue(math.isfinite(agent.last_loss))

    def test_predictive_gru_probabilities_are_normalized(self) -> None:
        agent = PredictiveGRUQAgent(
            n_actions=self.env.n_actions,
            n_outcomes=self.env.n_outcomes,
            action_outcome_counts=self.env.action_outcome_counts,
            max_goal_length=1,
            hidden_dim=8,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=1,
            device="cpu",
        )
        run_episode(self.env, self.goals[0], 0, agent, training=True, max_steps=3)
        agent.reset_episode(0, 1, False)
        self.assertAlmostEqual(float(agent.predict_outcomes(0).sum()), 1.0, places=6)
        self.assertTrue(math.isfinite(agent.last_total_loss))

    def test_mixed_outcome_head_masks_impossible_results(self) -> None:
        env = QuantumEnvironment(environment="qubit-pauli-sic", seed=0)
        goals = (GoalSpec("Z0", ((0, 0),)), GoalSpec("SIC3", ((3, 3),)))
        agent = MultiGoalGRUAgent(
            n_actions=env.n_actions,
            n_outcomes=env.n_outcomes,
            action_outcome_counts=env.action_outcome_counts,
            goals=goals,
            hidden_dim=8,
            goal_dim=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=1,
            device="cpu",
        )
        agent.reset_episode(0, 1, False)
        projective = agent.predict_outcomes(0)
        sic = agent.predict_outcomes(3)
        self.assertTrue(np.allclose(projective[2:], 0.0))
        self.assertAlmostEqual(float(projective.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(sic.sum()), 1.0, places=6)

    def test_multi_goal_gru_trains_and_reports_geometry(self) -> None:
        agent = MultiGoalGRUAgent(
            n_actions=self.env.n_actions,
            n_outcomes=self.env.n_outcomes,
            action_outcome_counts=self.env.action_outcome_counts,
            goals=self.goals,
            hidden_dim=8,
            goal_dim=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=1,
            device="cpu",
        )
        run_episode(self.env, self.goals[0], 0, agent, training=True, max_steps=3)
        self.assertTrue(math.isfinite(agent.last_total_loss))
        distances = agent.goal_embedding_distance_matrix()
        self.assertEqual(distances.shape, (2, 2))
        self.assertTrue(np.allclose(distances, distances.T))
        self.assertTrue(np.allclose(np.diag(distances), 0.0))


if __name__ == "__main__":
    unittest.main()
