"""Unit tests for goal semantics, summaries, and the backend boundary."""

from __future__ import annotations

import unittest

from quantum_environments import QuantumEnvironment
from quantum_rl_common import (
    GoalSpec,
    GoalTracker,
    parse_goals,
    parse_target,
    run_episode,
    summarize_by_goal,
    summarize_results,
    validate_goals,
)


class FixedActionAgent:
    """Tiny protocol implementation used to test the shared runner."""

    name = "fixed-action"

    def __init__(self, action: int):
        self.action = action
        self.observations: list[tuple[int, int, float, bool]] = []
        self.ended = False

    def reset_episode(self, goal_id: int, goal_length: int, training: bool) -> None:
        self.observations.clear()
        self.ended = False

    def act(self, goal_id: int, progress: int, training: bool) -> int:
        return self.action

    def observe(
        self,
        goal_id: int,
        progress: int,
        action: int,
        outcome: int,
        reward: float,
        done: bool,
        next_progress: int,
        training: bool,
    ) -> None:
        self.observations.append((action, outcome, reward, done))

    def end_episode(self, training: bool) -> None:
        self.ended = True


class GoalTests(unittest.TestCase):
    def test_parser_preserves_names_and_sequences(self) -> None:
        goals = parse_goals("prepare=0:1,1:0;2:1")
        self.assertEqual(goals[0], GoalSpec("prepare", ((0, 1), (1, 0))))
        self.assertEqual(goals[1].name, "goal1")
        self.assertEqual(parse_target(" 0:0, 2:1 "), ((0, 0), (2, 1)))

    def test_parser_reports_malformed_checkpoints(self) -> None:
        for text in ("", "0", "0:1:2", "-1:0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_target(text)

    def test_tracker_allows_intervening_exploration(self) -> None:
        tracker = GoalTracker(GoalSpec("sequence", ((0, 0), (1, 1))))
        self.assertEqual(tracker.update(2, 0), (-0.01, False))
        self.assertEqual(tracker.update(0, 0), (1.0, False))
        self.assertEqual(tracker.progress, 1)
        self.assertEqual(tracker.update(0, 1), (-0.01, False))
        self.assertEqual(tracker.update(1, 1), (10.0, True))

    def test_goal_validation_uses_action_specific_outcomes(self) -> None:
        env = QuantumEnvironment(environment="qubit-pauli-sic")
        validate_goals((GoalSpec("sic3", ((3, 3),)),), env)
        with self.assertRaisesRegex(ValueError, "unavailable outcome"):
            validate_goals((GoalSpec("bad", ((0, 3),)),), env)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_goals((GoalSpec("same", ((0, 0),)), GoalSpec("same", ((1, 0),))), env)

    def test_runner_completes_deterministic_goal(self) -> None:
        env = QuantumEnvironment(initial_state="one", seed=0)
        agent = FixedActionAgent(0)
        result = run_episode(
            env,
            GoalSpec("Z1", ((0, 1),)),
            0,
            agent,
            training=False,
            max_steps=4,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.total_reward, 10.0)
        self.assertTrue(agent.ended)

    def test_summary_handles_failures_and_empty_input(self) -> None:
        env = QuantumEnvironment(initial_state="one", seed=0)
        agent = FixedActionAgent(0)
        goals = (GoalSpec("hit", ((0, 1),)), GoalSpec("miss", ((1, 0),)))
        results = [
            run_episode(env, goals[0], 0, agent, training=False, max_steps=1),
            run_episode(env, goals[1], 1, agent, training=False, max_steps=1),
        ]
        self.assertEqual(summarize_results(results)["success_rate"], 0.5)
        self.assertIn("miss", summarize_by_goal(results, goals))
        self.assertTrue(str(summarize_results([])["success_rate"]) == "nan")


if __name__ == "__main__":
    unittest.main()

