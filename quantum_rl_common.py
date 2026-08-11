"""
quantum_rl_common.py
====================

Shared quantum environment, goal definitions, backend protocol, and experiment
runner for the quantum-measurement RL examples.

DESIGN PRINCIPLE
----------------
The hidden quantum model belongs to the environment, never to the agent.

The only information crossing the environment -> agent boundary is:

    chosen action (known because the agent chose it)
    observed classical outcome
    scalar reward / goal progress

The agent never receives:
    * rho
    * Kraus operators
    * Born probabilities
    * expectation values
    * tomography data

Default actions
---------------
    0 = projective Z measurement
    1 = projective X measurement
    2 = weak Z measurement

All measurements have outcomes 0 and 1.

Goals
-----
A goal is an ordered sequence of action/outcome checkpoints.  Other actions may
occur between checkpoints.  For example

    ((0,0), (1,0), (0,0))

means "eventually get Z:0, later X:0, later Z:0".

This file supports both one-goal and multi-goal agents through one interface.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from quantum_environments import (
    DEFAULT_GOALS_BY_ENVIRONMENT,
    QuantumEnvironment,
    available_environments,
)


# ============================================================================
# Goal definitions
# ============================================================================

@dataclass(frozen=True)
class GoalSpec:
    name: str
    target: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not self.target:
            raise ValueError("a goal must contain at least one checkpoint")

    @property
    def length(self) -> int:
        return len(self.target)


class GoalTracker:
    """
    Mutable per-episode progress through an immutable GoalSpec.
    """

    def __init__(
        self,
        spec: GoalSpec,
        step_penalty: float = -0.01,
        checkpoint_reward: float = 1.0,
        completion_reward: float = 10.0,
    ):
        self.spec = spec
        self.step_penalty = float(step_penalty)
        self.checkpoint_reward = float(checkpoint_reward)
        self.completion_reward = float(completion_reward)
        self.progress = 0

    def reset(self) -> None:
        self.progress = 0

    def update(self, action: int, outcome: int) -> tuple[float, bool]:
        desired = self.spec.target[self.progress]

        if (int(action), int(outcome)) != desired:
            return self.step_penalty, False

        self.progress += 1

        if self.progress == self.spec.length:
            return self.completion_reward, True

        return self.checkpoint_reward, False


def parse_target(text: str) -> tuple[tuple[int, int], ...]:
    """
    "0:0,1:1,2:0" -> ((0,0),(1,1),(2,0))
    """
    target = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"invalid checkpoint {token!r}; expected ACTION:OUTCOME"
            )
        a, o = (int(part) for part in parts)
        if a < 0 or o < 0:
            raise ValueError(f"checkpoint indices must be non-negative: {token!r}")
        target.append((a, o))

    if not target:
        raise ValueError(f"empty target: {text!r}")

    return tuple(target)


def parse_goals(text: str) -> tuple[GoalSpec, ...]:
    """
    Parse a semicolon-separated goal set.

    Example:
        'Z0=0:0;X0=1:0;Z0_X0=0:0,1:0'

    If no explicit name is supplied:
        '0:0;1:0'
    becomes goal0 and goal1.
    """
    specs = []

    for i, raw in enumerate(text.split(";")):
        raw = raw.strip()
        if not raw:
            continue

        if "=" in raw:
            name, target_text = raw.split("=", 1)
            name = name.strip()
        else:
            name = f"goal{i}"
            target_text = raw

        specs.append(GoalSpec(name=name, target=parse_target(target_text)))

    if not specs:
        raise ValueError("no goals were parsed")

    return tuple(specs)


def goal_to_string(
    spec: GoalSpec,
    action_names: Sequence[str] | None = None,
) -> str:
    """Render a goal, falling back to action indices for custom environments."""
    names = tuple(action_names or ("Z", "X", "weak-Z"))
    return " -> ".join(
        f"{names[a] if 0 <= a < len(names) else f'a{a}'}:{o}"
        for a, o in spec.target
    )


def validate_goals(goals: Sequence[GoalSpec], env: QuantumEnvironment) -> None:
    """Reject checkpoints that cannot occur in the selected environment."""
    names: set[str] = set()
    for spec in goals:
        if not spec.name:
            raise ValueError("goal names must not be empty")
        if spec.name in names:
            raise ValueError(f"duplicate goal name {spec.name!r}")
        names.add(spec.name)
        for action, outcome in spec.target:
            if not 0 <= action < env.n_actions:
                raise ValueError(
                    f"goal {spec.name!r} uses unavailable action {action} "
                    f"in {env.environment_name}"
                )
            if not 0 <= outcome < env.action_outcome_counts[action]:
                raise ValueError(
                    f"goal {spec.name!r} uses unavailable outcome {outcome} "
                    f"for action {env.action_names[action]!r}"
                )


# ============================================================================
# Swappable backend interface
# ============================================================================

class AgentBackend(Protocol):
    """
    Implement this protocol to plug in a new backend.

    A transformer, world-model planner, successor-feature agent, etc. can be
    compared without changing the environment or evaluation code.
    """

    name: str

    def reset_episode(
        self,
        goal_id: int,
        goal_length: int,
        training: bool,
    ) -> None:
        ...

    def act(
        self,
        goal_id: int,
        progress: int,
        training: bool,
    ) -> int:
        ...

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
        ...

    def end_episode(self, training: bool) -> None:
        ...


# ============================================================================
# Shared experiment runner
# ============================================================================

@dataclass
class EpisodeResult:
    goal_id: int
    goal_name: str
    success: bool
    steps: int
    total_reward: float


def run_episode(
    env: QuantumEnvironment,
    spec: GoalSpec,
    goal_id: int,
    agent: AgentBackend,
    *,
    training: bool,
    max_steps: int,
    step_penalty: float = -0.01,
    checkpoint_reward: float = 1.0,
    completion_reward: float = 10.0,
) -> EpisodeResult:
    env.reset()

    tracker = GoalTracker(
        spec,
        step_penalty=step_penalty,
        checkpoint_reward=checkpoint_reward,
        completion_reward=completion_reward,
    )

    agent.reset_episode(goal_id, spec.length, training)

    total_reward = 0.0

    for t in range(max_steps):
        progress = tracker.progress

        action = agent.act(goal_id, progress, training)
        outcome = env.step(action)

        reward, done = tracker.update(action, outcome)
        next_progress = tracker.progress

        agent.observe(
            goal_id=goal_id,
            progress=progress,
            action=action,
            outcome=outcome,
            reward=reward,
            done=done,
            next_progress=next_progress,
            training=training,
        )

        total_reward += reward

        if done:
            agent.end_episode(training)
            return EpisodeResult(
                goal_id=goal_id,
                goal_name=spec.name,
                success=True,
                steps=t + 1,
                total_reward=total_reward,
            )

    agent.end_episode(training)

    return EpisodeResult(
        goal_id=goal_id,
        goal_name=spec.name,
        success=False,
        steps=max_steps,
        total_reward=total_reward,
    )


def train_agent(
    env: QuantumEnvironment,
    goals: Sequence[GoalSpec],
    agent: AgentBackend,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    log_every: int = 1000,
    step_penalty: float = -0.01,
    checkpoint_reward: float = 1.0,
    completion_reward: float = 10.0,
) -> list[EpisodeResult]:
    """
    Sample goals uniformly during training so all backends see the same task
    distribution.
    """
    rng = np.random.default_rng(seed)
    results: list[EpisodeResult] = []

    for episode in range(1, episodes + 1):
        goal_id = int(rng.integers(len(goals)))
        spec = goals[goal_id]

        result = run_episode(
            env,
            spec,
            goal_id,
            agent,
            training=True,
            max_steps=max_steps,
            step_penalty=step_penalty,
            checkpoint_reward=checkpoint_reward,
            completion_reward=completion_reward,
        )
        results.append(result)

        if log_every and episode % log_every == 0:
            recent = results[-log_every:]
            s = summarize_results(recent)
            print(
                f"[{agent.name}] episode {episode:>7d} | "
                f"recent success={s['success_rate']:.3f} | "
                f"mean steps={s['mean_steps_success']:.2f}"
            )

    return results


def evaluate_agent(
    env: QuantumEnvironment,
    goals: Sequence[GoalSpec],
    agent: AgentBackend,
    *,
    episodes_per_goal: int,
    max_steps: int,
    step_penalty: float = -0.01,
    checkpoint_reward: float = 1.0,
    completion_reward: float = 10.0,
) -> list[EpisodeResult]:
    """
    Evaluate each goal equally often, with exploration disabled.
    """
    results = []

    for goal_id, spec in enumerate(goals):
        for _ in range(episodes_per_goal):
            results.append(
                run_episode(
                    env,
                    spec,
                    goal_id,
                    agent,
                    training=False,
                    max_steps=max_steps,
                    step_penalty=step_penalty,
                    checkpoint_reward=checkpoint_reward,
                    completion_reward=completion_reward,
                )
            )

    return results


def summarize_results(results: Sequence[EpisodeResult]) -> dict[str, float]:
    if not results:
        return {
            "success_rate": float("nan"),
            "mean_steps_success": float("nan"),
            "mean_reward": float("nan"),
        }

    successes = [r for r in results if r.success]

    return {
        "success_rate": len(successes) / len(results),
        "mean_steps_success": (
            float(np.mean([r.steps for r in successes]))
            if successes
            else float("nan")
        ),
        "mean_reward": float(np.mean([r.total_reward for r in results])),
    }


def summarize_by_goal(
    results: Sequence[EpisodeResult],
    goals: Sequence[GoalSpec],
) -> dict[str, dict[str, float]]:
    output = {}

    for goal_id, spec in enumerate(goals):
        subset = [r for r in results if r.goal_id == goal_id]
        output[spec.name] = summarize_results(subset)

    return output


def print_evaluation_summary(
    agent: AgentBackend,
    goals: Sequence[GoalSpec],
    results: Sequence[EpisodeResult],
    action_names: Sequence[str] | None = None,
) -> None:
    print()
    print("=" * 78)
    print(f"backend: {agent.name}")
    print("-" * 78)

    by_goal = summarize_by_goal(results, goals)

    for spec in goals:
        s = by_goal[spec.name]
        print(
            f"{spec.name:<18} "
            f"success={s['success_rate']:.3f}   "
            f"mean_steps={s['mean_steps_success']:.3f}   "
            f"target={goal_to_string(spec, action_names)}"
        )

    overall = summarize_results(results)
    print("-" * 78)
    print(
        f"OVERALL            success={overall['success_rate']:.3f}   "
        f"mean_steps={overall['mean_steps_success']:.3f}"
    )
    print("=" * 78)


def save_results_csv(
    path: str | Path,
    backend_name: str,
    phase: str,
    results: Sequence[EpisodeResult],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "phase",
                "episode",
                "goal_id",
                "goal_name",
                "success",
                "steps",
                "total_reward",
            ]
        )

        for i, r in enumerate(results, start=1):
            writer.writerow(
                [
                    backend_name,
                    phase,
                    i,
                    r.goal_id,
                    r.goal_name,
                    int(r.success),
                    r.steps,
                    r.total_reward,
                ]
            )


# ============================================================================
# Shared CLI
# ============================================================================

DEFAULT_GOALS = DEFAULT_GOALS_BY_ENVIRONMENT["qubit-zx-weak"]


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1_000)

    parser.add_argument(
        "--environment",
        choices=available_environments(),
        default="qubit-zx-weak",
        help="Hidden measurement world; each has its own default goals.",
    )
    parser.add_argument(
        "--initial-state",
        default=None,
        help="Environment-specific state name (omit to use its documented default).",
    )
    parser.add_argument("--weak-q", type=float, default=0.80)

    parser.add_argument(
        "--goals",
        type=str,
        default=None,
        help=(
            'Semicolon-separated goals, e.g. '
            '"Z0=0:0;X0=1:0;Z0X0=0:0,1:0"'
        ),
    )

    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--checkpoint-reward", type=float, default=1.0)
    parser.add_argument("--completion-reward", type=float, default=10.0)

    parser.add_argument("--csv", type=str, default=None)

    return parser


def make_environment(args, *, seed_offset: int = 0) -> QuantumEnvironment:
    return QuantumEnvironment(
        environment=args.environment,
        initial_state=args.initial_state,
        weak_q=args.weak_q,
        seed=args.seed + seed_offset,
    )


def make_goals(args) -> tuple[GoalSpec, ...]:
    text = args.goals or DEFAULT_GOALS_BY_ENVIRONMENT[args.environment]
    goals = parse_goals(text)
    # Configuration errors should be reported before a long training run.  A
    # short-lived environment is sufficient because validation uses metadata
    # only and does not expose the hidden state to an agent.
    env = QuantumEnvironment(
        environment=args.environment,
        initial_state=args.initial_state,
        weak_q=args.weak_q,
        seed=args.seed,
    )
    validate_goals(goals, env)
    return goals
