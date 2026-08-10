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


# ============================================================================
# Hidden quantum environment
# ============================================================================

class QuantumEnvironment:
    """
    One hidden qubit with three available measurement interventions.

    PUBLIC API FOR THE AGENT:
        env.n_actions
        env.n_outcomes
        outcome = env.step(action)

    The leading underscore on _rho and _kraus is intentional: those are hidden
    environment variables and must not be used by an agent.
    """

    action_names = ("Z", "X", "weak-Z")

    def __init__(
        self,
        initial_state: str = "one",
        weak_q: float = 0.80,
        seed: int = 0,
    ):
        if not (0.5 < weak_q < 1.0):
            raise ValueError("weak_q must lie strictly between 0.5 and 1.0")

        self.initial_state = initial_state
        self.weak_q = float(weak_q)
        self.rng = np.random.default_rng(seed)

        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        plus = (zero + one) / np.sqrt(2.0)
        minus = (zero - one) / np.sqrt(2.0)

        Z0 = np.outer(zero, zero.conj())
        Z1 = np.outer(one, one.conj())

        X0 = np.outer(plus, plus.conj())
        X1 = np.outer(minus, minus.conj())

        q = self.weak_q
        W0 = np.diag([np.sqrt(q), np.sqrt(1.0 - q)]).astype(complex)
        W1 = np.diag([np.sqrt(1.0 - q), np.sqrt(q)]).astype(complex)

        self._kraus = ((Z0, Z1), (X0, X1), (W0, W1))
        self.n_actions = len(self._kraus)
        self.n_outcomes = 2
        self._rho: np.ndarray | None = None
        self.reset()

    def _initial_density_matrix(self) -> np.ndarray:
        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        plus = (zero + one) / np.sqrt(2.0)
        minus = (zero - one) / np.sqrt(2.0)

        pure = {
            "zero": zero,
            "one": one,
            "plus": plus,
            "minus": minus,
        }

        if self.initial_state == "mixed":
            return np.eye(2, dtype=complex) / 2.0

        if self.initial_state not in pure:
            raise ValueError(
                "initial_state must be one of zero, one, plus, minus, mixed"
            )

        psi = pure[self.initial_state]
        return np.outer(psi, psi.conj())

    def reset(self) -> None:
        self._rho = self._initial_density_matrix()

    def step(self, action: int) -> int:
        """
        Secret environment calculation:

            p(o) = Tr(K_o rho K_o^†)
            rho' = K_o rho K_o^† / p(o)

        Only the integer outcome is returned to the agent.
        """
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"invalid action {action}")

        assert self._rho is not None
        Ks = self._kraus[int(action)]

        probs = np.array(
            [
                np.trace(K @ self._rho @ K.conj().T).real
                for K in Ks
            ],
            dtype=float,
        )
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()

        outcome = int(self.rng.choice(self.n_outcomes, p=probs))
        K = Ks[outcome]

        unnormalized = K @ self._rho @ K.conj().T
        self._rho = unnormalized / np.trace(unnormalized)

        return outcome


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
        a, o = token.split(":")
        target.append((int(a), int(o)))

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


def goal_to_string(spec: GoalSpec) -> str:
    names = QuantumEnvironment.action_names
    return " -> ".join(f"{names[a]}:{o}" for a, o in spec.target)


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
            f"target={goal_to_string(spec)}"
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

DEFAULT_GOALS = (
    "Z0=0:0;"
    "Z1=0:1;"
    "X0=1:0;"
    "X1=1:1;"
    "Z0_X0=0:0,1:0;"
    "X0_Z0=1:0,0:0"
)


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1_000)

    parser.add_argument(
        "--initial-state",
        choices=("zero", "one", "plus", "minus", "mixed"),
        default="one",
    )
    parser.add_argument("--weak-q", type=float, default=0.80)

    parser.add_argument(
        "--goals",
        type=str,
        default=DEFAULT_GOALS,
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
        initial_state=args.initial_state,
        weak_q=args.weak_q,
        seed=args.seed + seed_offset,
    )


def make_goals(args) -> tuple[GoalSpec, ...]:
    return parse_goals(args.goals)
