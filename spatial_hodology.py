"""Inverse-design and evaluate a two-dimensional hodological quantum world.

The agent is never given lattice coordinates.  A privileged outer loop chooses
the hidden quantum instrument parameters so that exact optimal hitting costs
are well represented by two Euclidean dimensions.  A tabular Q-learner then
has to discover navigation policies from action/outcome/reward experience.

The resulting geometry is evaluated twice:

* ``designed`` geometry: exact costs implied by the hidden instruments;
* ``learned`` geometry: empirical goal-hitting costs under learned policies.

Run the production study with::

    python spatial_hodology.py --output results/spatial-hodology

This module deliberately keeps its numerical geometry utilities independent of
PyTorch so the inverse-design analysis and figures remain lightweight.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from q_learning import TabularQAgent
from quantum_environments import DEFAULT_GOALS_BY_ENVIRONMENT, QuantumEnvironment
from quantum_rl_common import (
    EpisodeResult,
    GoalSpec,
    GoalTracker,
    parse_goals,
    summarize_results,
    validate_goals,
)


GRID_SIZE = 3
N_SITES = GRID_SIZE**2
COORDINATES = np.array(
    [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)], dtype=float
)
CARDINAL_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAGONAL_DIRECTIONS = ((1, -1), (1, 1), (-1, 1), (-1, -1))


class PlaceQAgent(TabularQAgent):
    """Q-learning over the latest coordinate-free place observation.

    All movement instruments in the place-observed world report their
    destination symbol.  The sufficient observable state is therefore the
    latest symbol, not the incoming action or a privileged coordinate.
    """

    name = "place-symbol-tabular-q"

    def __init__(self, **kwargs):
        super().__init__(history_length=1, **kwargs)
        self.current_place: int | None = None

    def _state(self, goal_id: int, progress: int):
        return int(goal_id), int(progress), self.current_place

    def reset_episode(self, goal_id: int, goal_length: int, training: bool) -> None:
        self.current_place = None
        self._last_state = None

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
        if self._last_state is None:
            raise RuntimeError("observe() called before act()")
        self.current_place = int(outcome)
        next_state = self._state(goal_id, next_progress)
        if training:
            old_q = self.Q[self._last_state][action]
            target = (
                float(reward)
                if done
                else float(reward) + self.gamma * np.max(self.Q[next_state])
            )
            self.Q[self._last_state][action] += self.alpha * (target - old_q)


def pairwise_distances(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    delta = rows[:, None, :] - rows[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def exact_movement_costs(
    diagonal_success: float = 1.0 / np.sqrt(2.0),
    *,
    cardinal_only: bool = False,
) -> np.ndarray:
    """Optimal expected movement costs on the open lattice.

    An axial attempt costs one intervention.  Repeating a diagonal action until
    its first success costs ``1 / diagonal_success`` in expectation.  Dijkstra
    therefore gives the stochastic-shortest-path cost exactly.
    """
    if not 0.0 < diagonal_success <= 1.0:
        raise ValueError("diagonal_success must lie in (0, 1]")
    moves = [(dx, dy, 1.0) for dx, dy in CARDINAL_DIRECTIONS]
    if not cardinal_only:
        moves += [
            (dx, dy, 1.0 / diagonal_success)
            for dx, dy in DIAGONAL_DIRECTIONS
        ]
    result = np.zeros((N_SITES, N_SITES), dtype=float)
    for source in range(N_SITES):
        costs = np.full(N_SITES, np.inf)
        costs[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            cost, site = heapq.heappop(queue)
            if cost != costs[site]:
                continue
            x, y = site % GRID_SIZE, site // GRID_SIZE
            for dx, dy, edge_cost in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    continue
                neighbor = ny * GRID_SIZE + nx
                candidate = cost + edge_cost
                if candidate < costs[neighbor] - 1e-12:
                    costs[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        result[source] = costs
    return result


def classical_mds(
    distances: np.ndarray,
    dimensions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical metric MDS coordinates and centered-Gram eigenvalues."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    usable = min(dimensions, n)
    coordinates = eigenvectors[:, :usable] * np.sqrt(
        np.maximum(eigenvalues[:usable], 0.0)
    )
    if usable < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - usable)))
    return coordinates, eigenvalues


def normalized_stress(distances: np.ndarray, coordinates: np.ndarray) -> float:
    """Kruskal-style normalized raw stress over unordered pairs."""
    fitted = pairwise_distances(coordinates)
    indices = np.triu_indices(len(distances), 1)
    denominator = np.sum(np.asarray(distances)[indices] ** 2)
    if denominator <= 0.0:
        return 0.0
    return float(
        np.sqrt(np.sum((fitted[indices] - np.asarray(distances)[indices]) ** 2) / denominator)
    )


def metric_mds(
    distances: np.ndarray,
    dimensions: int,
    *,
    restarts: int = 4,
    max_iterations: int = 400,
) -> np.ndarray:
    """Minimize metric raw stress with deterministic SMACOF majorization."""
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    starts = [classical_mds(distances, dimensions)[0]]
    if dimensions > 1:
        lower = metric_mds(
            distances,
            dimensions - 1,
            restarts=max(1, restarts // 2),
            max_iterations=max_iterations,
        )
        starts.append(np.pad(lower, ((0, 0), (0, 1))))
    for restart in range(restarts):
        starts.append(np.random.default_rng(19_733 + restart).normal(size=(n, dimensions)))

    best = starts[0]
    best_stress = normalized_stress(distances, best)
    for initial in starts:
        coordinates = initial - initial.mean(axis=0, keepdims=True)
        for _ in range(max_iterations):
            fitted = pairwise_distances(coordinates)
            ratios = np.divide(
                distances,
                fitted,
                out=np.zeros_like(distances),
                where=fitted > 1e-12,
            )
            majorizer = -ratios
            np.fill_diagonal(majorizer, 0.0)
            np.fill_diagonal(majorizer, -majorizer.sum(axis=1))
            updated = majorizer @ coordinates / n
            updated -= updated.mean(axis=0, keepdims=True)
            if np.linalg.norm(updated - coordinates) < 1e-10:
                coordinates = updated
                break
            coordinates = updated
        stress = normalized_stress(distances, coordinates)
        if stress < best_stress:
            best, best_stress = coordinates, stress
    return best


def procrustes_r2(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Similarity-transform agreement, invariant to rotation/scale/translation."""
    estimated = np.asarray(estimated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    estimated = estimated - estimated.mean(axis=0, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)
    left, _, right = np.linalg.svd(estimated.T @ reference)
    rotation = left @ right
    rotated = estimated @ rotation
    scale = np.sum(rotated * reference) / max(np.sum(rotated**2), 1e-12)
    residual = np.sum((scale * rotated - reference) ** 2)
    total = max(np.sum(reference**2), 1e-12)
    return float(1.0 - residual / total)


def distance_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation between upper triangles of two distance matrices."""
    indices = np.triu_indices(len(first), 1)
    x, y = np.asarray(first)[indices], np.asarray(second)[indices]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def geometry_metrics(distances: np.ndarray) -> dict[str, float]:
    """Intrinsic-dimension and recovery diagnostics for a symmetric geometry."""
    symmetric = (np.asarray(distances) + np.asarray(distances).T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    _, eigenvalues = classical_mds(symmetric, 2)
    coordinates_1d = metric_mds(symmetric, 1)
    coordinates_2d = metric_mds(symmetric, 2)
    coordinates_3d = metric_mds(symmetric, 3)
    positive = np.maximum(eigenvalues, 0.0)
    absolute = np.sum(np.abs(eigenvalues))
    return {
        "stress_1d": normalized_stress(symmetric, coordinates_1d),
        "stress_2d": normalized_stress(symmetric, coordinates_2d),
        "stress_3d": normalized_stress(symmetric, coordinates_3d),
        "positive_variance_2d": float(positive[:2].sum() / max(positive.sum(), 1e-12)),
        "negative_spectrum_fraction": float(
            np.abs(np.minimum(eigenvalues, 0.0)).sum() / max(absolute, 1e-12)
        ),
        "coordinate_procrustes_r2": procrustes_r2(coordinates_2d, COORDINATES),
        "euclidean_distance_correlation": distance_correlation(
            symmetric, pairwise_distances(COORDINATES)
        ),
    }


def search_diagonal_instrument(
    candidates: Sequence[float] | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Privileged outer-loop search for the most Euclidean move instrument."""
    values = candidates if candidates is not None else np.linspace(0.55, 0.95, 81)
    rows: list[dict[str, float]] = []
    for probability in values:
        metrics = geometry_metrics(exact_movement_costs(float(probability)))
        objective = (
            metrics["stress_2d"]
            + 0.25 * (1.0 - metrics["coordinate_procrustes_r2"])
            + 0.10 * metrics["negative_spectrum_fraction"]
        )
        rows.append(
            {
                "diagonal_success": float(probability),
                "objective": float(objective),
                **metrics,
            }
        )
    best = min(rows, key=lambda row: row["objective"])
    return rows, best


def canonical_actions(site: int) -> list[int]:
    """A deterministic cardinal history from the center to a named place."""
    target_x, target_y = site % GRID_SIZE, site // GRID_SIZE
    x = y = GRID_SIZE // 2
    actions: list[int] = []
    while y > target_y:
        actions.append(0)  # north
        y -= 1
    while y < target_y:
        actions.append(2)  # south
        y += 1
    while x < target_x:
        actions.append(1)  # east
        x += 1
    while x > target_x:
        actions.append(3)  # west
        x -= 1
    return actions


def _observe_starting_place(
    env: QuantumEnvironment,
    agent: TabularQAgent,
    goal_id: int,
) -> int:
    """Supply one coordinate-free place observation before the task clock."""
    probe_action = env.n_actions - 1
    agent.act(goal_id, 0, training=False)
    outcome = env.step(probe_action)
    agent.observe(goal_id, 0, probe_action, outcome, 0.0, False, 0, False)
    return outcome


def train_random_start_agent(
    env: QuantumEnvironment,
    goals: Sequence[GoalSpec],
    agent: TabularQAgent,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> list[EpisodeResult]:
    """Train uniformly over source--goal pairs after an initial place probe.

    The source selector belongs to the experimental harness.  The only source
    information received by the learner is the ordinary probe outcome; its
    integer identity has no supplied coordinate or metric meaning.
    """
    rng = np.random.default_rng(seed)
    results: list[EpisodeResult] = []
    for _ in range(episodes):
        source = int(rng.integers(N_SITES))
        goal_id = int(rng.integers(len(goals)))
        spec = goals[goal_id]
        env.initial_state = f"site-{source}"
        env.reset()
        agent.reset_episode(goal_id, spec.length, training=True)
        observed_source = _observe_starting_place(env, agent, goal_id)
        if observed_source != source:
            raise RuntimeError("sharp place probe did not identify the prepared site")
        tracker = GoalTracker(spec)
        total_reward = 0.0
        finished = False
        for step in range(1, max_steps + 1):
            progress = tracker.progress
            action = agent.act(goal_id, progress, training=True)
            outcome = env.step(action)
            reward, done = tracker.update(action, outcome)
            agent.observe(
                goal_id,
                progress,
                action,
                outcome,
                reward,
                done,
                tracker.progress,
                True,
            )
            total_reward += reward
            if done:
                results.append(
                    EpisodeResult(goal_id, spec.name, True, step, total_reward)
                )
                finished = True
                break
        if not finished:
            results.append(
                EpisodeResult(goal_id, spec.name, False, max_steps, total_reward)
            )
        agent.end_episode(training=True)
    return results


def _rollout_from_current_history(
    env: QuantumEnvironment,
    agent: TabularQAgent,
    spec: GoalSpec,
    goal_id: int,
    max_steps: int,
) -> tuple[bool, int]:
    tracker = GoalTracker(spec)
    for step in range(1, max_steps + 1):
        progress = tracker.progress
        action = agent.act(goal_id, progress, training=False)
        outcome = env.step(action)
        reward, done = tracker.update(action, outcome)
        agent.observe(
            goal_id, progress, action, outcome, reward, done, tracker.progress, False
        )
        if done:
            return True, step
    return False, max_steps + 1


def empirical_hodological_costs(
    agent: TabularQAgent,
    goals: Sequence[GoalSpec],
    environment: str,
    diagonal_success: float,
    *,
    episodes_per_pair: int,
    max_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate all learned goal-to-goal costs from operational rollouts."""
    costs = np.zeros((N_SITES, N_SITES), dtype=float)
    success = np.zeros_like(costs)
    for source in range(N_SITES):
        for target, spec in enumerate(goals):
            times = []
            successes = 0
            for episode in range(episodes_per_pair):
                env = QuantumEnvironment(
                    environment=environment,
                    initial_state=f"site-{source}",
                    weak_q=diagonal_success,
                    seed=seed + source * 100_000 + target * 1_000 + episode,
                )
                agent.reset_episode(target, spec.length, training=False)
                observed_source = _observe_starting_place(env, agent, target)
                if observed_source != source:
                    raise RuntimeError("sharp place probe did not identify source")
                achieved, steps = _rollout_from_current_history(
                    env, agent, spec, target, max_steps
                )
                successes += int(achieved)
                times.append(steps)
                agent.end_episode(training=False)
            # Remove the common final probe intervention.  Failures retain a
            # finite restricted-mean penalty of max_steps rather than vanishing.
            costs[source, target] = max(float(np.mean(times)) - 1.0, 0.0)
            success[source, target] = successes / episodes_per_pair
    np.fill_diagonal(costs, 0.0)
    return costs, success


def collect_policy_trajectory(
    agent: PlaceQAgent,
    goals: Sequence[GoalSpec],
    environment: str,
    diagonal_success: float,
    *,
    target: int,
    seed: int,
    max_steps: int,
) -> list[dict[str, object]]:
    """Record motion through place symbols under one learned goal policy."""
    env = QuantumEnvironment(
        environment=environment,
        initial_state="center",
        weak_q=diagonal_success,
        seed=seed + 3_000_000,
    )
    spec = goals[target]
    agent.reset_episode(target, spec.length, training=False)
    place = _observe_starting_place(env, agent, target)
    rows: list[dict[str, object]] = [
        {
            "environment": environment,
            "seed": seed,
            "target": target,
            "step": 0,
            "action": "initial-place-probe",
            "place": place,
            "success": False,
        }
    ]
    tracker = GoalTracker(spec)
    for step in range(1, max_steps + 1):
        progress = tracker.progress
        action = agent.act(target, progress, training=False)
        outcome = env.step(action)
        reward, done = tracker.update(action, outcome)
        agent.observe(
            target, progress, action, outcome, reward, done, tracker.progress, False
        )
        rows.append(
            {
                "environment": environment,
                "seed": seed,
                "target": target,
                "step": step,
                "action": env.action_names[action],
                "place": int(agent.current_place),
                "success": done,
            }
        )
        if done:
            break
    return rows


def place_q_rows(agent: PlaceQAgent) -> list[dict[str, object]]:
    """Serialize the compact learned policy table without pickling code."""
    rows: list[dict[str, object]] = []
    for (goal_id, progress, place), values in sorted(
        agent.Q.items(), key=lambda item: str(item[0])
    ):
        for action, value in enumerate(values):
            rows.append(
                {
                    "goal_id": goal_id,
                    "progress": progress,
                    "place": place,
                    "action": action,
                    "q_value": float(value),
                }
            )
    return rows


def _matrix_rows(names: Sequence[str], matrix: np.ndarray) -> list[dict[str, object]]:
    return [
        {"source": name, **{target: float(value) for target, value in zip(names, row)}}
        for name, row in zip(names, matrix)
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_design(output: Path, search_rows: Sequence[dict[str, float]], best: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    probabilities = [row["diagonal_success"] for row in search_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(probabilities, [row["stress_2d"] for row in search_rows], label="2D")
    axes[0].plot(probabilities, [row["stress_3d"] for row in search_rows], label="3D")
    axes[0].plot(
        probabilities,
        [row["objective"] for row in search_rows],
        color="tab:purple",
        linestyle="--",
        label="design objective",
    )
    axes[0].axvline(best["diagonal_success"], color="black", linestyle="--", alpha=0.7)
    axes[0].axvline(1 / np.sqrt(2), color="tab:red", linestyle=":", label=r"$1/\sqrt{2}$")
    axes[0].set(xlabel="diagonal success probability", ylabel="normalized stress", title="Inverse design of movement cost")
    axes[0].legend(frameon=False, loc="upper right")

    exact = exact_movement_costs(best["diagonal_success"])
    embedded = metric_mds(exact, 2)
    _, eigenvalues = classical_mds(exact, 2)
    colors = np.arange(N_SITES)
    axes[1].scatter(embedded[:, 0], embedded[:, 1], c=colors, cmap="viridis", s=90)
    for index, (x, y) in enumerate(embedded):
        axes[1].text(x, y, chr(65 + index), ha="center", va="center", color="white", weight="bold")
    axes[1].set_aspect("equal")
    axes[1].set_title(
        f"Exact hodological MDS\n2D stress={normalized_stress(exact, embedded):.3f}; "
        f"top eigenvalues={eigenvalues[0]:.2f},{eigenvalues[1]:.2f}"
    )
    axes[1].set_xlabel("emergent coordinate 1")
    axes[1].set_ylabel("emergent coordinate 2")
    figure.tight_layout()
    figure.savefig(output / "design_optimization.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_learned_geometries(
    output: Path,
    records: Sequence[dict[str, object]],
    matrices: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    columns = 3
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows), squeeze=False)
    for axis, record in zip(axes.flat, records):
        run_id = str(record["run_id"])
        matrix = (matrices[run_id] + matrices[run_id].T) / 2.0
        coordinates = metric_mds(matrix, 2)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=np.arange(N_SITES), cmap="viridis", s=75)
        for index, (x, y) in enumerate(coordinates):
            axis.text(x, y, chr(65 + index), ha="center", va="center", color="white", fontsize=8, weight="bold")
        axis.set_aspect("equal")
        axis.set_title(
            f"{record['environment'].replace('qudit-grid-3x3', 'grid')} / seed {record['seed']}\n"
            f"stress$_2$={record['stress_2d']:.2f}, $R^2_P$={record['coordinate_procrustes_r2']:.2f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(records):]:
        axis.axis("off")
    figure.suptitle("Geometry recovered from learned policy hitting costs", weight="bold")
    figure.tight_layout()
    figure.savefig(output / "learned_hodological_spaces.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(output: Path, records: Sequence[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    environments = [
        "qudit-grid-3x3",
        "qudit-grid-3x3-blind",
        "qudit-grid-3x3-cardinal",
    ]
    labels = ["place observed", "success/failure only", "cardinal only"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    fields = (
        ("center_success", "success from reset"),
        ("pair_success", "all-pairs success"),
        ("stress_2d", "learned 2D stress"),
    )
    for axis, (field, title) in zip(axes, fields):
        values = [[float(row[field]) for row in records if row["environment"] == env] for env in environments]
        means = [np.mean(group) for group in values]
        errors = [np.std(group, ddof=1) if len(group) > 1 else 0.0 for group in values]
        axis.bar(
            labels,
            means,
            yerr=errors,
            color=("#3b82f6", "#8b5cf6", "#f59e0b"),
            alpha=0.85,
            capsize=4,
        )
        for index, group in enumerate(values):
            axis.scatter(np.full(len(group), index), group, color="black", s=18, zorder=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        if "success" in field:
            axis.set_ylim(0, 1.05)
    figure.suptitle("Navigation competence and Euclidean compatibility", weight="bold")
    figure.tight_layout()
    figure.savefig(output / "performance_geometry_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_trajectories(
    output: Path,
    records: Sequence[dict[str, object]],
    matrices: dict[str, np.ndarray],
    trajectory_rows: Sequence[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in records if row["environment"] == "qudit-grid-3x3"]
    figure, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.2), squeeze=False)
    for axis, record in zip(axes.flat, selected):
        run_id = str(record["run_id"])
        symmetric = (matrices[run_id] + matrices[run_id].T) / 2.0
        embedded = metric_mds(symmetric, 2)
        seed = int(record["seed"])
        path_rows = [row for row in trajectory_rows if int(row["seed"]) == seed]
        places = [int(row["place"]) for row in path_rows]
        path = embedded[places]
        axis.scatter(embedded[:, 0], embedded[:, 1], color="#dbeafe", edgecolor="#2563eb", s=100, zorder=1)
        for index, (x, y) in enumerate(embedded):
            axis.text(x, y, chr(65 + index), ha="center", va="center", fontsize=8, weight="bold")
        axis.plot(path[:, 0], path[:, 1], color="#dc2626", linewidth=2.2, zorder=2)
        axis.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap="plasma", s=45, zorder=3)
        for step, (x, y) in enumerate(path):
            axis.annotate(str(step), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7)
        target = int(path_rows[0]["target"])
        axis.scatter(*embedded[target], marker="*", s=230, color="#16a34a", edgecolor="black", zorder=4)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"seed {seed}: E $\\rightarrow$ {chr(65 + target)}\n"
            f"{len(path) - 1} interventions; {'success' if path_rows[-1]['success'] else 'censored'}"
        )
    figure.suptitle(
        "Learned policy trajectories in emergent hodological coordinates",
        weight="bold",
        y=1.03,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output / "emergent_policy_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_fiber_bundle_outlook(output: Path) -> None:
    """Render the proposed base-space/internal-fiber decomposition."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.5, 6.2))
    axis = figure.add_subplot(111, projection="3d")
    levels = (0.0, 0.38, 0.76)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            axis.plot([x, x], [y, y], [levels[0], levels[-1]], color="#64748b", alpha=0.55)
            axis.scatter(
                [x] * len(levels),
                [y] * len(levels),
                levels,
                c=("#0ea5e9", "#8b5cf6", "#ec4899"),
                s=28,
                depthshade=False,
            )
    for y in range(GRID_SIZE):
        axis.plot(range(GRID_SIZE), [y] * GRID_SIZE, [0] * GRID_SIZE, color="#94a3b8", alpha=0.5)
    for x in range(GRID_SIZE):
        axis.plot([x] * GRID_SIZE, range(GRID_SIZE), [0] * GRID_SIZE, color="#94a3b8", alpha=0.5)
    axis.plot([1, 2], [1, 1], [levels[1], levels[1]], color="#dc2626", linewidth=4, label="horizontal/spatial action")
    axis.plot([2, 2], [1, 1], [levels[0], levels[-1]], color="#2563eb", linewidth=4, label="vertical/internal action")
    axis.text(1.5, 1.0, levels[1] + 0.06, "move in base", color="#991b1b", ha="center")
    axis.text(2.03, 1.0, 0.42, "change fiber state", color="#1d4ed8")
    axis.set(
        xlabel="emergent base coordinate 1",
        ylabel="emergent base coordinate 2",
        zlabel="internal / predictive fiber",
        title="Fiber-bundle research target (schematic, not an empirical result)",
    )
    axis.set_zticks(levels, ("$f_0$", "$f_1$", "$f_2$"))
    axis.view_init(elev=24, azim=-55)
    axis.legend(loc="upper left", frameon=False)
    figure.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.90)
    figure.savefig(output / "fiber_bundle_outlook.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/spatial-hodology")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--pair-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Place-observed worlds are Markov in the latest interaction.",
    )
    parser.add_argument("--blind-history-length", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    return parser


def _workspace_output(path: str) -> Path:
    workspace = Path(__file__).resolve().parent
    candidate = (workspace / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"output must stay inside {workspace}")
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    output = _workspace_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    search_rows, best = search_diagonal_instrument()
    diagonal_success = float(best["diagonal_success"])
    _write_csv(output / "design_search.csv", search_rows)

    exact_rows = []
    for name, matrix in (
        ("optimized-diagonal", exact_movement_costs(diagonal_success)),
        ("cardinal-only", exact_movement_costs(cardinal_only=True)),
    ):
        exact_rows.append({"design": name, **geometry_metrics(matrix)})
        _write_csv(output / f"exact_{name}_distances.csv", _matrix_rows(list("ABCDEFGHI"), matrix))
    _write_csv(output / "exact_geometry_summary.csv", exact_rows)

    seeds = [int(value) for value in args.seeds.split(",")]
    records: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}
    trajectory_rows: list[dict[str, object]] = []
    for environment in (
        "qudit-grid-3x3",
        "qudit-grid-3x3-blind",
        "qudit-grid-3x3-cardinal",
    ):
        goals = parse_goals(DEFAULT_GOALS_BY_ENVIRONMENT[environment])
        for seed in seeds:
            print(f"training {environment} seed {seed}", flush=True)
            train_env = QuantumEnvironment(
                environment=environment,
                initial_state="center",
                weak_q=diagonal_success,
                seed=seed,
            )
            validate_goals(goals, train_env)
            common_agent_args = {
                "n_actions": train_env.n_actions,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": args.epsilon_decay,
                "seed": seed + 12_345,
            }
            agent = (
                TabularQAgent(
                    history_length=args.blind_history_length,
                    **common_agent_args,
                )
                if environment.endswith("blind")
                else PlaceQAgent(**common_agent_args)
            )
            training = train_random_start_agent(
                train_env,
                goals,
                agent,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=seed + 9_999,
            )
            costs, pair_success = empirical_hodological_costs(
                agent,
                goals,
                environment,
                diagonal_success,
                episodes_per_pair=args.pair_episodes,
                max_steps=args.max_steps,
                seed=seed + 2_000_000,
            )
            run_id = f"{environment}__tabular__seed{seed}"
            matrices[run_id] = costs
            _write_csv(output / f"{run_id}__costs.csv", _matrix_rows(list("ABCDEFGHI"), costs))
            _write_csv(output / f"{run_id}__success.csv", _matrix_rows(list("ABCDEFGHI"), pair_success))
            symmetric = (costs + costs.T) / 2.0
            metrics = geometry_metrics(symmetric)
            exact = exact_movement_costs(
                diagonal_success, cardinal_only=environment.endswith("cardinal")
            )
            record: dict[str, object] = {
                "run_id": run_id,
                "environment": environment,
                "seed": seed,
                "training_success": summarize_results(training)["success_rate"],
                "center_success": float(pair_success[N_SITES // 2].mean()),
                "pair_success": float(pair_success.mean()),
                "exact_cost_correlation": distance_correlation(symmetric, exact),
                "directionality": float(
                    np.mean(np.abs(costs - costs.T)) / max(np.mean(symmetric), 1e-12)
                ),
                **metrics,
            }
            records.append(record)
            if isinstance(agent, PlaceQAgent):
                _write_csv(output / f"{run_id}__q_values.csv", place_q_rows(agent))
                if environment == "qudit-grid-3x3":
                    trajectory_rows.extend(
                        collect_policy_trajectory(
                            agent,
                            goals,
                            environment,
                            diagonal_success,
                            target=seed % N_SITES,
                            seed=seed,
                            max_steps=args.max_steps,
                        )
                    )
            print(
                f"  center={record['center_success']:.3f} pairs={record['pair_success']:.3f} "
                f"stress2={record['stress_2d']:.3f} R2={record['coordinate_procrustes_r2']:.3f}",
                flush=True,
            )

    _write_csv(output / "learned_geometry_summary.csv", records)
    _write_csv(output / "policy_trajectories.csv", trajectory_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "recover a 2D Euclidean-compatible hodological space",
        "hilbert_dimension": N_SITES,
        "initial_state": "|center><center|",
        "goal_count": N_SITES,
        "optimized_diagonal_success": diagonal_success,
        "theoretical_cost_ratio": 1.0 / diagonal_success,
        "configuration": vars(args),
        "best_design": best,
        "exact_geometry": exact_rows,
        "runs": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _plot_design(output, search_rows, best)
    _plot_learned_geometries(output, records, matrices)
    _plot_summary(output, records)
    _plot_trajectories(output, records, matrices, trajectory_rows)
    _plot_fiber_bundle_outlook(output)
    print(f"saved spatial hodology study to {output}")


if __name__ == "__main__":
    main()
