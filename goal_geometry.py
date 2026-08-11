"""Independent diagnostics for a trained multi-goal recurrent agent.

The network's Euclidean goal embeddings are only one candidate geometry.  This
module measures two independent operational objects from held-out rollouts:

* strategy distance: average square-root Jensen--Shannon policy divergence;
* trajectory distance: Euclidean separation between empirical behavior
  signatures.

It also checks whether learned blank-history cost predicts empirical hitting
time, and measures how each observed intervention/outcome displaces the full
vector of goal reachabilities.  No hidden density matrix is used.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from multi_goal_gru import MultiGoalGRUAgent
from quantum_environments import QuantumEnvironment
from quantum_rl_common import GoalSpec, GoalTracker


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Bounded metric ``sqrt(JS(p, q))`` using natural logarithms."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p /= p.sum()
    q /= q.sum()
    midpoint = (p + q) / 2.0
    divergence = 0.5 * (
        np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint))
    )
    return float(np.sqrt(max(divergence, 0.0)))


def pairwise_euclidean(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    delta = rows[:, None, :] - rows[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def pca_projection(rows: np.ndarray, dimensions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-light PCA projection and explained-variance fractions."""
    rows = np.asarray(rows, dtype=float)
    centered = rows - rows.mean(axis=0, keepdims=True)
    if len(rows) == 1:
        return np.zeros((1, dimensions)), np.zeros(dimensions)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    usable = min(dimensions, vh.shape[0])
    projected = centered @ vh[:usable].T
    if usable < dimensions:
        projected = np.pad(projected, ((0, 0), (0, dimensions - usable)))
    variances = singular**2
    fractions = variances / variances.sum() if variances.sum() else variances
    explained = np.pad(fractions[:usable], (0, dimensions - usable))
    return projected, explained


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, sufficient for small geometry diagnostics."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def matrix_rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation between upper triangles of two distance matrices."""
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2:
        return float("nan")
    indices = np.triu_indices(first.shape[0], k=1)
    x = _rank(first[indices])
    y = _rank(second[indices])
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _policies_for_all_goals(
    agent: MultiGoalGRUAgent,
) -> list[np.ndarray]:
    assert agent.z is not None
    policies = []
    with torch.no_grad():
        for goal_id, progress in enumerate(agent.all_goal_progress):
            progress = min(progress, agent.goals[goal_id].length)
            q_values = agent.network.q_values(agent.z, goal_id, progress)
            policy = torch.softmax(q_values / agent.policy_temperature, dim=-1)
            policies.append(policy.detach().cpu().numpy())
    return policies


def collect_geometry_diagnostics(
    agent: MultiGoalGRUAgent,
    env: QuantumEnvironment,
    goals: Sequence[GoalSpec],
    *,
    episodes_per_goal: int = 100,
    max_steps: int = 20,
) -> dict[str, object]:
    """Collect operational geometry from exploration-free held-out rollouts."""
    n_goals = len(goals)
    n_actions = env.n_actions
    feature_pairs = [
        (action, outcome)
        for action, count in enumerate(env.action_outcome_counts)
        for outcome in range(count)
    ]
    action_counts = np.zeros((n_goals, n_actions), dtype=float)
    pair_counts = np.zeros((n_goals, len(feature_pairs)), dtype=float)
    successes = np.zeros(n_goals, dtype=float)
    steps = np.zeros((n_goals, episodes_per_goal), dtype=float)
    success_steps: list[list[int]] = [[] for _ in goals]
    policy_sum = np.zeros((n_goals, n_goals), dtype=float)
    policy_samples = 0
    displacement_sum: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(n_goals, dtype=float)
    )
    displacement_count: dict[tuple[int, int], int] = defaultdict(int)

    pair_to_index = {pair: i for i, pair in enumerate(feature_pairs)}
    for goal_id, spec in enumerate(goals):
        for episode in range(episodes_per_goal):
            env.reset()
            tracker = GoalTracker(spec)
            agent.reset_episode(goal_id, spec.length, training=False)
            finished = False
            for t in range(max_steps):
                policies = _policies_for_all_goals(agent)
                for i in range(n_goals):
                    for j in range(i + 1, n_goals):
                        distance = jensen_shannon_distance(policies[i], policies[j])
                        policy_sum[i, j] += distance
                        policy_sum[j, i] += distance
                policy_samples += 1

                before = agent.distances_to_all_goals()
                progress = tracker.progress
                action = agent.act(goal_id, progress, training=False)
                outcome = env.step(action)
                _, done = tracker.update(action, outcome)
                agent.observe(
                    goal_id,
                    progress,
                    action,
                    outcome,
                    0.0,
                    done,
                    tracker.progress,
                    False,
                )
                after = agent.distances_to_all_goals()
                pair = (action, outcome)
                displacement_sum[pair] += after - before
                displacement_count[pair] += 1
                action_counts[goal_id, action] += 1
                pair_counts[goal_id, pair_to_index[pair]] += 1
                if done:
                    successes[goal_id] += 1
                    steps[goal_id, episode] = t + 1
                    success_steps[goal_id].append(t + 1)
                    finished = True
                    break
            if not finished:
                steps[goal_id, episode] = max_steps
            agent.end_episode(training=False)

    strategy_distance = policy_sum / max(policy_samples, 1)
    action_denominator = np.maximum(action_counts.sum(axis=1, keepdims=True), 1.0)
    pair_denominator = np.maximum(pair_counts.sum(axis=1, keepdims=True), 1.0)
    # Include duration and success because complete strategies differ in risk
    # and efficiency as well as in intervention frequency.
    trajectory_features = np.concatenate(
        [
            action_counts / action_denominator,
            pair_counts / pair_denominator,
            steps.mean(axis=1, keepdims=True) / max_steps,
            (successes / episodes_per_goal)[:, None],
        ],
        axis=1,
    )
    trajectory_distance = pairwise_euclidean(trajectory_features)

    embeddings = agent.goal_embeddings()
    embedding_distance = pairwise_euclidean(embeddings)
    projection, explained = pca_projection(embeddings)
    agent.reset_episode(0, goals[0].length, training=False)
    predicted_reachability = agent.distances_to_all_goals()
    reachability = []
    curves = []
    for goal_id, spec in enumerate(goals):
        empirical = (
            float(np.mean(success_steps[goal_id]))
            if success_steps[goal_id]
            else float("nan")
        )
        success_rate = float(successes[goal_id] / episodes_per_goal)
        reachability.append(
            {
                "goal_id": goal_id,
                "goal_name": spec.name,
                "predicted_steps": float(predicted_reachability[goal_id]),
                "empirical_steps_success": empirical,
                "success_rate": success_rate,
            }
        )
        observed = np.asarray(success_steps[goal_id])
        for horizon in range(1, max_steps + 1):
            curves.append(
                {
                    "goal_id": goal_id,
                    "goal_name": spec.name,
                    "horizon": horizon,
                    "success_probability": float(np.sum(observed <= horizon) / episodes_per_goal),
                }
            )

    displacements = []
    for action, outcome in feature_pairs:
        count = displacement_count[(action, outcome)]
        mean = displacement_sum[(action, outcome)] / max(count, 1)
        for goal_id, spec in enumerate(goals):
            displacements.append(
                {
                    "action": action,
                    "action_name": env.action_names[action],
                    "outcome": outcome,
                    "goal_id": goal_id,
                    "goal_name": spec.name,
                    "mean_delta_distance": float(mean[goal_id]),
                    "samples": count,
                }
            )

    feature_names = (
        [f"action:{name}" for name in env.action_names]
        + [f"pair:{env.action_names[a]}:{o}" for a, o in feature_pairs]
        + ["normalized_steps", "success_rate"]
    )
    return {
        "goal_names": [goal.name for goal in goals],
        "embeddings": embeddings,
        "embedding_projection": projection,
        "embedding_explained_variance": explained,
        "embedding_distance": embedding_distance,
        "strategy_distance": strategy_distance,
        "trajectory_features": trajectory_features,
        "trajectory_feature_names": feature_names,
        "trajectory_distance": trajectory_distance,
        "reachability": reachability,
        "reachability_curves": curves,
        "displacements": displacements,
        "embedding_strategy_spearman": matrix_rank_correlation(
            embedding_distance, strategy_distance
        ),
        "embedding_trajectory_spearman": matrix_rank_correlation(
            embedding_distance, trajectory_distance
        ),
    }


def _write_matrix(path: Path, names: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["goal"] + list(names))
        for name, row in zip(names, matrix):
            writer.writerow([name] + row.tolist())


def _write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_geometry_diagnostics(directory: str | Path, diagnostics: dict[str, object]) -> None:
    """Write a stable, plotting-friendly geometry artifact bundle."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = diagnostics["goal_names"]
    assert isinstance(names, list)
    embeddings = np.asarray(diagnostics["embeddings"])
    projection = np.asarray(diagnostics["embedding_projection"])
    with (directory / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["goal_id", "goal_name", "pc1", "pc2"]
            + [f"e{i}" for i in range(embeddings.shape[1])]
        )
        for i, name in enumerate(names):
            writer.writerow([i, name, *projection[i].tolist(), *embeddings[i].tolist()])
    for filename, key in (
        ("embedding_distances.csv", "embedding_distance"),
        ("strategy_distances.csv", "strategy_distance"),
        ("trajectory_distances.csv", "trajectory_distance"),
    ):
        _write_matrix(directory / filename, names, np.asarray(diagnostics[key]))
    features = np.asarray(diagnostics["trajectory_features"])
    feature_names = diagnostics["trajectory_feature_names"]
    assert isinstance(feature_names, list)
    with (directory / "trajectory_features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["goal_id", "goal_name"] + feature_names)
        for i, name in enumerate(names):
            writer.writerow([i, name] + features[i].tolist())
    _write_dict_rows(directory / "reachability.csv", diagnostics["reachability"])
    _write_dict_rows(directory / "reachability_curves.csv", diagnostics["reachability_curves"])
    _write_dict_rows(directory / "intervention_displacements.csv", diagnostics["displacements"])
    summary = {
        "embedding_strategy_spearman": diagnostics["embedding_strategy_spearman"],
        "embedding_trajectory_spearman": diagnostics["embedding_trajectory_spearman"],
        "pca_explained_variance": np.asarray(
            diagnostics["embedding_explained_variance"]
        ).tolist(),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

