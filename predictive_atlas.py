"""Learn a spatial atlas from weak quantum beacons and delayed landmarks.

This experiment removes exact online place observations from the positive
spatial construction.  Movement reports only success/failure.  Four binary QND
instruments have overlapping place-dependent response fields, so no individual
outcome identifies a site.  A GRU learns to integrate a scan history and
predict the outcome of a common terminal landmark probe.  A separately learned
action model then supports belief-state planning toward arbitrary landmark
goals.

The terminal probe is a commitment: during navigation its outcome ends the
episode, so an exact place label can train the predictive atlas but can never
be used for a later action in that episode.  Concealed coordinates and density
matrices are used only for offline validation and plotting.

Production run::

    conda run -n qbist_spacetime python predictive_atlas.py \
      --output results/predictive-atlas
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum_environments import QuantumEnvironment, environment_definition
from spatial_hodology import (
    COORDINATES,
    N_SITES,
    distance_correlation,
    exact_movement_costs,
    geometry_metrics,
    metric_mds,
    pairwise_distances,
)


GRID_SIZE = 3
N_MOVES = 8
BEACON_ACTIONS = tuple(range(8, 12))
TERMINAL_PROBE = 12
PLACE_NAMES = tuple("ABCDEFGHI")
DIAGONAL_SUCCESS = 0.715


@dataclass
class LocalizationSummary:
    condition: str
    seed: int
    cycles_seen: int
    test_accuracy: float
    bayes_accuracy: float
    negative_log_likelihood: float
    brier_score: float
    mean_confidence: float
    mean_entropy: float


@dataclass
class NavigationSummary:
    condition: str
    seed: int
    all_pairs_success: float
    reset_success: float
    mean_movement_cost: float
    mean_total_interventions: float
    transition_tv_error: float
    directionality: float
    exact_cost_correlation: float
    stress_1d: float
    stress_2d: float
    stress_3d: float
    positive_variance_2d: float
    negative_spectrum_fraction: float
    coordinate_procrustes_r2: float
    euclidean_distance_correlation: float


def _workspace_path(path: str) -> Path:
    workspace = Path(__file__).resolve().parent
    candidate = (
        (workspace / path).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"output must stay inside {workspace}")
    return candidate


def _set_site(env: QuantumEnvironment, site: int) -> None:
    """Privileged episode preparation; the site is never passed to the agent."""
    env.initial_state = f"site-{int(site)}"
    env.reset()


def _true_site(env: QuantumEnvironment) -> int:
    """Privileged validation decoder, forbidden to the online controller."""
    rho = env._rho
    if rho is None:
        raise RuntimeError("environment has no active state")
    return int(np.argmax(np.diag(rho).real))


def beacon_fields(environment: str = "qudit-grid-3x3-beacons") -> np.ndarray:
    definition = environment_definition(environment, weak_q=DIAGONAL_SUCCESS)
    fields = []
    for measurement in definition.measurements[8:12]:
        operator = measurement.outcome_kraus[1][0]
        fields.append(np.diag(operator.conj().T @ operator).real)
    return np.asarray(fields, dtype=float)


def scan_environment(
    env: QuantumEnvironment,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions: list[int] = []
    outcomes: list[int] = []
    for _ in range(int(cycles)):
        for action in BEACON_ACTIONS:
            actions.append(action)
            outcomes.append(env.step(action))
    return np.asarray(actions, dtype=np.int64), np.asarray(outcomes, dtype=np.int64)


def collect_localization_dataset(
    *,
    environment: str,
    cycles: int,
    samples_per_site: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect weak scans labeled only by a delayed terminal-probe experience."""
    env = QuantumEnvironment(
        environment=environment,
        weak_q=DIAGONAL_SUCCESS,
        seed=seed,
    )
    actions: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(seed + 31)
    schedule = np.repeat(np.arange(N_SITES), samples_per_site)
    rng.shuffle(schedule)
    for site in schedule:
        _set_site(env, int(site))
        scan_actions, scan_outcomes = scan_environment(env, cycles)
        terminal_label = env.step(TERMINAL_PROBE)
        actions.append(scan_actions)
        outcomes.append(scan_outcomes)
        labels.append(terminal_label)
    return np.stack(actions), np.stack(outcomes), np.asarray(labels, dtype=np.int64)


class BeaconGRULocalizer(nn.Module):
    """Predict the terminal landmark from a sequence of weak binary outcomes."""

    def __init__(self, embedding_dim: int = 12, hidden_dim: int = 48):
        super().__init__()
        self.token_embedding = nn.Embedding(len(BEACON_ACTIONS) * 2, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, N_SITES)

    def forward(self, actions: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        local_actions = actions - BEACON_ACTIONS[0]
        tokens = 2 * local_actions + outcomes
        encoded = self.token_embedding(tokens)
        _, hidden = self.gru(encoded)
        return self.classifier(hidden[-1])


def _select_history(
    actions: np.ndarray,
    outcomes: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "last-cycle":
        return actions[:, -len(BEACON_ACTIONS) :], outcomes[:, -len(BEACON_ACTIONS) :]
    return actions, outcomes


def _probabilities(
    model: BeaconGRULocalizer,
    actions: np.ndarray,
    outcomes: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            a = torch.as_tensor(actions[start : start + batch_size], device=device)
            o = torch.as_tensor(outcomes[start : start + batch_size], device=device)
            rows.append(torch.softmax(model(a, o), dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def bayes_scan_probabilities(
    outcomes: np.ndarray,
    fields: np.ndarray,
) -> np.ndarray:
    """Privileged Bayes ceiling using the true beacon likelihoods."""
    cycles = outcomes.shape[1] // len(BEACON_ACTIONS)
    reshaped = outcomes.reshape(len(outcomes), cycles, len(BEACON_ACTIONS))
    log_likelihood = np.zeros((len(outcomes), N_SITES), dtype=float)
    clipped = np.clip(fields, 1e-8, 1.0 - 1e-8)
    for site in range(N_SITES):
        q = clipped[:, site][None, None, :]
        log_likelihood[:, site] = (
            reshaped * np.log(q) + (1 - reshaped) * np.log(1.0 - q)
        ).sum(axis=(1, 2))
    log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
    probabilities = np.exp(log_likelihood)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def localization_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(N_SITES)[labels]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    confusion = np.zeros((N_SITES, N_SITES), dtype=float)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1.0
    confusion /= np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "negative_log_likelihood": float(-np.log(chosen).mean()),
        "brier_score": float(np.sum((probabilities - one_hot) ** 2, axis=1).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float(entropy.mean()),
    }, confusion


def train_localizer(
    *,
    condition: str,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[BeaconGRULocalizer, list[dict[str, float]], dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_actions, train_outcomes, train_labels = train_data
    test_actions, test_outcomes, test_labels = test_data
    train_actions, train_outcomes = _select_history(
        train_actions, train_outcomes, condition
    )
    test_actions, test_outcomes = _select_history(test_actions, test_outcomes, condition)

    model = BeaconGRULocalizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(
        torch.as_tensor(train_actions),
        torch.as_tensor(train_outcomes),
        torch.as_tensor(train_labels),
    )
    generator = torch.Generator().manual_seed(seed + 909)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)
    curves: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for actions, outcomes, labels in loader:
            actions = actions.to(device)
            outcomes = outcomes.to(device)
            labels = labels.to(device)
            loss = nn.functional.cross_entropy(model(actions, outcomes), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        test_probabilities = _probabilities(
            model, test_actions, test_outcomes, device=device
        )
        test_metrics, _ = localization_metrics(test_probabilities, test_labels)
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "test_accuracy": test_metrics["accuracy"],
                "test_nll": test_metrics["negative_log_likelihood"],
            }
        )
    probabilities = _probabilities(model, test_actions, test_outcomes, device=device)
    metrics, confusion = localization_metrics(probabilities, test_labels)
    return model, curves, metrics, confusion


def learned_transition_model(
    *,
    trials_per_source_action: int,
    seed: int,
    environment: str = "qudit-grid-3x3-beacons",
) -> np.ndarray:
    """Survey landmark-to-landmark dynamics without using coordinates.

    Each trial begins at a previously verified landmark label, applies one
    blind movement, observes success/failure, and ends with a landmark probe.
    Counts estimate P(outcome,destination | source, action).
    """
    env = QuantumEnvironment(
        environment=environment,
        weak_q=DIAGONAL_SUCCESS,
        seed=seed,
    )
    counts = np.full((N_MOVES, 2, N_SITES, N_SITES), 0.05, dtype=float)
    for source in range(N_SITES):
        for action in range(N_MOVES):
            for _ in range(trials_per_source_action):
                _set_site(env, source)
                outcome = env.step(action)
                destination = env.step(TERMINAL_PROBE)
                counts[action, outcome, source, destination] += 1.0
    return counts / counts.sum(axis=(1, 3), keepdims=True)


def exact_transition_joint() -> np.ndarray:
    directions = (
        (0, -1), (1, 0), (0, 1), (-1, 0),
        (1, -1), (1, 1), (-1, 1), (-1, -1),
    )
    joint = np.zeros((N_MOVES, 2, N_SITES, N_SITES), dtype=float)
    for action, (dx, dy) in enumerate(directions):
        for source in range(N_SITES):
            x, y = source % GRID_SIZE, source // GRID_SIZE
            nx, ny = x + dx, y + dy
            legal = 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
            p = (1.0 if action < 4 else DIAGONAL_SUCCESS) if legal else 0.0
            destination = ny * GRID_SIZE + nx if legal else source
            joint[action, 0, source, destination] += p
            joint[action, 1, source, source] += 1.0 - p
    return joint


def transition_tv_error(learned: np.ndarray) -> float:
    exact = exact_transition_joint()
    return float(0.5 * np.abs(learned - exact).sum(axis=(1, 3)).mean())


def planning_values(
    transition_joint: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic-shortest-path values and Q costs for every goal."""
    transitions = transition_joint.sum(axis=1)
    values = np.zeros((N_SITES, N_SITES), dtype=float)
    q_values = np.zeros((N_SITES, N_SITES, N_MOVES), dtype=float)
    for goal in range(N_SITES):
        value = np.full(N_SITES, 4.0, dtype=float)
        value[goal] = 0.0
        for _ in range(max_iterations):
            q = 1.0 + np.einsum("asd,d->sa", transitions, value)
            updated = q.min(axis=1)
            updated[goal] = 0.0
            if np.max(np.abs(updated - value)) < tolerance:
                value = updated
                break
            value = updated
        values[:, goal] = value
        q_values[:, goal, :] = 1.0 + np.einsum("asd,d->sa", transitions, value)
        q_values[goal, goal, :] = 0.0
    return values, q_values


def infer_scan(
    model: BeaconGRULocalizer,
    actions: np.ndarray,
    outcomes: np.ndarray,
    *,
    condition: str,
    device: torch.device,
) -> np.ndarray:
    selected_actions, selected_outcomes = _select_history(
        actions[None, :], outcomes[None, :], condition
    )
    return _probabilities(
        model, selected_actions, selected_outcomes, device=device, batch_size=1
    )[0]


def update_belief(
    belief: np.ndarray,
    transition_joint: np.ndarray,
    action: int,
    outcome: int,
) -> np.ndarray:
    predicted = belief @ transition_joint[action, outcome]
    total = predicted.sum()
    if total <= 1e-12:
        return np.full(N_SITES, 1.0 / N_SITES)
    return predicted / total


def evaluate_atlas(
    *,
    condition: str,
    environment: str,
    localizer: BeaconGRULocalizer | None,
    transition_joint: np.ndarray,
    q_values: np.ndarray,
    cycles: int,
    episodes_per_pair: int,
    max_moves: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    env = QuantumEnvironment(
        environment=environment,
        weak_q=DIAGONAL_SUCCESS,
        seed=seed,
    )
    costs = np.zeros((N_SITES, N_SITES), dtype=float)
    successes = np.zeros_like(costs)
    total_interventions = []
    trajectories: list[dict[str, object]] = []
    for source in range(N_SITES):
        for goal in range(N_SITES):
            pair_costs = []
            pair_success = []
            for trial in range(episodes_per_pair):
                _set_site(env, source)
                actions, outcomes = scan_environment(env, cycles)
                if condition == "oracle":
                    belief = np.eye(N_SITES)[source].copy()
                else:
                    if localizer is None:
                        raise ValueError("non-oracle evaluation needs a localizer")
                    belief = infer_scan(
                        localizer,
                        actions,
                        outcomes,
                        condition=condition,
                        device=device,
                    )
                true_path = [source]
                map_path = [int(np.argmax(belief))]
                confidence_path = [float(belief.max())]
                movement_actions: list[int] = []
                movement_outcomes: list[int] = []
                for _ in range(max_moves):
                    if int(np.argmax(belief)) == goal:
                        break
                    expected_q = np.einsum("s,sa->a", belief, q_values[:, goal, :])
                    action = int(np.argmin(expected_q))
                    outcome = env.step(action)
                    belief = update_belief(belief, transition_joint, action, outcome)
                    movement_actions.append(action)
                    movement_outcomes.append(outcome)
                    true_path.append(_true_site(env))
                    map_path.append(int(np.argmax(belief)))
                    confidence_path.append(float(belief.max()))
                terminal_outcome = env.step(TERMINAL_PROBE)
                success = terminal_outcome == goal
                restricted_cost = len(movement_actions) if success else max_moves
                pair_costs.append(restricted_cost)
                pair_success.append(float(success))
                total_interventions.append(
                    len(actions) + len(movement_actions) + 1
                )
                if trial == 0 and source == 4 and goal in (0, 2, 6, 8):
                    trajectories.append(
                        {
                            "condition": condition,
                            "source": source,
                            "goal": goal,
                            "success": bool(success),
                            "true_path": true_path,
                            "map_path": map_path,
                            "confidence": confidence_path,
                            "actions": movement_actions,
                            "outcomes": movement_outcomes,
                            "terminal_outcome": terminal_outcome,
                        }
                    )
            costs[source, goal] = float(np.mean(pair_costs))
            successes[source, goal] = float(np.mean(pair_success))
    np.fill_diagonal(costs, 0.0)
    return costs, successes, float(np.mean(total_interventions)), trajectories


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def plot_beacons_and_confusions(
    output: Path,
    fields: np.ndarray,
    confusions: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    names = ("field 0", "field 1", "field 2", "field 3")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for index, (axis, field, name) in enumerate(zip(axes[0], fields, names)):
        image = axis.imshow(field.reshape(3, 3), vmin=0, vmax=1, cmap="viridis")
        for site, value in enumerate(field):
            axis.text(site % 3, site // 3, f"{PLACE_NAMES[site]}\n{value:.2f}",
                      ha="center", va="center", color="white" if value < .35 else "black")
        axis.set_title(f"weak beacon {index}")
        axis.set_xticks([]); axis.set_yticks([])
    fig.colorbar(image, ax=axes[0].tolist(), fraction=.02, pad=.02,
                 label="outcome-one probability")
    for axis, condition in zip(axes[1], ("full-history", "last-cycle", "null", "oracle")):
        if condition == "oracle":
            matrix = np.eye(N_SITES)
        else:
            matrix = confusions[condition]
        axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        axis.set_title(condition.replace("-", " "))
        axis.set_xlabel("predicted landmark")
        axis.set_ylabel("terminal landmark")
        axis.set_xticks(range(9), PLACE_NAMES, fontsize=7)
        axis.set_yticks(range(9), PLACE_NAMES, fontsize=7)
    fig.suptitle("Overlapping quantum beacon fields and learned localization")
    fig.tight_layout(rect=(0, 0, .98, .95))
    fig.savefig(output / "beacon_fields_and_confusions.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(output: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    conditions = ("full-history", "last-cycle", "null")
    colors = ("#2962a8", "#e07a1f", "#7b4ab5")
    for condition, color in zip(conditions, colors):
        epochs = sorted({int(row["epoch"]) for row in rows if row["condition"] == condition})
        accuracy = [
            np.mean([float(row["test_accuracy"]) for row in rows
                     if row["condition"] == condition and int(row["epoch"]) == epoch])
            for epoch in epochs
        ]
        loss = [
            np.mean([float(row["train_loss"]) for row in rows
                     if row["condition"] == condition and int(row["epoch"]) == epoch])
            for epoch in epochs
        ]
        axes[0].plot(epochs, accuracy, label=condition, color=color)
        axes[1].plot(epochs, loss, label=condition, color=color)
    axes[0].axhline(1 / 9, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0].set(ylabel="held-out landmark accuracy", xlabel="epoch", ylim=(0, 1.02))
    axes[1].set(ylabel="cross-entropy loss", xlabel="epoch")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Delayed terminal outcomes train a predictive localization memory")
    fig.tight_layout()
    fig.savefig(output / "localization_learning_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_navigation_summary(output: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    conditions = ("oracle", "full-history", "last-cycle", "null")
    labels = ("oracle map", "full history", "last cycle", "null fields")
    colors = ("#333333", "#2962a8", "#e07a1f", "#7b4ab5")
    metrics = (
        ("all_pairs_success", "all-pairs success", (0, 1.04)),
        ("stress_2d", "learned 2D stress", (0, None)),
        ("coordinate_procrustes_r2", "coordinate Procrustes $R^2$", (0, 1.04)),
        ("exact_cost_correlation", "exact-cost correlation", (0, 1.04)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for axis, (metric, title, limits) in zip(axes, metrics):
        means = []
        errors = []
        for condition in conditions:
            values = [float(row[metric]) for row in rows if row["condition"] == condition]
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        axis.bar(range(4), means, yerr=errors, color=colors, capsize=3)
        axis.set_title(title)
        axis.set_xticks(range(4), labels, rotation=25, ha="right", fontsize=8)
        if limits[1] is not None:
            axis.set_ylim(*limits)
        else:
            axis.set_ylim(bottom=limits[0])
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Predictive memory turns weak sensation into a navigable atlas")
    fig.tight_layout()
    fig.savefig(output / "predictive_atlas_performance.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_atlas_geometries(
    output: Path,
    costs: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    conditions = ("oracle", "full-history", "last-cycle", "null")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for axis, condition in zip(axes, conditions):
        matrix = (costs[condition] + costs[condition].T) / 2
        np.fill_diagonal(matrix, 0.0)
        coordinates = metric_mds(matrix, 2)
        stress = geometry_metrics(matrix)["stress_2d"]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=range(9), cmap="viridis", s=65)
        for site, (x, y) in enumerate(coordinates):
            axis.text(x, y, PLACE_NAMES[site], ha="center", va="center", fontsize=8)
        axis.set_title(f"{condition.replace('-', ' ')}\nstress={stress:.3f}")
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Coordinates reconstructed only from empirical goal difficulty")
    fig.tight_layout()
    fig.savefig(output / "predictive_atlas_geometries.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_belief_trajectories(
    output: Path,
    trajectories: list[dict[str, object]],
    full_cost: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    chosen = [row for row in trajectories if row["condition"] == "full-history"][:4]
    coordinates = metric_mds((full_cost + full_cost.T) / 2.0, 2)
    fig, axes = plt.subplots(1, len(chosen), figsize=(12.5, 3.3), squeeze=False)
    for axis, record in zip(axes[0], chosen):
        axis.scatter(coordinates[:, 0], coordinates[:, 1], color="#d6d6d6", s=35)
        true_path = np.asarray([coordinates[int(site)] for site in record["true_path"]])
        map_path = np.asarray([coordinates[int(site)] for site in record["map_path"]])
        axis.plot(true_path[:, 0], true_path[:, 1], "-o", color="#c23b32", label="privileged true")
        axis.plot(map_path[:, 0], map_path[:, 1], "--s", color="#2962a8", label="belief MAP")
        goal = int(record["goal"])
        axis.scatter(*coordinates[goal], marker="*", s=170, color="#2f9e44", zorder=5)
        for site, (x, y) in enumerate(coordinates):
            axis.text(x, y, PLACE_NAMES[site], ha="center", va="center", fontsize=7)
        axis.set_title(f"E → {PLACE_NAMES[goal]} | {'success' if record['success'] else 'failure'}")
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal", adjustable="datalim")
    if chosen:
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Belief-state policy trajectories in the learned atlas")
    fig.tight_layout()
    fig.savefig(output / "belief_state_trajectories.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = _workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    matrices = output / "matrices"
    models = output / "models"
    matrices.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    conditions = (
        ("full-history", "qudit-grid-3x3-beacons"),
        ("last-cycle", "qudit-grid-3x3-beacons"),
        ("null", "qudit-grid-3x3-null-beacons"),
    )
    localization_rows: list[dict[str, object]] = []
    navigation_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    all_trajectories: list[dict[str, object]] = []
    confusions_by_condition: dict[str, list[np.ndarray]] = {name: [] for name, _ in conditions}
    costs_by_condition: dict[str, list[np.ndarray]] = {
        name: [] for name in ("oracle", "full-history", "last-cycle", "null")
    }

    for seed in args.seeds:
        print(f"[seed {seed}] collecting informative weak-scan data", flush=True)
        informative_train = collect_localization_dataset(
            environment="qudit-grid-3x3-beacons",
            cycles=args.scan_cycles,
            samples_per_site=args.calibration_per_site,
            seed=seed + 1_000,
        )
        informative_test = collect_localization_dataset(
            environment="qudit-grid-3x3-beacons",
            cycles=args.scan_cycles,
            samples_per_site=args.test_per_site,
            seed=seed + 2_000,
        )
        null_train = collect_localization_dataset(
            environment="qudit-grid-3x3-null-beacons",
            cycles=args.scan_cycles,
            samples_per_site=args.calibration_per_site,
            seed=seed + 3_000,
        )
        null_test = collect_localization_dataset(
            environment="qudit-grid-3x3-null-beacons",
            cycles=args.scan_cycles,
            samples_per_site=args.test_per_site,
            seed=seed + 4_000,
        )
        localizers: dict[str, BeaconGRULocalizer] = {}
        for condition, environment in conditions:
            train_data, test_data = (
                (null_train, null_test) if condition == "null"
                else (informative_train, informative_test)
            )
            print(f"[seed {seed}] training {condition} localizer", flush=True)
            model, curves, metrics, confusion = train_localizer(
                condition=condition,
                train_data=train_data,
                test_data=test_data,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=seed + 10_000,
                device=device,
            )
            localizers[condition] = model
            torch.save(model.state_dict(), models / f"{condition}__seed{seed}.pt")
            selected_outcomes = _select_history(test_data[0], test_data[1], condition)[1]
            bayes = bayes_scan_probabilities(
                selected_outcomes,
                beacon_fields(environment),
            )
            bayes_accuracy = float(np.mean(bayes.argmax(axis=1) == test_data[2]))
            summary = LocalizationSummary(
                condition=condition,
                seed=seed,
                cycles_seen=(1 if condition == "last-cycle" else args.scan_cycles),
                test_accuracy=metrics["accuracy"],
                bayes_accuracy=bayes_accuracy,
                negative_log_likelihood=metrics["negative_log_likelihood"],
                brier_score=metrics["brier_score"],
                mean_confidence=metrics["mean_confidence"],
                mean_entropy=metrics["mean_entropy"],
            )
            localization_rows.append(asdict(summary))
            confusions_by_condition[condition].append(confusion)
            for row in curves:
                learning_rows.append({"condition": condition, "seed": seed, **row})
            _save_matrix(matrices / f"confusion__{condition}__seed{seed}.csv", confusion)

        print(f"[seed {seed}] surveying blind movement transitions", flush=True)
        transition_joint = learned_transition_model(
            trials_per_source_action=args.transition_trials,
            seed=seed + 20_000,
        )
        tv_error = transition_tv_error(transition_joint)
        _, q_values = planning_values(transition_joint)
        _save_matrix(
            matrices / f"transition_unconditional__seed{seed}.csv",
            transition_joint.sum(axis=1).reshape(N_MOVES * N_SITES, N_SITES),
        )

        for condition, environment in (
            ("oracle", "qudit-grid-3x3-beacons"), *conditions
        ):
            print(f"[seed {seed}] evaluating {condition} atlas", flush=True)
            cost, success, mean_total, trajectories = evaluate_atlas(
                condition=condition,
                environment=environment,
                localizer=localizers.get(condition),
                transition_joint=transition_joint,
                q_values=q_values,
                cycles=args.scan_cycles,
                episodes_per_pair=args.pair_episodes,
                max_moves=args.max_moves,
                seed=seed + 30_000 + len(navigation_rows),
                device=device,
            )
            metrics = geometry_metrics(cost)
            symmetric = (cost + cost.T) / 2.0
            exact = exact_movement_costs(DIAGONAL_SUCCESS)
            directionality = float(
                np.linalg.norm(cost - cost.T) /
                max(np.linalg.norm(cost + cost.T), 1e-12)
            )
            summary = NavigationSummary(
                condition=condition,
                seed=seed,
                all_pairs_success=float(success.mean()),
                reset_success=float(success[4].mean()),
                mean_movement_cost=float(cost.mean()),
                mean_total_interventions=mean_total,
                transition_tv_error=tv_error,
                directionality=directionality,
                exact_cost_correlation=distance_correlation(symmetric, exact),
                **metrics,
            )
            navigation_rows.append(asdict(summary))
            costs_by_condition[condition].append(cost)
            all_trajectories.extend({"seed": seed, **row} for row in trajectories)
            _save_matrix(matrices / f"cost__{condition}__seed{seed}.csv", cost)
            _save_matrix(matrices / f"success__{condition}__seed{seed}.csv", success)

    _write_csv(output / "localization_summary.csv", localization_rows)
    _write_csv(output / "navigation_summary.csv", navigation_rows)
    _write_csv(output / "localization_learning_curves.csv", learning_rows)
    (output / "trajectories.json").write_text(json.dumps(all_trajectories, indent=2))
    aggregate_confusions = {
        condition: np.mean(rows, axis=0)
        for condition, rows in confusions_by_condition.items()
    }
    aggregate_costs = {
        condition: np.mean(rows, axis=0)
        for condition, rows in costs_by_condition.items()
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "weak-QND-beacon predictive atlas",
        "diagonal_success": DIAGONAL_SUCCESS,
        "scan_cycles": args.scan_cycles,
        "calibration_per_site": args.calibration_per_site,
        "test_per_site": args.test_per_site,
        "epochs": args.epochs,
        "transition_trials_per_source_action": args.transition_trials,
        "pair_episodes": args.pair_episodes,
        "max_moves": args.max_moves,
        "seeds": args.seeds,
        "device": str(device),
        "agent_information": [
            "goal landmark label",
            "chosen movement and success/failure",
            "weak beacon action/outcome history",
            "terminal landmark outcome after commitment",
        ],
        "withheld_online": [
            "density matrix",
            "Kraus operators",
            "beacon likelihoods",
            "lattice coordinates",
            "exact current place label before commitment",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        plot_beacons_and_confusions(output, beacon_fields(), aggregate_confusions)
        plot_learning_curves(output, learning_rows)
        plot_navigation_summary(output, navigation_rows)
        plot_atlas_geometries(output, aggregate_costs)
        plot_belief_trajectories(output, all_trajectories, aggregate_costs["full-history"])
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print(
            "matplotlib is unavailable in this environment; run "
            f"python plot_predictive_atlas.py {output}"
        )
    print(json.dumps({
        "localization": localization_rows,
        "navigation": navigation_rows,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/predictive-atlas")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scan-cycles", type=int, default=12)
    parser.add_argument("--calibration-per-site", type=int, default=400)
    parser.add_argument("--test-per-site", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--transition-trials", type=int, default=100)
    parser.add_argument("--pair-episodes", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.scan_cycles < 1 or args.calibration_per_site < 1:
        raise ValueError("scan cycles and calibration samples must be positive")
    run(args)


if __name__ == "__main__":
    main()
