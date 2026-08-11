"""Create publication-quality plots from an experiment bundle.

This script has no PyTorch dependency.  It can therefore run in the base
environment after neural simulations finish in ``qbist_spacetime``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND_LABELS = {
    "tabular": "Finite history",
    "gru": "GRU",
    "multi-gru": "Predictive geometry GRU",
}
BACKEND_COLORS = {
    "tabular": "#3b82f6",
    "gru": "#f59e0b",
    "multi-gru": "#8b5cf6",
}


def _workspace_path(path: str) -> Path:
    workspace = Path(__file__).resolve().parent
    candidate = (
        (workspace / path).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"paths must stay inside {workspace}")
    return candidate


def _read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _scenario(row: dict[str, str]) -> str:
    return f"{row['environment']}\n{row['initial_state']}"


def _annotated_heatmap(
    axis,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    *,
    title: str,
    cmap: str = "viridis",
    value_format: str = ".2f",
    center: float | None = None,
):
    kwargs = {}
    if center is not None:
        limit = max(abs(np.nanmin(matrix) - center), abs(np.nanmax(matrix) - center), 1e-9)
        kwargs.update(vmin=center - limit, vmax=center + limit)
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, **kwargs)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=40, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    threshold = float(np.nanmean(matrix)) if np.isfinite(matrix).any() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "--" if not np.isfinite(value) else format(value, value_format)
            color = "white" if np.isfinite(value) and value > threshold else "black"
            axis.text(column, row, text, ha="center", va="center", fontsize=7, color=color)
    return image


def plot_performance(bundle: Path, plots: Path) -> None:
    summaries = [
        row
        for row in _read_dicts(bundle / "summary.csv")
        if row["phase"] == "evaluation" and row["goal_name"] == "OVERALL"
    ]
    scenarios = list(dict.fromkeys(_scenario(row) for row in summaries))
    backends = [
        backend
        for backend in BACKEND_LABELS
        if any(row["backend"] == backend for row in summaries)
    ]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_steps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summaries:
        key = (_scenario(row), row["backend"])
        grouped[key].append(float(row["success_rate"]))
        value = float(row["mean_steps_success"])
        if np.isfinite(value):
            grouped_steps[key].append(value)
    success = np.full((len(scenarios), len(backends)), np.nan)
    steps = np.full_like(success, np.nan)
    for i, scenario in enumerate(scenarios):
        for j, backend in enumerate(backends):
            if grouped[(scenario, backend)]:
                success[i, j] = np.mean(grouped[(scenario, backend)])
            if grouped_steps[(scenario, backend)]:
                steps[i, j] = np.mean(grouped_steps[(scenario, backend)])
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.65 * len(scenarios))))
    image = _annotated_heatmap(
        axes[0], success, scenarios, [BACKEND_LABELS[b] for b in backends],
        title="Evaluation success rate", cmap="YlGn", value_format=".0%",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.04, label="success rate")
    image = _annotated_heatmap(
        axes[1], steps, scenarios, [BACKEND_LABELS[b] for b in backends],
        title="Steps conditional on success", cmap="magma_r", value_format=".1f",
    )
    fig.colorbar(image, ax=axes[1], fraction=0.04, label="interventions")
    fig.suptitle("Quantum-control performance across intervention worlds", fontsize=15)
    fig.tight_layout()
    fig.savefig(plots / "performance_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(bundle: Path, plots: Path) -> None:
    rows = _read_dicts(bundle / "training_episodes.csv")
    scenarios = list(dict.fromkeys(_scenario(row) for row in rows))
    columns = 2
    plot_rows = int(np.ceil(len(scenarios) / columns))
    fig, axes = plt.subplots(plot_rows, columns, figsize=(12, 3.5 * plot_rows), squeeze=False)
    for axis, scenario in zip(axes.flat, scenarios):
        selected = [row for row in rows if _scenario(row) == scenario]
        for backend in BACKEND_LABELS:
            backend_rows = [row for row in selected if row["backend"] == backend]
            if not backend_rows:
                continue
            by_seed: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for row in backend_rows:
                success = 1.0 if row["success"].strip().lower() in {"1", "true"} else 0.0
                by_seed[row["seed"]].append((int(row["episode"]), success))
            maximum = max(episode for values in by_seed.values() for episode, _ in values)
            bins = min(25, maximum)
            edges = np.linspace(1, maximum + 1, bins + 1)
            curves = []
            for values in by_seed.values():
                values.sort()
                episodes = np.array([value[0] for value in values])
                successes = np.array([value[1] for value in values])
                curve = [
                    np.mean(successes[(episodes >= edges[i]) & (episodes < edges[i + 1])])
                    for i in range(bins)
                ]
                curves.append(curve)
            curves_array = np.asarray(curves)
            x = (edges[:-1] + edges[1:]) / 2.0
            mean = np.nanmean(curves_array, axis=0)
            sem = np.nanstd(curves_array, axis=0) / np.sqrt(len(curves_array))
            color = BACKEND_COLORS[backend]
            axis.plot(x, mean, label=BACKEND_LABELS[backend], color=color, linewidth=2)
            axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18)
        axis.set_title(scenario.replace("\n", " / "), loc="left", fontweight="bold")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("training episode")
        axis.set_ylabel("success / episode bin")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(scenarios):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.suptitle(
        "Learning curves: shared task distribution, independent seeds",
        fontsize=15,
        y=1.015,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plots / "learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return [row[0] for row in rows[1:]], np.array([[float(v) for v in row[1:]] for row in rows[1:]])


def plot_geometry_directory(directory: Path, plots: Path) -> None:
    destination = plots / directory.name
    destination.mkdir(exist_ok=True)
    embedding_rows = _read_dicts(directory / "embeddings.csv")
    names, embedding = _read_matrix(directory / "embedding_distances.csv")
    _, strategy = _read_matrix(directory / "strategy_distances.csv")
    _, trajectory = _read_matrix(directory / "trajectory_distances.csv")
    reachability = _read_dicts(directory / "reachability.csv")
    x = np.array([float(row["pc1"]) for row in embedding_rows])
    y = np.array([float(row["pc2"]) for row in embedding_rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes[0, 0].scatter(
        x,
        y,
        s=95,
        c=np.arange(len(names)),
        cmap="viridis",
        edgecolor="white",
        linewidth=1.2,
    )
    for i, name in enumerate(names):
        axes[0, 0].annotate(
            name, (x[i], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    axes[0, 0].axhline(0, color="0.8", linewidth=0.7)
    axes[0, 0].axvline(0, color="0.8", linewidth=0.7)
    axes[0, 0].set_title("Learned goal coordinates (PCA)", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("principal coordinate 1")
    axes[0, 0].set_ylabel("principal coordinate 2")
    _annotated_heatmap(
        axes[0, 1], embedding, names, names, title="Embedding distance", cmap="Blues"
    )
    _annotated_heatmap(
        axes[1, 0],
        strategy,
        names,
        names,
        title="Held-out strategy distance",
        cmap="Purples",
    )
    predicted = np.array([float(row["predicted_steps"]) for row in reachability])
    empirical = np.array([float(row["empirical_steps_success"]) for row in reachability])
    rates = np.array([float(row["success_rate"]) for row in reachability])
    finite = np.isfinite(empirical)
    scatter = axes[1, 1].scatter(
        predicted[finite],
        empirical[finite],
        c=rates[finite],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=90,
    )
    for i, name in enumerate(names):
        if finite[i]:
            axes[1, 1].annotate(
                name,
                (predicted[i], empirical[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    if finite.any():
        low = min(predicted[finite].min(), empirical[finite].min())
        high = max(predicted[finite].max(), empirical[finite].max())
        axes[1, 1].plot(
            [low, high],
            [low, high],
            linestyle="--",
            color="0.4",
            label="perfect calibration",
        )
    axes[1, 1].set_title("Directed reachability calibration", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("learned blank-history distance")
    axes[1, 1].set_ylabel("empirical steps (successful trials)")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, frameon=False, fontsize=8)
    fig.colorbar(scatter, ax=axes[1, 1], fraction=0.04, label="success rate")
    summary = json.loads((directory / "summary.json").read_text())
    fig.suptitle(
        f"Goal geometry: {directory.name}\n"
        f"rank corr(embedding, strategy)={summary['embedding_strategy_spearman']:.2f}; "
        f"corr(embedding, trajectory)={summary['embedding_trajectory_spearman']:.2f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(destination / "goal_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    features = _read_dicts(directory / "trajectory_features.csv")
    feature_names = list(features[0])[2:]
    feature_matrix = np.array([[float(row[name]) for name in feature_names] for row in features])
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(names) * 0.55)))
    image = _annotated_heatmap(
        axes[0], feature_matrix, names, feature_names,
        title="Trajectory strategy signatures", cmap="YlGnBu", value_format=".2f",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.035)
    _annotated_heatmap(
        axes[1],
        trajectory,
        names,
        names,
        title="Trajectory-feature distance",
        cmap="Oranges",
    )
    fig.tight_layout()
    fig.savefig(destination / "strategy_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    displacement_rows = _read_dicts(directory / "intervention_displacements.csv")
    pairs = list(
        dict.fromkeys(
            f"{row['action_name']}:{row['outcome']}" for row in displacement_rows
        )
    )
    displacement = np.full((len(pairs), len(names)), np.nan)
    for row in displacement_rows:
        pair = f"{row['action_name']}:{row['outcome']}"
        displacement[pairs.index(pair), names.index(row["goal_name"])] = float(
            row["mean_delta_distance"]
        )
    fig, axis = plt.subplots(figsize=(max(8, len(names) * 0.85), max(4, len(pairs) * 0.48)))
    image = _annotated_heatmap(
        axis, displacement, pairs, names,
        title=(
            "Intervention displacement in goal-distance coordinates\n"
            "negative = closer; positive = farther"
        ),
        cmap="coolwarm", value_format="+.2f", center=0.0,
    )
    fig.colorbar(image, ax=axis, fraction=0.035, label="mean change in learned distance")
    fig.tight_layout()
    fig.savefig(destination / "intervention_displacements.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    curve_rows = _read_dicts(directory / "reachability_curves.csv")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for name in names:
        selected = [row for row in curve_rows if row["goal_name"] == name]
        axis.step(
            [int(row["horizon"]) for row in selected],
            [float(row["success_probability"]) for row in selected],
            where="post",
            label=name,
            linewidth=1.8,
        )
    axis.set_title("Finite-time reachability from blank history", loc="left", fontweight="bold")
    axis.set_xlabel("intervention horizon T")
    axis.set_ylabel("P(goal achieved by T)")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.2)
    axis.legend(ncol=min(3, len(names)), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "reachability_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_validation(bundle: Path, plots: Path) -> None:
    """Summarize geometry agreement and cost calibration across all seeds."""
    directories = sorted(path for path in (bundle / "geometry").iterdir() if path.is_dir())
    labels = []
    strategy = []
    trajectory = []
    predicted = []
    empirical = []
    success = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        compact = directory.name.replace("__multi-gru__", "\n").replace("qubit-", "q-")
        labels.append(compact)
        strategy.append(summary["embedding_strategy_spearman"])
        trajectory.append(summary["embedding_trajectory_spearman"])
        for row in _read_dicts(directory / "reachability.csv"):
            empirical_value = float(row["empirical_steps_success"])
            if np.isfinite(empirical_value):
                predicted.append(float(row["predicted_steps"]))
                empirical.append(empirical_value)
                success.append(float(row["success_rate"]))
    positions = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(
        positions - width / 2,
        strategy,
        width,
        label="held-out policy strategy",
        color="#7c3aed",
    )
    axes[0].bar(
        positions + width / 2,
        trajectory,
        width,
        label="trajectory signatures",
        color="#f97316",
    )
    axes[0].axhline(0, color="0.25", linewidth=0.8)
    axes[0].set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("Spearman rank correlation")
    axes[0].set_title("Does embedding distance predict behavior?", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    scatter = axes[1].scatter(
        predicted,
        empirical,
        c=success,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        alpha=0.85,
    )
    low = min(min(predicted), min(empirical))
    high = max(max(predicted), max(empirical))
    axes[1].plot(
        [low, high],
        [low, high],
        linestyle="--",
        color="0.35",
        label="perfect calibration",
    )
    correlation = float(np.corrcoef(predicted, empirical)[0, 1])
    mae = float(np.mean(np.abs(np.asarray(predicted) - np.asarray(empirical))))
    axes[1].set_title(
        f"Cost-to-go calibration across goals\nr={correlation:.2f}, MAE={mae:.2f} interventions",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_xlabel("learned blank-history distance")
    axes[1].set_ylabel("empirical steps on successful trials")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[1], fraction=0.04, label="success rate")
    fig.suptitle("Validation of the learned goal geometry", fontsize=15)
    fig.tight_layout()
    fig.savefig(plots / "geometry_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Experiment output directory")
    parser.add_argument("--output", default=None, help="Defaults to BUNDLE/plots")
    args = parser.parse_args()
    bundle = _workspace_path(args.bundle)
    plots = _workspace_path(args.output) if args.output else bundle / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_performance(bundle, plots)
    plot_learning_curves(bundle, plots)
    plot_geometry_validation(bundle, plots)
    geometry_root = bundle / "geometry"
    for directory in sorted(path for path in geometry_root.iterdir() if path.is_dir()):
        plot_geometry_directory(directory, plots)
    print(f"saved plots to {plots}")


if __name__ == "__main__":
    main()
