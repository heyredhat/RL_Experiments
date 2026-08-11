"""Render the predictive-atlas result bundle without requiring PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantum_environments import environment_definition
from spatial_hodology import geometry_metrics, metric_mds


CONDITIONS = ("oracle", "full-history", "last-cycle", "null")
PLACE_NAMES = tuple("ABCDEFGHI")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fields() -> np.ndarray:
    definition = environment_definition("qudit-grid-3x3-beacons", weak_q=0.715)
    rows = []
    for measurement in definition.measurements[8:12]:
        operator = measurement.outcome_kraus[1][0]
        rows.append(np.diag(operator.conj().T @ operator).real)
    return np.asarray(rows)


def _mean_matrix(root: Path, stem: str, condition: str, seeds: list[int]) -> np.ndarray:
    return np.mean(
        [np.loadtxt(root / "matrices" / f"{stem}__{condition}__seed{seed}.csv", delimiter=",")
         for seed in seeds],
        axis=0,
    )


def beacon_figure(root: Path, seeds: list[int]) -> None:
    fields = _fields()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), layout="constrained")
    for index, (axis, field) in enumerate(zip(axes[0], fields)):
        image = axis.imshow(field.reshape(3, 3), vmin=0, vmax=1, cmap="viridis")
        for site, value in enumerate(field):
            axis.text(site % 3, site // 3, f"{PLACE_NAMES[site]}\n{value:.2f}",
                      ha="center", va="center",
                      color="white" if value < .35 else "black")
        axis.set_title(f"weak beacon {index}")
        axis.set_xticks([]); axis.set_yticks([])
    fig.colorbar(image, ax=axes[0].tolist(), fraction=.02, pad=.025, shrink=.82,
                 label="outcome-one probability")
    for axis, condition in zip(axes[1], CONDITIONS):
        matrix = (
            np.eye(9) if condition == "oracle"
            else _mean_matrix(root, "confusion", condition, seeds)
        )
        axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        axis.set_title(condition.replace("-", " "))
        axis.set_xlabel("predicted landmark")
        axis.set_ylabel("terminal landmark")
        axis.set_xticks(range(9), PLACE_NAMES, fontsize=7)
        axis.set_yticks(range(9), PLACE_NAMES, fontsize=7)
    fig.suptitle("Overlapping quantum beacon fields and learned localization")
    fig.savefig(root / "beacon_fields_and_confusions.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def learning_figure(root: Path) -> None:
    rows = _rows(root / "localization_learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    colors = {"full-history": "#2962a8", "last-cycle": "#e07a1f", "null": "#7b4ab5"}
    for condition, color in colors.items():
        epochs = sorted({int(row["epoch"]) for row in rows if row["condition"] == condition})
        accuracy = [np.mean([float(row["test_accuracy"]) for row in rows
                            if row["condition"] == condition and int(row["epoch"]) == e])
                    for e in epochs]
        loss = [np.mean([float(row["train_loss"]) for row in rows
                        if row["condition"] == condition and int(row["epoch"]) == e])
                for e in epochs]
        axes[0].plot(epochs, accuracy, label=condition, color=color)
        axes[1].plot(epochs, loss, label=condition, color=color)
    axes[0].axhline(1 / 9, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0].set(xlabel="epoch", ylabel="held-out landmark accuracy", ylim=(0, 1.02))
    axes[1].set(xlabel="epoch", ylabel="cross-entropy loss")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=.2)
    fig.suptitle("Delayed terminal outcomes train a predictive localization memory")
    fig.tight_layout()
    fig.savefig(root / "localization_learning_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def performance_figure(root: Path) -> None:
    rows = _rows(root / "navigation_summary.csv")
    labels = ("oracle map", "full history", "last cycle", "null fields")
    colors = ("#333333", "#2962a8", "#e07a1f", "#7b4ab5")
    metrics = (
        ("all_pairs_success", "all-pairs success", (0, 1.04)),
        ("stress_2d", "learned 2D stress", (0, None)),
        ("coordinate_procrustes_r2", "coordinate Procrustes $R^2$", (0, 1.04)),
        ("exact_cost_correlation", "exact-cost correlation", (-.1, 1.04)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for axis, (metric, title, limits) in zip(axes, metrics):
        grouped = [[float(row[metric]) for row in rows if row["condition"] == condition]
                   for condition in CONDITIONS]
        means = [np.mean(values) for values in grouped]
        errors = [np.std(values, ddof=1) if len(values) > 1 else 0 for values in grouped]
        axis.bar(range(4), means, yerr=errors, color=colors, capsize=3)
        axis.set_title(title)
        axis.set_xticks(range(4), labels, rotation=25, ha="right", fontsize=8)
        axis.set_ylim(bottom=limits[0], top=limits[1])
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Predictive memory turns weak sensation into a navigable atlas")
    fig.tight_layout()
    fig.savefig(root / "predictive_atlas_performance.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def geometry_figure(root: Path, seeds: list[int]) -> dict[str, np.ndarray]:
    costs = {condition: _mean_matrix(root, "cost", condition, seeds) for condition in CONDITIONS}
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for axis, condition in zip(axes, CONDITIONS):
        matrix = (costs[condition] + costs[condition].T) / 2
        np.fill_diagonal(matrix, 0)
        coordinates = metric_mds(matrix, 2)
        stress = geometry_metrics(matrix)["stress_2d"]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=range(9), cmap="viridis", s=65)
        for site, (x, y) in enumerate(coordinates):
            axis.text(x, y, PLACE_NAMES[site], ha="center", va="center", fontsize=8)
        axis.set_title(f"{condition.replace('-', ' ')}\nstress={stress:.3f}")
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Coordinates reconstructed only from empirical goal difficulty")
    fig.tight_layout()
    fig.savefig(root / "predictive_atlas_geometries.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return costs


def trajectory_figure(root: Path, costs: dict[str, np.ndarray]) -> None:
    trajectories = json.loads((root / "trajectories.json").read_text())
    chosen = [row for row in trajectories if row["condition"] == "full-history"][:4]
    coordinates = metric_mds((costs["full-history"] + costs["full-history"].T) / 2, 2)
    fig, axes = plt.subplots(1, max(len(chosen), 1), figsize=(12.5, 3.3), squeeze=False)
    for axis, record in zip(axes[0], chosen):
        axis.scatter(coordinates[:, 0], coordinates[:, 1], color="#d6d6d6", s=35)
        true_path = np.asarray([coordinates[int(site)] for site in record["true_path"]])
        map_path = np.asarray([coordinates[int(site)] for site in record["map_path"]])
        axis.plot(true_path[:, 0], true_path[:, 1], "-o", color="#c23b32", label="offline truth")
        axis.plot(map_path[:, 0], map_path[:, 1], "--s", color="#2962a8", label="belief MAP")
        goal = int(record["goal"])
        axis.scatter(*coordinates[goal], marker="*", s=170, color="#2f9e44", zorder=5)
        for site, (x, y) in enumerate(coordinates):
            axis.text(x, y, PLACE_NAMES[site], ha="center", va="center", fontsize=7)
        axis.set_title(f"E → {PLACE_NAMES[goal]} | {'success' if record['success'] else 'failure'}")
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal", adjustable="datalim")
    if chosen:
        axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Belief-state policy trajectories in the learned atlas")
    fig.tight_layout()
    fig.savefig(root / "belief_state_trajectories.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    args = parser.parse_args()
    root = Path(args.bundle).resolve()
    if not (root / "manifest.json").exists():
        raise ValueError(f"not a predictive-atlas bundle: {root}")
    manifest = json.loads((root / "manifest.json").read_text())
    seeds = [int(value) for value in manifest["seeds"]]
    beacon_figure(root, seeds)
    learning_figure(root)
    performance_figure(root)
    costs = geometry_figure(root, seeds)
    trajectory_figure(root, costs)
    print(f"wrote predictive-atlas figures to {root}")


if __name__ == "__main__":
    main()
