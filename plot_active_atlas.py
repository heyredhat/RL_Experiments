"""Render the active-atlas result bundle with only NumPy and Matplotlib."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_SITES = 9
PLACE_NAMES = tuple("ABCDEFGHI")
COORDINATES = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)
CONDITIONS = (
    "oracle",
    "active",
    "active-exact-sensors",
    "fixed-12",
    "entropy",
    "active-null",
)
COLORS = {
    "oracle": "#303030",
    "active": "#1769aa",
    "active-exact-sensors": "#36a2ae",
    "fixed-12": "#dd8d29",
    "entropy": "#7b4ab5",
    "active-null": "#a7a7a7",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]], field: str) -> dict[str, tuple[float, float]]:
    result = {}
    for condition in CONDITIONS:
        values = np.array([float(row[field]) for row in rows if row["condition"] == condition])
        if len(values):
            result[condition] = (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return result


def classical_mds(distance: np.ndarray, dimensions: int = 2) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    centering = np.eye(len(distance)) - np.ones_like(distance) / len(distance)
    gram = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order[:dimensions]], 0.0)
    return eigenvectors[:, order[:dimensions]] * np.sqrt(positive)


def aggregate_matrix(bundle: Path, name: str, condition: str) -> np.ndarray:
    paths = sorted((bundle / "matrices").glob(f"{name}__{condition}__seed*.csv"))
    return np.mean([np.loadtxt(path, delimiter=",") for path in paths], axis=0)


def align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = source - source.mean(axis=0)
    y = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotated = x @ (u @ vt)
    scale = np.sum(rotated * y) / max(np.sum(rotated * rotated), 1e-15)
    return rotated * scale + target.mean(axis=0)


def plot_reversible_design(bundle: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    names = ("north layer swap", "east layer swap")
    mappings = (
        [3, 4, 5, 0, 1, 2, 6, 7, 8],
        [0, 2, 1, 3, 5, 4, 6, 8, 7],
    )
    for axis, name, mapping in zip(axes[:2], names, mappings):
        axis.scatter(COORDINATES[:, 0], -COORDINATES[:, 1], s=120, color="#f0f3f5", edgecolor="#333")
        for site, destination in enumerate(mapping):
            if site < destination:
                start = np.array([COORDINATES[site, 0], -COORDINATES[site, 1]])
                end = np.array([COORDINATES[destination, 0], -COORDINATES[destination, 1]])
                axis.annotate("", end, start, arrowprops=dict(arrowstyle="<->", color="#1769aa", lw=2))
            axis.text(COORDINATES[site, 0], -COORDINATES[site, 1], PLACE_NAMES[site], ha="center", va="center")
        axis.set_title(name + "\n(local involution)")
        axis.set_aspect("equal"); axis.axis("off")
    calibration = read_rows(bundle / "beacon_calibration.csv")
    informative = [r for r in calibration if r["environment"].endswith("reversible-beacons")]
    exact = np.array([float(r["exact_p_one"]) for r in informative])
    learned = np.array([float(r["learned_p_one"]) for r in informative])
    axes[2].scatter(exact, learned, alpha=.45, s=22, color="#1769aa")
    axes[2].plot([0, 1], [0, 1], "--", color="#555")
    axes[2].set(xlabel="exact response (offline)", ylabel="learned response", xlim=(0, 1), ylim=(0, 1))
    axes[2].set_title("operational beacon calibration")
    fig.suptitle("Reversible motion prevents boundary homing; sensors are learned from experience", fontsize=13)
    fig.savefig(bundle / "reversible_design_and_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_performance(bundle: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        ("all_pairs_success", "all-pairs success", (0, 1.05)),
        ("mean_senses", "sensing interventions", None),
        ("mean_total_interventions", "total interventions", None),
        ("movement_stress_2d", "movement 2D stress", None),
        ("movement_procrustes_r2", "movement Procrustes $R^2$", (0, 1.05)),
        ("mean_commit_entropy", "entropy at commitment", None),
    )
    shown = ("oracle", "active", "fixed-12", "entropy", "active-null")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, (field, title, limits) in zip(axes.flat, fields):
        stats = aggregate(rows, field)
        means = [stats[c][0] for c in shown]
        errors = [stats[c][1] for c in shown]
        axis.bar(range(len(shown)), means, yerr=errors, capsize=3, color=[COLORS[c] for c in shown])
        axis.set_xticks(range(len(shown)), [c.replace("active-", "").replace("fixed-12", "fixed") for c in shown], rotation=18)
        axis.set_title(title)
        if limits: axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Active sensing trades information cost against reliable spatial control", fontsize=14)
    fig.savefig(bundle / "active_atlas_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_geometries(bundle: Path) -> None:
    shown = ("oracle", "active", "fixed-12", "entropy", "active-null")
    fig, axes = plt.subplots(2, len(shown), figsize=(14, 6.1), constrained_layout=True)
    for column, condition in enumerate(shown):
        for row, matrix_name in enumerate(("movement", "total")):
            matrix = aggregate_matrix(bundle, matrix_name, condition)
            distance = (matrix + matrix.T) / 2
            np.fill_diagonal(distance, 0.0)
            coordinates = align(classical_mds(distance), COORDINATES)
            axis = axes[row, column]
            axis.scatter(coordinates[:, 0], -coordinates[:, 1], c=range(9), cmap="viridis", s=62)
            for site, (x, y) in enumerate(coordinates):
                axis.text(x, -y, PLACE_NAMES[site], ha="center", va="center", fontsize=8)
            axis.set_aspect("equal", adjustable="datalim"); axis.set_xticks([]); axis.set_yticks([])
            if row == 0: axis.set_title(condition.replace("active-", ""))
            if column == 0: axis.set_ylabel("movement geometry" if row == 0 else "all-intervention geometry")
    fig.suptitle("Sensing can preserve the spatial base while warping total hodological cost", fontsize=14)
    fig.savefig(bundle / "active_atlas_geometries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sensing_maps(bundle: Path) -> None:
    shown = ("active", "fixed-12", "entropy", "active-null")
    fig, axes = plt.subplots(1, len(shown), figsize=(13, 3.5), constrained_layout=True)
    maximum = max(float(aggregate_matrix(bundle, "sensing", c).max()) for c in shown)
    for axis, condition in zip(axes, shown):
        matrix = aggregate_matrix(bundle, "sensing", condition)
        image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="magma")
        axis.set_title(condition.replace("active-", ""))
        axis.set_xticks(range(9), PLACE_NAMES); axis.set_yticks(range(9), PLACE_NAMES)
        axis.set_xlabel("goal")
    axes[0].set_ylabel("source")
    fig.colorbar(image, ax=axes, shrink=.78, label="mean beacon actions")
    fig.suptitle("Epistemic overhead is pair-dependent, not a constant spatial distance", fontsize=13)
    fig.savefig(bundle / "active_sensing_overhead.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_epistemic_fiber(bundle: Path) -> None:
    records = json.loads((bundle / "trajectories.json").read_text())
    # Use the controller selected by the experiment as the coherent-atlas
    # model, rather than the cheaper goal-relative quotient representation.
    chosen = [
        r for r in records if r["condition"] == "entropy" and r["seed"] == 0
    ][:4]
    movement = aggregate_matrix(bundle, "movement", "entropy")
    base = align(classical_mds((movement + movement.T) / 2), COORDINATES)
    fig = plt.figure(figsize=(13, 3.7), constrained_layout=True)
    for index, record in enumerate(chosen, 1):
        axis = fig.add_subplot(1, 4, index, projection="3d")
        beliefs = np.asarray(record["beliefs"], dtype=float)
        barycenters = beliefs @ base
        heights = np.array([-np.sum(np.clip(b, 1e-15, 1) * np.log(np.clip(b, 1e-15, 1))) / np.log(9) for b in beliefs])
        axis.plot(barycenters[:, 0], barycenters[:, 1], heights, "-o", ms=3, color="#1769aa")
        axis.scatter(base[:, 0], base[:, 1], np.zeros(9), s=13, color="#999", alpha=.55)
        goal = int(record["goal"])
        axis.scatter(base[goal, 0], base[goal, 1], 0, marker="*", s=90, color="#2f9e44")
        axis.set_title(f"E→{PLACE_NAMES[goal]} | {record['senses']} senses")
        axis.set_zlim(0, 1); axis.set_zticks((0, .5, 1)); axis.set_xticks([]); axis.set_yticks([])
        if index == 1: axis.set_zlabel("belief entropy")
        axis.view_init(elev=24, azim=-58)
    fig.suptitle(
        "Atlas-preserving belief paths: an epistemic fiber over the learned spatial base",
        fontsize=13,
    )
    fig.savefig(bundle / "epistemic_fiber_trajectories.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.bundle / "summary.csv")
    plot_reversible_design(args.bundle)
    plot_performance(args.bundle, rows)
    plot_geometries(args.bundle)
    plot_sensing_maps(args.bundle)
    plot_epistemic_fiber(args.bundle)


if __name__ == "__main__":
    main()
