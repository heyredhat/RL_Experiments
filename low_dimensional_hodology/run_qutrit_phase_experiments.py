"""Generate deterministic validation artifacts for the exact qutrit phase lattice."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qutrit_phase_lattice import (
    QutritPhaseLattice,
    bellman_residual,
    distance_matrix,
    schoenberg_gram,
    trace_distance_matrix,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "qutrit-phase"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_source = source - source.mean(axis=0)
    centered_target = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotated = centered_source @ (u @ vt)
    scale = np.sum(rotated * centered_target) / np.sum(rotated * rotated)
    return rotated * scale + target.mean(axis=0)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = QutritPhaseLattice(order=11)
    patch = model.patch_coordinates()
    control = distance_matrix(model, patch)
    quantum = trace_distance_matrix(model, patch)
    gram = schoenberg_gram(control)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    mds = eigenvectors[:, order[:2]] * np.sqrt(np.maximum(eigenvalues[order[:2]], 0.0))
    mds_distance = np.linalg.norm(mds[:, None] - mds[None, :], axis=2)

    all_states = [model.density(coordinate) for coordinate in model.coordinates()]
    minimum_state_separation = min(
        np.linalg.norm(first - second)
        for index, first in enumerate(all_states)
        for second in all_states[index + 1 :]
    )

    rng = np.random.default_rng(20260812)
    waiting_rows: list[dict[str, object]] = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if (dx, dy) == (0, 0):
                continue
            expected = float(np.hypot(dx, dy))
            probability = 1.0 / expected
            empirical = float(rng.geometric(probability, size=50_000).mean())
            waiting_rows.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "success_probability": f"{probability:.12f}",
                    "analytic_cost": f"{expected:.12f}",
                    "empirical_cost": f"{empirical:.12f}",
                    "absolute_error": f"{abs(empirical - expected):.12f}",
                }
            )
    write_csv(RESULTS / "waiting_time_validation.csv", waiting_rows)
    np.savetxt(RESULTS / "exact_patch_distance.csv", control, delimiter=",", fmt="%.12g")
    np.savetxt(RESULTS / "patch_trace_distance.csv", quantum, delimiter=",", fmt="%.12g")
    np.savetxt(RESULTS / "patch_mds_coordinates.csv", mds, delimiter=",", fmt="%.12g")

    target = np.array(patch, dtype=float)
    control_values = upper(control)
    quantum_values = upper(quantum)
    correlation = float(np.corrcoef(control_values, quantum_values)[0, 1])
    metric = model.fubini_study_metric()
    summary = {
        "hilbert_dimension": 3,
        "torus_order": model.order,
        "physical_orbit_states": model.order**2,
        "goal_patch_states": 9,
        "fubini_study_metric": metric.tolist(),
        "minimum_orbit_frobenius_separation": minimum_state_separation,
        "bellman_max_residual": bellman_residual(model, (0, 0)),
        "patch_euclidean_max_error": float(np.max(np.abs(control - np.linalg.norm(target[:, None] - target[None, :], axis=2)))),
        "patch_mds_max_error": float(np.max(np.abs(control - mds_distance))),
        "schoenberg_eigenvalues": np.sort(np.linalg.eigvalsh(gram))[::-1].tolist(),
        "control_vs_trace_distance_pearson": correlation,
        "waiting_time_max_monte_carlo_error": max(float(row["absolute_error"]) for row in waiting_rows),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    phases = np.array([(x * model.epsilon, y * model.epsilon) for x, y in patch])
    axes[0, 0].scatter(phases[:, 0], phases[:, 1], c=range(9), cmap="viridis", s=110)
    for label, (x, y) in zip(patch, phases):
        axes[0, 0].text(x + 0.025, y + 0.025, str(label), fontsize=8)
    axes[0, 0].set(
        title="Nine nonorthogonal qutrit goals in a 2D phase chart",
        xlabel=r"phase coordinate $\alpha$",
        ylabel=r"phase coordinate $\beta$",
        aspect="equal",
    )

    axes[0, 1].scatter(control_values, quantum_values, color="#7b4ab5", alpha=0.8)
    axes[0, 1].set(
        title="Physical distinguishability is not control distance",
        xlabel="exact hodological distance",
        ylabel="qutrit trace distance",
    )

    axes[1, 0].scatter(target[:, 0], target[:, 1], marker="o", s=100, label="ordinary grid")
    aligned = align(mds, target)
    axes[1, 0].scatter(aligned[:, 0], aligned[:, 1], marker="x", s=90, label="cost MDS")
    axes[1, 0].set(title="Exact rank-two cost geometry", aspect="equal")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        [float(row["analytic_cost"]) for row in waiting_rows],
        [float(row["empirical_cost"]) for row in waiting_rows],
        color="#1769aa",
    )
    axes[1, 1].plot([1, 3], [1, 3], "--", color="0.4")
    axes[1, 1].set(
        title="Unit-cost retry instruments realize Euclidean length",
        xlabel="analytic expected interventions",
        ylabel="Monte Carlo mean",
    )
    figure.suptitle("Exact qutrit phase hodology: a local planar chart without nine basis states", fontsize=14)
    figure.savefig(RESULTS / "qutrit_phase_lattice.png", dpi=200)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
