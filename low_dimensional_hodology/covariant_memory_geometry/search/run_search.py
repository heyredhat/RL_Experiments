"""Run the deterministic covariant-memory qutrit search and diagnostics."""

from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from covariant_memory import (
    COORDS, MemoryInstrument, analytic_distance, analytic_shell_values,
    branch_rank_diagnostics, covariance_residual, exhaustive_bellman_verification,
    geodesic_path_closure, kraus_residual, metric_diagnostics,
    observable_action_probe_audit, oracle_labelled_action_audit,
    oracle_labelled_transition_nll, schoenberg, search_grid,
    state_hitting_bellman, torus_diagnostics, torus_distance,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 20260812


def write_csv(name: str, rows: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def open_flat_diagnostics(distance: np.ndarray) -> dict[str, float]:
    target = np.array([[np.hypot(a[0]-b[0], a[1]-b[1]) for b in COORDS] for a in COORDS])
    mask = np.triu(np.ones_like(target, dtype=bool), 1)
    scale = float(np.dot(target[mask], distance[mask]) / np.dot(target[mask], target[mask]))
    error = distance[mask] - scale * target[mask]
    return {
        "open_grid_correlation": float(np.corrcoef(target[mask], distance[mask])[0,1]),
        "open_grid_relative_rmse": float(np.sqrt(np.mean(error**2)) / np.mean(distance[mask])),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    grid = search_grid()
    write_csv("search_grid.csv", grid)
    exhaustive = exhaustive_bellman_verification()
    write_csv("bellman_grid_verification.csv", exhaustive)

    minimum_mi = 0.05
    feasible = [r for r in grid if float(r["memory"]) > 0 and float(r["immediate_mi_bits"]) >= minimum_mi and bool(r["is_metric"])]
    selected_row = min(feasible, key=lambda r: (float(r["relative_rmse_after_scale"]), -float(r["memory"])))
    square_memory = (4.0 * sqrt(2.0) - 5.0) / 3.0
    candidates = {
        "state-hitting-hesse": MemoryInstrument(0.0, 1.0),
        "exact-local-square": MemoryInstrument(square_memory, 1.0),
        "selected-torus": MemoryInstrument(float(selected_row["memory"]), float(selected_row["sharpness"])),
        "null-report": MemoryInstrument(0.8, 0.0),
        "memory-only": MemoryInstrument(1.0, 1.0),
        "weak-report": MemoryInstrument(0.8, 0.5),
    }
    observable_summary, observable_rows = observable_action_probe_audit(
        candidates["selected-torus"], 500, SEED
    )
    write_csv("observable_action_probe.csv", observable_rows)
    rows = []
    matrices = {}
    closure_rows = []
    for name, model in candidates.items():
        analytic = analytic_distance(model)
        bellman, iterations = state_hitting_bellman(model)
        matrices[name] = bellman
        edge, diagonal = analytic_shell_values(model)
        diagnostics = {
            **metric_diagnostics(bellman), **torus_diagnostics(bellman),
            **schoenberg(bellman), **open_flat_diagnostics(bellman),
        }
        oracle_action_accuracy = oracle_action_mae = float("nan")
        heldout = {
            "oracle_joint_exact_nll_bits": float("nan"),
            "oracle_joint_learned_nll_bits": float("nan"),
            "oracle_joint_state_marginal_nll_bits": float("nan"),
        }
        if model.memory > 0:
            oracle_action_accuracy, oracle_action_mae, _ = oracle_labelled_action_audit(model, 500, SEED)
            raw_heldout = oracle_labelled_transition_nll(model, 400, 2000, 8, SEED)
            heldout = {
                "oracle_joint_exact_nll_bits": raw_heldout["exact_nll_bits"],
                "oracle_joint_learned_nll_bits": raw_heldout["learned_nll_bits"],
                "oracle_joint_state_marginal_nll_bits": raw_heldout["marginal_nll_bits"],
                "oracle_joint_heldout_strings": raw_heldout["heldout_strings"],
                "oracle_joint_heldout_length": raw_heldout["heldout_length"],
            }
        closure = geodesic_path_closure(model)
        closure_rows.append({"candidate": name, **closure})
        rows.append(
            {
                "candidate": name, "memory": model.memory, "sharpness": model.sharpness,
                "immediate_mi_bits": model.immediate_information_bits(),
                "edge_value": edge, "diagonal_value": diagonal,
                "diagonal_edge_ratio": diagonal/edge,
                "bellman_analytic_max_error": float(np.max(np.abs(bellman-analytic))),
                "bellman_iterations": iterations,
                "kraus_completeness_residual": kraus_residual(model),
                "hesse_branch_kernel_covariance_residual": covariance_residual(model),
                "oracle_labelled_action_accuracy": oracle_action_accuracy,
                "oracle_labelled_kernel_mae": oracle_action_mae,
                **branch_rank_diagnostics(model), **heldout, **closure, **diagnostics,
            }
        )
        np.savetxt(RESULTS / f"bellman_{name}.csv", bellman, delimiter=",", fmt="%.12f")
    write_csv("candidate_diagnostics.csv", rows)
    write_csv("path_closure.csv", closure_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0,0]
    sharp = [r for r in grid if abs(float(r["sharpness"])-1.0) < 1e-12]
    ax.plot([r["memory"] for r in sharp], [r["edge_mean"] for r in sharp], marker="o", label="edge")
    ax.plot([r["memory"] for r in sharp], [r["diagonal_mean"] for r in sharp], marker="o", label="diagonal")
    ax.axvline(square_memory, color="tab:green", ls="--", label="exact local square")
    ax.axvline(float(selected_row["memory"]), color="tab:red", ls=":", label="selected torus")
    ax.set(title="Exact state-hitting Bellman shells", xlabel="memory probability μ", ylabel="expected interventions"); ax.legend()

    ax = axes[0,1]
    scatter = ax.scatter([r["immediate_mi_bits"] for r in grid], [r["relative_rmse_after_scale"] for r in grid], c=[r["memory"] for r in grid], cmap="viridis", s=22)
    ax.scatter([selected_row["immediate_mi_bits"]], [selected_row["relative_rmse_after_scale"]], marker="*", s=220, c="red")
    ax.set(title="Information–torus distortion frontier", xlabel="immediate MI (bits)", ylabel="scaled torus relative RMSE")
    fig.colorbar(scatter, ax=ax, label="memory probability")

    ax = axes[1,0]
    labels = [r["candidate"] for r in rows]
    x = np.arange(len(labels)); width=.36
    ax.bar(x-width/2, [r["mds_2d_stress"] for r in rows], width, label="2D MDS stress")
    ax.bar(x+width/2, [r["negative_eigenmass_fraction"] for r in rows], width, label="negative eigenmass")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_yscale("log")
    ax.set(title="Open-flat Euclidean diagnostics", ylabel="diagnostic (log scale)"); ax.legend()

    ax = axes[1,1]
    image = ax.imshow(matrices["selected-torus"], cmap="magma")
    ax.set(title="Selected integrated Bellman metric", xlabel="goal token", ylabel="predictive state token")
    fig.colorbar(image, ax=ax, label="expected interventions")
    fig.suptitle("Qutrit instruments with a full-operator-rank observed memory branch", fontsize=14)
    fig.savefig(FIGURES / "covariant_memory_search.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    by_name = {row["candidate"]: row for row in rows}
    summary = {
        "seed": SEED, "search_candidates": len(grid), "minimum_selected_mi_bits": minimum_mi,
        "exhaustive_grid_bellman_verification": {
            "points": len(exhaustive),
            "maximum_analytic_bellman_error": max(float(row["analytic_bellman_max_error"]) for row in exhaustive),
            "all_metric": all(bool(row["is_metric"]) for row in exhaustive),
        },
        "corrected_hesse_baseline": {k: by_name["state-hitting-hesse"][k] for k in ("edge_value","diagonal_value","is_metric","immediate_mi_bits")},
        "exact_local_square": {k: by_name["exact-local-square"][k] for k in ("memory","immediate_mi_bits","edge_value","diagonal_value","diagonal_edge_ratio","is_metric")},
        "selected_torus": {k: by_name["selected-torus"][k] for k in ("memory","sharpness","immediate_mi_bits","edge_value","diagonal_value","relative_rmse_after_scale","strict_shell_margin","is_metric","equal_length_path_closure_residual","all_length_path_closure_residual")},
        "observable_action_probe": observable_summary,
        "oracle_labelled_transition_benchmark": {k: by_name["selected-torus"][k] for k in ("oracle_labelled_action_accuracy","oracle_joint_exact_nll_bits","oracle_joint_learned_nll_bits","oracle_joint_state_marginal_nll_bits")},
        "maximum_reported_candidate_kraus_completeness_residual": max(float(row["kraus_completeness_residual"]) for row in rows),
        "maximum_reported_candidate_hesse_kernel_covariance_residual": max(float(row["hesse_branch_kernel_covariance_residual"]) for row in rows),
        "artifacts": [
            "search_grid.csv", "bellman_grid_verification.csv",
            "candidate_diagnostics.csv", "observable_action_probe.csv",
            "path_closure.csv", "bellman_state-hitting-hesse.csv",
            "bellman_exact-local-square.csv", "bellman_selected-torus.csv",
            "bellman_null-report.csv", "bellman_memory-only.csv",
            "bellman_weak-report.csv", "figures/covariant_memory_search.png",
        ],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
