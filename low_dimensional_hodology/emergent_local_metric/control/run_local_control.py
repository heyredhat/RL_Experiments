"""Optimize local qutrit repertoires and generate deterministic artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from local_control import (
    LocalQutritModel,
    anisotropy_by_radius,
    bellman_distances,
    bellman_residual,
    classes_for,
    costs_from_parameters,
    euclidean,
    grid_displacements,
    kraus_completeness_residual,
    optimize_class_costs,
    pairwise_metric,
    repertoire,
    schoenberg_diagnostics,
    translation_covariance_residual,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TRAIN_RADIUS = 4
EVAL_RADIUS = 12
NAMES = ("D4", "D8", "D16", "D32")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    model = LocalQutritModel()
    training = grid_displacements(TRAIN_RADIUS)
    evaluation = grid_displacements(EVAL_RADIUS)
    target_functions = {
        "euclidean": euclidean,
        "fubini-study": model.scaled_fubini_study_displacement,
    }
    fit_rows: list[dict] = []
    displacement_rows: list[dict] = []
    anisotropy_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    solutions = {}

    for target_name, target_function in target_functions.items():
        for name in NAMES:
            actions = repertoire(name)
            parameters, history = optimize_class_costs(actions, training, target_function)
            costs = costs_from_parameters(actions, parameters)
            distances = bellman_distances(actions, costs, EVAL_RADIUS * 2)
            solutions[(target_name, name)] = (actions, parameters, costs, distances)
            for key in classes_for(actions):
                fit_rows.append(
                    {
                        "target": target_name, "repertoire": name,
                        "class_dx": key[0], "class_dy": key[1],
                        "step_norm": f"{euclidean(key):.12f}",
                        "optimized_expected_cost": f"{parameters[key]:.12f}",
                        "success_probability": f"{1.0/parameters[key]:.12f}",
                        "initial_objective": f"{history[0]:.12e}",
                        "final_objective": f"{history[-1]:.12e}",
                    }
                )
            for point in evaluation:
                truth = target_function(point)
                prediction = distances[point]
                displacement_rows.append(
                    {
                        "target": target_name, "repertoire": name,
                        "dx": point[0], "dy": point[1],
                        "radius": f"{euclidean(point):.12f}",
                        "training": int(euclidean(point) <= TRAIN_RADIUS + 1e-12),
                        "truth": f"{truth:.12f}", "prediction": f"{prediction:.12f}",
                        "relative_error": f"{prediction/truth-1.0:.12f}",
                        "absolute_error": f"{prediction-truth:.12f}",
                    }
                )
            for row in anisotropy_by_radius(distances, range(2, EVAL_RADIUS + 1, 2)):
                anisotropy_rows.append({"target": target_name, "repertoire": name, **row})

            diagnostic_points = tuple((x, y) for x in range(-2, 3) for y in range(-2, 3))
            metric = pairwise_metric(diagnostic_points, distances)
            schoenberg = schoenberg_diagnostics(metric)
            heldout = [p for p in evaluation if euclidean(p) > TRAIN_RADIUS]
            relative = np.array([distances[p] / target_function(p) - 1.0 for p in heldout])
            diagnostic_rows.append(
                {
                    "target": target_name, "repertoire": name,
                    "direction_count": len(actions), "parameter_count": len(parameters),
                    "max_local_step": f"{max(euclidean(a) for a in actions):.12f}",
                    "locality_ratio_to_eval_radius": f"{max(euclidean(a) for a in actions)/EVAL_RADIUS:.12f}",
                    "heldout_relative_rmse": f"{np.sqrt(np.mean(relative**2)):.12f}",
                    "heldout_mean_relative_error": f"{np.mean(relative):.12f}",
                    "heldout_max_abs_relative_error": f"{np.max(np.abs(relative)):.12f}",
                    "bellman_residual": f"{bellman_residual(distances, actions, costs, EVAL_RADIUS):.3e}",
                    "translation_covariance_residual": f"{translation_covariance_residual(model, ((0,0),(2,-1)), actions):.3e}",
                    "kraus_completeness_residual": f"{kraus_completeness_residual(model, actions, costs):.3e}",
                    **schoenberg,
                }
            )

    write_csv("optimized_costs.csv", fit_rows)
    write_csv("heldout_displacements.csv", displacement_rows)
    write_csv("anisotropy_by_radius.csv", anisotropy_rows)
    write_csv("diagnostics.csv", diagnostic_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = dict(zip(NAMES, plt.cm.viridis(np.linspace(0.08, 0.92, len(NAMES)))))
    for target_index, target_name in enumerate(target_functions):
        ax = axes[0, target_index]
        for name in NAMES:
            rows = [r for r in displacement_rows if r["target"] == target_name and r["repertoire"] == name]
            radii = sorted({int(np.ceil(float(r["radius"]))) for r in rows})
            means, lows, highs = [], [], []
            for radius in radii:
                values = [float(r["prediction"])/float(r["truth"]) for r in rows if radius-1 < float(r["radius"]) <= radius]
                means.append(np.mean(values)); lows.append(np.min(values)); highs.append(np.max(values))
            ax.plot(radii, means, marker="o", ms=3, label=name, color=colors[name])
            ax.fill_between(radii, lows, highs, alpha=0.11, color=colors[name])
        ax.axvspan(0, TRAIN_RADIUS, color="0.8", alpha=0.2, label="training radius" if target_index == 0 else None)
        ax.axhline(1.0, color="black", lw=1, ls="--")
        ax.set(title=f"Generalization to {target_name} target", xlabel="radius", ylabel="predicted / target")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in NAMES:
        rows = [r for r in anisotropy_rows if r["target"] == "euclidean" and r["repertoire"] == name]
        ax.plot([r["radius"] for r in rows], [r["anisotropy_range"] for r in rows], marker="o", label=name, color=colors[name])
    ax.set(title="Directional anisotropy of learned Euclidean metric", xlabel="radius", ylabel="range / mean")
    ax.legend()

    ax = axes[1, 1]
    width = 0.18
    x = np.arange(len(NAMES))
    for index, target_name in enumerate(target_functions):
        rows = [r for r in diagnostic_rows if r["target"] == target_name]
        ax.bar(x + (index-0.5)*width, [float(r["negative_eigenmass_fraction"]) for r in rows], width, label=f"{target_name}: Schoenberg")
        ax.bar(x + (index+1.5)*width, [float(r["mds_2d_stress"]) for r in rows], width, alpha=0.45, label=f"{target_name}: 2D stress")
    ax.set_xticks(x + width/2, NAMES)
    ax.set_yscale("log")
    ax.set(title="Euclidean embeddability diagnostics", ylabel="diagnostic (log scale)")
    ax.legend(fontsize=7)
    fig.suptitle("Can one local qutrit repertoire extrapolate an ordinary metric?", fontsize=14)
    fig.savefig(FIGURES / "local_control_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    euclidean_diagnostics = [r for r in diagnostic_rows if r["target"] == "euclidean"]
    fs_diagnostics = [r for r in diagnostic_rows if r["target"] == "fubini-study"]
    summary = {
        "model": "translation-covariant equilateral-phase qutrit with binary random-unitary local instruments",
        "training_radius": TRAIN_RADIUS,
        "evaluation_radius": EVAL_RADIUS,
        "phase_scale": model.phase_scale,
        "training_displacement_count": len(training),
        "heldout_displacement_count": sum(euclidean(point) > TRAIN_RADIUS for point in evaluation),
        "repertoires": {name: len(repertoire(name)) for name in NAMES},
        "best_euclidean_heldout_rmse": min(
            ({"repertoire": r["repertoire"], "value": float(r["heldout_relative_rmse"])} for r in euclidean_diagnostics),
            key=lambda item: item["value"],
        ),
        "best_fubini_study_heldout_rmse": min(
            ({"repertoire": r["repertoire"], "value": float(r["heldout_relative_rmse"])} for r in fs_diagnostics),
            key=lambda item: item["value"],
        ),
        "maximum_bellman_residual": max(float(r["bellman_residual"]) for r in diagnostic_rows),
        "maximum_covariance_residual": max(float(r["translation_covariance_residual"]) for r in diagnostic_rows),
        "maximum_kraus_residual": max(float(r["kraus_completeness_residual"]) for r in diagnostic_rows),
        "artifacts": ["optimized_costs.csv", "heldout_displacements.csv", "anisotropy_by_radius.csv", "diagnostics.csv", "figures/local_control_summary.png"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
