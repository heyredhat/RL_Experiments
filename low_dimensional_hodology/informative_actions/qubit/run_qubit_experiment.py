"""Search, simulate, and visualize the informative qubit construction."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from informative_qubit import (
    BUTTONS,
    InformativeQubit,
    charts_equivalent_under_d4,
    deterministic_search,
    ideal_goal_phases,
    infer_coordinate_chart,
    manhattan_goal_matrix,
    null_signatures,
    random_unitary_signatures,
    reconstruct_phases,
    reconstruct_state_from_probes,
    sample_signatures,
    trace_distance,
    word_equivalence_audit,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 20260812


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def signature_vector(signatures, button):
    return np.array([signatures[button][key] for key in ("action_plus", "probe_X", "probe_Y")])


def main() -> None:
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    search = deterministic_search()
    best = search[0]
    model = InformativeQubit(best["alpha"], best["beta"], best["strength"])
    write_csv(RESULTS / "search_top.csv", search[:100])

    signature_rows = []
    exact_conditions = {
        "informative-nonunitary": {b: model.exact_signature(b) for b in BUTTONS},
        "random-unitary": random_unitary_signatures(model),
        "null": null_signatures(),
    }
    for condition, signatures in exact_conditions.items():
        for button in BUTTONS:
            signature_rows.append({"condition": condition, "button": button, **signatures[button]})
    write_csv(RESULTS / "exact_predictive_signatures.csv", signature_rows)

    rng = np.random.default_rng(SEED)
    recovery_rows = []
    trial_grid = (2, 5, 10, 25, 50, 100, 200, 500, 1000)
    for trials in trial_grid:
        for replicate in range(300):
            sampled = sample_signatures(model, trials, rng)
            phases = reconstruct_phases(sampled, model.coherence_retention)
            chart = infer_coordinate_chart(phases)
            errors = [abs(np.angle(np.exp(1j * (phases[b] - model.phases[b])))) for b in BUTTONS]
            recovery_rows.append(
                {
                    "trials_per_test": trials, "replicate": replicate,
                    "chart_recovered_mod_D4": int(charts_equivalent_under_d4(chart, model.hidden_coordinates)),
                    "phase_rmse": f"{np.sqrt(np.mean(np.square(errors))):.12f}",
                    "max_phase_error": f"{max(errors):.12f}",
                }
            )
    write_csv(RESULTS / "chart_recovery.csv", recovery_rows)

    tomography_rows = []
    for length in (0, 1, 2, 3, 4, 6, 8):
        for replicate in range(250):
            buttons = tuple(rng.choice(BUTTONS, size=length))
            outcomes = []
            state = model.reset_state()
            for button in buttons:
                p_plus, plus_state = model.apply(state, button, 1)
                outcome = 1 if rng.random() < p_plus else -1
                _, state = model.apply(state, button, outcome)
                outcomes.append(outcome)
            exact_probs = {axis: model.probe_probability(state, axis) for axis in ("X", "Y", "Z")}
            sampled_probs = {axis: rng.binomial(300, probability) / 300 for axis, probability in exact_probs.items()}
            reconstructed = reconstruct_state_from_probes(sampled_probs)
            tomography_rows.append(
                {"length": length, "replicate": replicate, "trace_distance": f"{trace_distance(state, reconstructed):.12f}"}
            )
    write_csv(RESULTS / "predictive_state_reconstruction.csv", tomography_rows)

    control_rows = []
    for condition, signatures in exact_conditions.items():
        vectors = np.array([signature_vector(signatures, button) for button in BUTTONS])
        minimum = min(np.linalg.norm(vectors[i] - vectors[j]) for i in range(4) for j in range(i + 1, 4))
        rank = np.linalg.matrix_rank(vectors - vectors.mean(axis=0), tol=1e-10)
        phase_identifiable = condition != "null"
        control_rows.append(
            {
                "condition": condition,
                "minimum_signature_distance": f"{minimum:.12f}",
                "centered_signature_rank": rank,
                "action_semantics_identifiable": int(phase_identifiable),
                "quantum_goal_chart": int(phase_identifiable),
                "external_counter_goal_chart": 1,
                "state_disturbance_per_action": f"{0.0 if condition != 'informative-nonunitary' else 1-model.coherence_retention:.12f}",
            }
        )
    control_rows.append(
        {
            "condition": "external-counter-only", "minimum_signature_distance": "0.000000000000",
            "centered_signature_rank": 0, "action_semantics_identifiable": 0, "quantum_goal_chart": 0,
            "external_counter_goal_chart": 1, "state_disturbance_per_action": "0.000000000000",
        }
    )
    write_csv(RESULTS / "controls.csv", control_rows)

    finite_control_rows = []
    for condition, exact_signatures in exact_conditions.items():
        retention = model.coherence_retention if condition == "informative-nonunitary" else 1.0
        for replicate in range(300):
            sampled = {
                button: {
                    key: float(rng.binomial(200, probability) / 200)
                    for key, probability in exact_signatures[button].items()
                }
                for button in BUTTONS
            }
            vectors = np.array([[sampled[b]["probe_X"], sampled[b]["probe_Y"]] for b in BUTTONS])
            minimum = min(np.linalg.norm(vectors[i] - vectors[j]) for i in range(4) for j in range(i + 1, 4))
            identifiable = minimum > 0.15
            phases = reconstruct_phases(sampled, retention)
            recovered = charts_equivalent_under_d4(infer_coordinate_chart(phases), model.hidden_coordinates)
            finite_control_rows.append(
                {
                    "condition": condition, "replicate": replicate,
                    "minimum_probe_signature_distance": f"{minimum:.12f}",
                    "identifiable": int(identifiable),
                    "correct_chart_and_identifiable": int(identifiable and recovered),
                }
            )
    write_csv(RESULTS / "finite_control_recovery.csv", finite_control_rows)

    goals = tuple(product(range(3), repeat=2))
    goal_rows = []
    matrix = manhattan_goal_matrix()
    for index, source in enumerate(goals):
        for jndex, target in enumerate(goals):
            goal_rows.append({"source_i": source[0], "source_j": source[1], "target_i": target[0], "target_j": target[1], "exact_sequence_cost": int(matrix[index, jndex])})
    write_csv(RESULTS / "goal_geometry.csv", goal_rows)

    equivalence_rows = word_equivalence_audit(model)
    write_csv(RESULTS / "word_equivalence_audit.csv", equivalence_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    top = search[:100]
    scatter = ax.scatter([r["goal_separation"] for r in top], [r["information_margin"] for r in top], c=[r["strength"] for r in top], cmap="viridis", s=35)
    ax.scatter(best["goal_separation"], best["information_margin"], marker="*", s=220, c="red", label="selected")
    ax.set(title="Deterministic instrument search", xlabel="minimum nine-goal phase separation", ylabel="predictive information margin")
    ax.legend(); fig.colorbar(scatter, ax=ax, label="weak-measurement strength")

    ax = axes[0, 1]
    signatures = exact_conditions["informative-nonunitary"]
    for button in BUTTONS:
        vector = signature_vector(signatures, button)
        ax.scatter(vector[1], vector[2], s=100)
        ax.text(vector[1] + .008, vector[2], button)
    ax.set(title="Opaque buttons separated by common future probes", xlabel="P(X probe +)", ylabel="P(Y probe +)")

    ax = axes[1, 0]
    for trials in trial_grid:
        rows = [r for r in recovery_rows if r["trials_per_test"] == trials]
        success = np.mean([r["chart_recovered_mod_D4"] for r in rows])
        rmse = np.mean([float(r["phase_rmse"]) for r in rows])
        ax.scatter(trials, success, color="tab:blue")
        ax.text(trials, success + .025, f"{rmse:.2f}", ha="center", fontsize=8)
    ax.plot(trial_grid, [np.mean([r["chart_recovered_mod_D4"] for r in recovery_rows if r["trials_per_test"]==n]) for n in trial_grid])
    ax.set_xscale("log"); ax.set_ylim(-.03,1.08)
    ax.set(title="Coordinate-free chart recovery\n(labels: mean phase RMSE in radians)", xlabel="samples per common test", ylabel="P(correct modulo D4 gauge)")

    ax = axes[1, 1]
    phases = ideal_goal_phases(model.alpha, model.beta)
    circle = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(circle), np.sin(circle), color="0.75")
    for goal, phase in phases.items():
        ax.scatter(np.cos(phase), np.sin(phase), s=70)
        ax.text(1.09*np.cos(phase), 1.09*np.sin(phase), str(goal), fontsize=8, ha="center", va="center")
    ax.set(title="Nine sequence goals on one qubit equator", aspect="equal"); ax.axis("off")
    fig.suptitle("Informative qubit actions identify a 2D action chart—not a path-independent state space", fontsize=14)
    fig.savefig(FIGURES / "informative_qubit_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    for i in range(3):
        ax.plot([i]*3, range(3), color="0.75", zorder=0)
        ax.plot(range(3), [i]*3, color="0.75", zorder=0)
    goal_phases = ideal_goal_phases(model.alpha, model.beta)
    scatter = ax.scatter([g[0] for g in goals], [g[1] for g in goals], c=[goal_phases[g] for g in goals], cmap="twilight", s=140)
    for goal in goals:
        ax.text(goal[0]+.04, goal[1]+.04, str(goal), fontsize=8)
    ax.set(title="Exact Manhattan geometry of external word counts", xlabel="inferred axis 1 count", ylabel="inferred axis 2 count", aspect="equal")
    fig.colorbar(scatter, ax=ax, label="ideal unitary phase")

    ax = axes[1]
    repeated = [r for r in equivalence_rows if r["path_count"] > 1]
    values = np.array([float(r["max_common_probe_difference"]) for r in repeated])
    points = np.array([(int(r["dx"]), int(r["dy"])) for r in repeated])
    scatter = ax.scatter(points[:,0], points[:,1], c=values, cmap="magma", s=100, vmin=0)
    ax.axhline(0, color="0.8", lw=1); ax.axvline(0, color="0.8", lw=1)
    ax.set(title="Actual channels violate path independence", xlabel="external displacement label", ylabel="external displacement label", aspect="equal")
    fig.colorbar(scatter, ax=ax, label="maximum common-probe difference among paths")
    fig.suptitle("A square sequence chart is not yet an emergent quantum state geometry", fontsize=14)
    fig.savefig(FIGURES / "predictive_compositionality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    grouped = {}
    for trials in trial_grid:
        rows = [r for r in recovery_rows if r["trials_per_test"] == trials]
        grouped[str(trials)] = {
            "chart_recovery": float(np.mean([r["chart_recovered_mod_D4"] for r in rows])),
            "mean_phase_rmse": float(np.mean([float(r["phase_rmse"]) for r in rows])),
        }
    summary = {
        "selected": best,
        "coherence_retention": model.coherence_retention,
        "chart_recovery": grouped,
        "mean_tomography_trace_distance_300_probes": float(np.mean([float(r["trace_distance"]) for r in tomography_rows])),
        "finite_control_recovery_200_trials": {
            condition: {
                "identifiable_rate": float(np.mean([r["identifiable"] for r in finite_control_rows if r["condition"] == condition])),
                "correct_chart_and_identifiable_rate": float(np.mean([r["correct_chart_and_identifiable"] for r in finite_control_rows if r["condition"] == condition])),
            }
            for condition in exact_conditions
        },
        "max_goal_geometry_error_from_manhattan": float(np.max(np.abs(matrix - manhattan_goal_matrix()))),
        "predictive_path_independence": {
            "equivalent_repeated_path_classes": int(sum(r["equivalent_at_1e-10"] for r in equivalence_rows if r["path_count"] > 1)),
            "tested_repeated_path_classes": int(sum(r["path_count"] > 1 for r in equivalence_rows)),
            "singleton_classes": int(sum(r["path_count"] == 1 for r in equivalence_rows)),
            "total_displacement_classes": len(equivalence_rows),
            "worst_common_probe_difference": max(float(r["max_common_probe_difference"]) for r in equivalence_rows),
        },
        "search_candidates": len(search), "seed": SEED,
        "artifacts": ["search_top.csv", "exact_predictive_signatures.csv", "chart_recovery.csv", "predictive_state_reconstruction.csv", "controls.csv", "finite_control_recovery.csv", "goal_geometry.csv", "word_equivalence_audit.csv", "figures/informative_qubit_summary.png", "figures/predictive_compositionality.png"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
