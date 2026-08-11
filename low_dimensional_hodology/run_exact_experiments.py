"""Generate all compact artifacts for the exactly solvable qubit model."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exact_qubit_lattice import (
    PhaseLattice,
    approximate_shortest_distance,
    canonical_word,
    classical_mds,
    distance_matrix,
    exact_euclidean_macro_cost,
    exact_word_distance,
    euclidean_success_probability,
    goals_3x3,
    minimum_goal_separation,
    noisy_goal_fidelity,
    qutrit_torus_distance,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    model = PhaseLattice()
    goals = goals_3x3()

    pair_rows = []
    for source in goals:
        for target in goals:
            word = canonical_word(source, target)
            reached = model.apply_word(model.density(source), word)
            target_density = model.density(target)
            fidelity = float(np.real(np.trace(reached @ target_density)))
            pair_rows.append(
                {
                    "source_i": source[0], "source_j": source[1],
                    "target_i": target[0], "target_j": target[1],
                    "word": "".join(word) or "identity",
                    "manhattan_cost": exact_word_distance(source, target),
                    "euclidean_expected_unit_cost": f"{exact_euclidean_macro_cost(source, target):.12f}",
                    "qutrit_torus_cost": qutrit_torus_distance(source, target),
                    "terminal_fidelity": f"{fidelity:.16f}",
                }
            )
    write_csv(RESULTS / "all_pairs.csv", list(pair_rows[0]), pair_rows)

    goal_rows = []
    for goal in goals:
        nearest_false = max(model.fidelity(goal, other) for other in goals if other != goal)
        goal_rows.append(
            {
                "i": goal[0], "j": goal[1],
                "phase_mod_2pi": f"{model.phase(goal) % (2*np.pi):.12f}",
                "max_single_shot_false_accept": f"{nearest_false:.12f}",
            }
        )
    write_csv(RESULTS / "goals.csv", list(goal_rows[0]), goal_rows)

    robustness_rows = []
    for tolerance in (1e-12, 1e-8, 1e-5, 1e-3, 1e-2, 5e-2):
        exact = []
        approximate = []
        for source in goals:
            for target in goals:
                if source == target:
                    continue
                exact.append(exact_word_distance(source, target))
                approximate.append(approximate_shortest_distance(model, source, target, tolerance))
        exact_array = np.asarray(exact)
        approximate_array = np.asarray(approximate)
        robustness_rows.append(
            {
                "infidelity_tolerance": tolerance,
                "shortened_pair_fraction": f"{np.mean(approximate_array < exact_array):.12f}",
                "mean_exact_cost": f"{np.mean(exact_array):.12f}",
                "mean_tolerant_cost": f"{np.mean(approximate_array):.12f}",
            }
        )
    write_csv(RESULTS / "finite_tolerance.csv", list(robustness_rows[0]), robustness_rows)

    rng = np.random.default_rng(20260811)
    noise_rows = []
    for eta in (1.0, 0.99, 0.95, 0.9, 0.8):
        for length in range(5):
            analytic = noisy_goal_fidelity(length, eta)
            trials = 20000
            empirical = float(np.mean(rng.random(trials) < analytic))
            noise_rows.append(
                {
                    "eta": eta, "path_length": length,
                    "analytic_acceptance": f"{analytic:.12f}",
                    "monte_carlo_acceptance": f"{empirical:.12f}",
                    "absolute_error": f"{abs(empirical-analytic):.12f}",
                }
            )
    write_csv(RESULTS / "dephasing_validation.csv", list(noise_rows[0]), noise_rows)

    waiting_rows = []
    for delta in product(range(-2, 3), repeat=2):
        if delta == (0, 0):
            continue
        probability = euclidean_success_probability(delta)
        trials = 50000
        samples = rng.geometric(probability, size=trials)
        expected = float(np.hypot(*delta))
        empirical = float(np.mean(samples))
        waiting_rows.append(
            {
                "delta_i": delta[0], "delta_j": delta[1],
                "success_probability": f"{probability:.12f}",
                "analytic_expected_interventions": f"{expected:.12f}",
                "monte_carlo_mean_interventions": f"{empirical:.12f}",
                "absolute_error": f"{abs(empirical-expected):.12f}",
                "trials": trials,
            }
        )
    write_csv(RESULTS / "euclidean_waiting_validation.csv", list(waiting_rows[0]), waiting_rows)

    manhattan = distance_matrix(exact_word_distance)
    torus = distance_matrix(qutrit_torus_distance)
    euclidean = distance_matrix(exact_euclidean_macro_cost)
    mds_coordinates, mds_stress = classical_mds(manhattan)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ax = axes[0, 0]
    for i in range(3):
        ax.plot([i] * 3, range(3), color="0.75", zorder=0)
        ax.plot(range(3), [i] * 3, color="0.75", zorder=0)
    phases = np.array([model.phase(goal) % (2*np.pi) for goal in goals])
    scatter = ax.scatter([g[0] for g in goals], [g[1] for g in goals], c=phases, cmap="twilight", s=150)
    for goal in goals:
        ax.text(goal[0], goal[1], f" {goal}", va="bottom", fontsize=8)
    ax.set(title="Exact hodological graph: a 3x3 patch of Z²", xlabel="U exponent", ylabel="V exponent", aspect="equal")
    fig.colorbar(scatter, ax=ax, label="qubit phase (radians)")

    ax = axes[0, 1]
    circle = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(circle), np.sin(circle), color="0.7")
    for goal, phase in zip(goals, phases):
        ax.scatter(np.cos(phase), np.sin(phase), s=65)
        ax.text(1.08*np.cos(phase), 1.08*np.sin(phase), str(goal), fontsize=7, ha="center", va="center")
    ax.set(title="The same nine goals are nonorthogonal qubit rays", aspect="equal")
    ax.axis("off")

    ax = axes[1, 0]
    difference = manhattan - torus
    image = ax.imshow(difference, cmap="magma", vmin=0)
    ax.set(title="Finite-order qutrit Weyl shortcut: open cost − torus cost", xlabel="target goal index", ylabel="source goal index")
    fig.colorbar(image, ax=ax, label="lost units of distance")

    ax = axes[1, 1]
    for eta in (0.99, 0.95, 0.9, 0.8):
        ax.plot(range(5), [noisy_goal_fidelity(length, eta) for length in range(5)], marker="o", label=f"η={eta}")
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set(title="Exactly soluble deformation by dephasing", xlabel="shortest-path length", ylabel="goal-verifier acceptance", ylim=(0.45, 1.02))
    ax.legend()
    fig.suptitle("Low-dimensional exact hodology: algebra, operational states, and failure modes", fontsize=14)
    fig.savefig(FIGURES / "exact_qubit_lattice.png", dpi=180)
    plt.close(fig)

    manifest = {
        "model": "faithful irrational qubit phase representation of Z^2",
        "epsilon": model.epsilon,
        "alpha": model.alpha,
        "beta": model.beta,
        "goals": len(goals),
        "minimum_angular_separation": minimum_goal_separation(model),
        "max_pair_terminal_infidelity": max(1.0 - float(row["terminal_fidelity"]) for row in pair_rows),
        "manhattan_mds_2d_stress": mds_stress,
        "euclidean_coordinate_distance_max_error": float(np.max(np.abs(euclidean - np.array([[np.hypot(a[0]-b[0], a[1]-b[1]) for b in goals] for a in goals])))),
        "qutrit_pairs_with_strict_shortcuts": int(np.sum(np.triu(torus < manhattan, 1))),
        "euclidean_waiting_max_monte_carlo_error": max(float(row["absolute_error"]) for row in waiting_rows),
        "artifacts": ["all_pairs.csv", "goals.csv", "finite_tolerance.csv", "dephasing_validation.csv", "euclidean_waiting_validation.csv", "figures/exact_qubit_lattice.png"],
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
