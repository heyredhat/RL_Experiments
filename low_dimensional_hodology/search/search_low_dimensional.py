#!/usr/bin/env python3
"""Reproducible search for low-dimensional quantum goal geometries.

This script deliberately separates three notions which are easily conflated:

1. geometry in a classical goal-progress automaton;
2. geometry in the distinguishability of conditional quantum states; and
3. geometry in the controlled transition/hitting-cost structure.

It studies an automaton-only null control, local qubit rotation patches, and an
exact qutrit Weyl--Heisenberg/Hesse-SIC construction.  Outputs are written next
to this file under ``results/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def ket_density(ket: np.ndarray) -> np.ndarray:
    ket = np.asarray(ket, dtype=complex)
    ket = ket / np.linalg.norm(ket)
    return np.outer(ket, ket.conj())


def pure_trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Trace distance between normalized pure-state kets."""
    overlap = abs(np.vdot(first, second)) ** 2
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, overlap))))


def pairwise_trace_distance(kets: np.ndarray) -> np.ndarray:
    count = len(kets)
    distances = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distances[i, j] = distances[j, i] = pure_trace_distance(kets[i], kets[j])
    return distances


def square_coordinates() -> np.ndarray:
    return np.array([(x, y) for x in range(3) for y in range(3)], dtype=float)


def torus_manhattan() -> np.ndarray:
    coords = square_coordinates().astype(int)
    result = np.zeros((9, 9), dtype=float)
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            dx = min((x1 - x2) % 3, (x2 - x1) % 3)
            dy = min((y1 - y2) % 3, (y2 - y1) % 3)
            result[i, j] = dx + dy
    return result


def open_manhattan() -> np.ndarray:
    coords = square_coordinates()
    return np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=-1)


def euclidean_grid() -> np.ndarray:
    coords = square_coordinates()
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float)[np.triu_indices(len(matrix), 1)]


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.std(first) < 1e-14 or np.std(second) < 1e-14:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and np.isclose(values[order[stop]], values[order[start]]):
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return pearson(average_ranks(first), average_ranks(second))


def classical_mds(distances: np.ndarray, dimensions: int = 2) -> tuple[np.ndarray, float, np.ndarray]:
    """Classical MDS coordinates, raw normalized stress, and Gram eigenvalues."""
    distances = np.asarray(distances, dtype=float)
    count = len(distances)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distances * distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coords = eigenvectors[:, :dimensions] * np.sqrt(positive)
    fitted = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    numerator = np.sum((upper_values(distances) - upper_values(fitted)) ** 2)
    denominator = np.sum(upper_values(distances) ** 2)
    stress = float(np.sqrt(numerator / denominator)) if denominator else 0.0
    return coords, stress, eigenvalues


def rotation(axis: str, theta: float) -> np.ndarray:
    pauli = {
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    }[axis]
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * pauli


def qubit_patch(theta: float) -> tuple[np.ndarray, float]:
    """Nine states ``Rx(theta)^x Ry(theta)^y |0>`` and commutator defect."""
    rx = rotation("x", theta)
    ry = rotation("y", theta)
    zero = np.array([1.0, 0.0], dtype=complex)
    states = []
    for x in range(3):
        for y in range(3):
            states.append(np.linalg.matrix_power(rx, x) @ np.linalg.matrix_power(ry, y) @ zero)
    left = rx @ ry @ zero
    right = ry @ rx @ zero
    return np.asarray(states), pure_trace_distance(left, right)


def qubit_translation_defects(theta: float) -> tuple[float, float]:
    """Mean/max failure of the two controls to translate patch coordinates.

    With our word convention the x translation is exact in the patch interior,
    while the y translation fails by the noncommutativity of ``Rx`` and ``Ry``.
    """
    states, _ = qubit_patch(theta)
    rx = rotation("x", theta)
    ry = rotation("y", theta)
    defects = []
    for x in range(3):
        for y in range(3):
            source = states[3 * x + y]
            if x < 2:
                defects.append(pure_trace_distance(rx @ source, states[3 * (x + 1) + y]))
            if y < 2:
                defects.append(pure_trace_distance(ry @ source, states[3 * x + y + 1]))
    return float(np.mean(defects)), float(np.max(defects))


def qubit_search(theta_count: int = 180) -> list[dict[str, float]]:
    target_euclidean = upper_values(euclidean_grid())
    target_manhattan = upper_values(open_manhattan())
    rows: list[dict[str, float]] = []
    for theta in np.linspace(0.01, 1.50, theta_count):
        states, commutator = qubit_patch(float(theta))
        mean_defect, max_defect = qubit_translation_defects(float(theta))
        distances = pairwise_trace_distance(states)
        _, stress, eigenvalues = classical_mds(distances, 2)
        values = upper_values(distances)
        min_separation = float(np.min(values))
        positive = np.maximum(eigenvalues, 0.0)
        planarity = float(np.sum(positive[:2]) / np.sum(positive)) if np.sum(positive) else 0.0
        rows.append(
            {
                "theta": float(theta),
                "min_trace_separation": min_separation,
                "commutator_defect": commutator,
                "mean_translation_defect": mean_defect,
                "max_translation_defect": max_defect,
                "pearson_euclidean": pearson(values, target_euclidean),
                "spearman_manhattan": spearman(values, target_manhattan),
                "mds_2d_stress": stress,
                "positive_gram_planarity": planarity,
                # A transparent search score: geometry is useless if all states coalesce.
                "separation_adjusted_score": (
                    pearson(values, target_euclidean) * min_separation / (stress + 0.02)
                ),
            }
        )
    return rows


def hesse_sic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qutrit Hesse SIC orbit and Weyl controls X,Z.

    Labels are ``(m,n)`` in row-major order, with
    ``|psi_mn> = X^m Z^n |fiducial>``.
    """
    omega = np.exp(2j * np.pi / 3)
    x = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    z = np.diag([1.0, omega, omega**2]).astype(complex)
    fiducial = np.array([0.0, 1.0, -1.0], dtype=complex) / np.sqrt(2)
    states = []
    for m in range(3):
        for n in range(3):
            states.append(np.linalg.matrix_power(x, m) @ np.linalg.matrix_power(z, n) @ fiducial)
    return np.asarray(states), x, z


def qutrit_phase_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nine equal-amplitude qutrit states carrying two ternary phases."""
    omega = np.exp(2j * np.pi / 3)
    states = np.array(
        [[1.0, omega**m, omega**n] for m in range(3) for n in range(3)],
        dtype=complex,
    ) / np.sqrt(3.0)
    u = np.diag([1.0, omega, 1.0]).astype(complex)
    v = np.diag([1.0, 1.0, omega]).astype(complex)
    return states, u, v


def sic_probabilities(states: np.ndarray) -> np.ndarray:
    """Hesse-SIC outcome probabilities ``tr(rho Pi_o/3)``."""
    overlaps = np.abs(states.conj() @ states.T) ** 2
    return overlaps / 3.0


def hesse_transitions(states: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Infer exact permutation action of controls on the SIC orbit."""
    transitions = np.zeros((len(controls), 9), dtype=int)
    for action, unitary in enumerate(controls):
        for source, state in enumerate(states):
            moved = unitary @ state
            fidelities = np.abs(states.conj() @ moved) ** 2
            target = int(np.argmax(fidelities))
            if fidelities[target] < 1.0 - 1e-10:
                raise RuntimeError("Weyl control did not close on Hesse SIC orbit")
            transitions[action, source] = target
    return transitions


def solve_sic_hitting_cost(
    probabilities: np.ndarray,
    transitions: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve nine stochastic shortest-path Bellman equations.

    Four unit-cost Weyl controls move deterministically.  The fifth unit-cost
    action performs the SIC measurement.  Outcome ``goal`` terminates; every
    other outcome resets the state to that outcome's SIC ket.
    """
    costs = np.zeros((9, 9), dtype=float)
    policies = np.zeros((9, 9), dtype=int)
    max_iterations = 200_000
    final_iterations = 0
    for goal in range(9):
        value = np.full(9, 12.0)
        for iteration in range(max_iterations):
            control_q = 1.0 + value[transitions]
            measurement_q = np.ones(9)
            mask = np.arange(9) != goal
            measurement_q += probabilities[:, mask] @ value[mask]
            all_q = np.vstack([control_q, measurement_q[None, :]])
            updated = np.min(all_q, axis=0)
            if np.max(np.abs(updated - value)) < tolerance:
                break
            value = updated
        else:
            raise RuntimeError("Bellman iteration failed to converge")
        final_iterations = max(final_iterations, iteration + 1)
        costs[:, goal] = updated
        policies[:, goal] = np.argmin(all_q, axis=0)
    return costs, policies, final_iterations


def solve_confirmation_goal(
    probabilities: np.ndarray,
    transitions: np.ndarray,
    confirmations: int,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Solve a sequence goal requiring repeated target outcomes.

    A goal succeeds only after ``confirmations`` consecutive uses of the common
    nine-outcome measurement return its designated outcome.  A different
    outcome or any intervening control resets sequence progress.  The returned
    matrix gives costs from physical state ``s`` at zero progress to goal ``g``.
    """
    if confirmations < 1:
        raise ValueError("confirmations must be positive")
    costs = np.zeros((9, 9), dtype=float)
    max_used = 0
    for goal in range(9):
        value = np.full((confirmations, 9), 6.0 * confirmations + 6.0)
        for iteration in range(300_000):
            updated = np.empty_like(value)
            for progress in range(confirmations):
                control_q = 1.0 + value[0, transitions]
                measurement_q = np.ones(9)
                for state in range(9):
                    for outcome in range(9):
                        probability = probabilities[state, outcome]
                        if outcome == goal and progress == confirmations - 1:
                            continue
                        next_progress = progress + 1 if outcome == goal else 0
                        measurement_q[state] += probability * value[next_progress, outcome]
                updated[progress] = np.minimum(np.min(control_q, axis=0), measurement_q)
            if np.max(np.abs(updated - value)) < tolerance:
                break
            value = updated
        else:
            raise RuntimeError("confirmation Bellman iteration failed to converge")
        max_used = max(max_used, iteration + 1)
        costs[:, goal] = updated[0]
    return costs, max_used


def shortest_control_distance(transitions: np.ndarray) -> np.ndarray:
    result = np.full((9, 9), np.inf)
    for source in range(9):
        result[source, source] = 0.0
        queue: deque[int] = deque([source])
        while queue:
            state = queue.popleft()
            for target in transitions[:, state]:
                if not np.isfinite(result[source, target]):
                    result[source, target] = result[source, state] + 1
                    queue.append(int(target))
    return result


def counter_model(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Outcome probabilities and post-outcome states for X/Z counter actions.

    Quantum states are ``+X,-X,+Z,-Z``.  The null model has one state and is a
    state-independent fair coin.  Axis ``+`` outcomes advance an external DFA.
    """
    if kind == "coin":
        probabilities = np.full((2, 1, 2), 0.5)
        next_states = np.zeros((2, 1, 2), dtype=int)
        return probabilities, next_states
    if kind != "qubit":
        raise ValueError(kind)
    probabilities = np.zeros((2, 4, 2), dtype=float)
    next_states = np.zeros((2, 4, 2), dtype=int)
    # Outcomes are minus=0, plus=1. Measurement X resets to state -X/+X.
    probabilities[0, 0] = [0.0, 1.0]
    probabilities[0, 1] = [1.0, 0.0]
    probabilities[0, 2:] = 0.5
    next_states[0, :, 0] = 1
    next_states[0, :, 1] = 0
    # Measurement Z resets to state -Z/+Z.
    probabilities[1, 2] = [0.0, 1.0]
    probabilities[1, 3] = [1.0, 0.0]
    probabilities[1, :2] = 0.5
    next_states[1, :, 0] = 3
    next_states[1, :, 1] = 2
    return probabilities, next_states


def solve_counter_goal(target_x: int, target_z: int, kind: str) -> tuple[float, np.ndarray]:
    """Optimal cost to accumulate X+ and Z+ outcomes up to a 2D goal.

    Progress is classical goal memory.  For the qubit, the initial density
    matrix is maximally mixed, so either first measurement produces its two
    eigenstates equiprobably.  The returned scalar includes that first action.
    """
    if target_x == 0 and target_z == 0:
        return 0.0, np.zeros((1, 1, 1))
    probabilities, next_states = counter_model(kind)
    state_count = probabilities.shape[1]
    value = np.zeros((target_x + 1, target_z + 1, state_count), dtype=float)
    value.fill(10.0 * (target_x + target_z + 1))
    value[target_x, target_z, :] = 0.0
    for _ in range(200_000):
        updated = value.copy()
        for count_x in range(target_x + 1):
            for count_z in range(target_z + 1):
                if count_x == target_x and count_z == target_z:
                    continue
                for state in range(state_count):
                    q_values = []
                    for action in range(2):
                        q = 1.0
                        for outcome in range(2):
                            new_x = min(target_x, count_x + int(action == 0 and outcome == 1))
                            new_z = min(target_z, count_z + int(action == 1 and outcome == 1))
                            q += probabilities[action, state, outcome] * value[
                                new_x, new_z, next_states[action, state, outcome]
                            ]
                        q_values.append(q)
                    updated[count_x, count_z, state] = min(q_values)
        if np.max(np.abs(updated - value)) < 1e-12:
            value = updated
            break
        value = updated
    else:
        raise RuntimeError("counter Bellman iteration failed to converge")

    if kind == "coin":
        initial = float(value[0, 0, 0])
    else:
        # At the mixed state, choose X or Z; each costs one and branches equally.
        initial_q = []
        for action in range(2):
            q = 1.0
            for outcome in range(2):
                new_x = min(target_x, int(action == 0 and outcome == 1))
                new_z = min(target_z, int(action == 1 and outcome == 1))
                q += 0.5 * value[new_x, new_z, next_states[action, 0, outcome]]
            initial_q.append(q)
        initial = float(min(initial_q))
    return initial, value


def counter_surfaces() -> tuple[np.ndarray, np.ndarray]:
    coin = np.zeros((3, 3), dtype=float)
    qubit = np.zeros((3, 3), dtype=float)
    for target_x in range(3):
        for target_z in range(3):
            coin[target_x, target_z] = solve_counter_goal(target_x, target_z, "coin")[0]
            qubit[target_x, target_z] = solve_counter_goal(target_x, target_z, "qubit")[0]
    return coin, qubit


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    qubit_rows: list[dict[str, float]],
    best_row: dict[str, float],
    hesse_coords: np.ndarray,
    excess_cost: np.ndarray,
    control_distance: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)

    theta = np.array([row["theta"] for row in qubit_rows])
    axes[0, 0].plot(theta, [row["pearson_euclidean"] for row in qubit_rows], label="distance correlation")
    axes[0, 0].plot(theta, [row["positive_gram_planarity"] for row in qubit_rows], label="2D positive-Gram fraction")
    axes[0, 0].plot(theta, [row["min_trace_separation"] for row in qubit_rows], label="minimum separation")
    axes[0, 0].axvline(best_row["theta"], color="black", ls="--", lw=1, label="selected compromise")
    axes[0, 0].set(xlabel=r"rotation angle $\theta$", ylabel="diagnostic", title="Qubit patch: locality versus distinguishability")
    axes[0, 0].legend(fontsize=8)

    for index, (x_coord, y_coord) in enumerate(hesse_coords):
        axes[0, 1].scatter(x_coord, y_coord, s=90, color="#235789")
        axes[0, 1].text(x_coord + 0.03, y_coord + 0.03, f"{index // 3},{index % 3}", fontsize=9)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set(title="Exact qutrit control-cost MDS", xlabel="MDS 1", ylabel="MDS 2")

    x_values = upper_values(control_distance)
    y_values = upper_values(excess_cost)
    jitter = np.linspace(-0.025, 0.025, len(x_values))
    axes[1, 0].scatter(x_values + jitter, y_values, alpha=0.8, color="#f18f01")
    axes[1, 0].set(
        xlabel="exact toroidal control distance",
        ylabel="excess optimal goal-hitting cost",
        title="Qutrit goals inherit the controlled transition graph",
    )

    states_zero, _ = qubit_patch(0.0)
    states_best, _ = qubit_patch(float(best_row["theta"]))
    null_quantum = upper_values(pairwise_trace_distance(states_zero))
    best_quantum = upper_values(pairwise_trace_distance(states_best))
    automaton = upper_values(open_manhattan())
    axes[1, 1].scatter(automaton, null_quantum, label=r"null qubit ($\theta=0$)", alpha=0.75)
    axes[1, 1].scatter(automaton, best_quantum, label="selected qubit patch", alpha=0.75)
    axes[1, 1].set(
        xlabel="goal-automaton Manhattan distance",
        ylabel="quantum trace distance",
        title="A perfect goal automaton can survive zero quantum geometry",
    )
    axes[1, 1].legend(fontsize=8)

    figure.savefig(FIGURES / "low_dimensional_search.png", dpi=220)
    plt.close(figure)


def make_qutrit_comparison_figure(
    hesse_overlap: np.ndarray,
    phase_overlap: np.ndarray,
    phase_excess: np.ndarray,
    control_distance: np.ndarray,
    coin_counter: np.ndarray,
    qubit_counter: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, 13)
    axes[0].hist(upper_values(hesse_overlap), bins=bins, alpha=0.7, label="Hesse SIC")
    axes[0].hist(upper_values(phase_overlap), bins=bins, alpha=0.7, label="phase grid")
    axes[0].set(
        xlabel=r"pairwise fidelity $|\langle\psi_i|\psi_j\rangle|^2$",
        ylabel="pair count",
        title="Same control topology, different state geometry",
    )
    axes[0].legend(fontsize=8)

    axes[1].scatter(upper_values(control_distance), upper_values(phase_excess), color="#7a5195", alpha=0.85)
    axes[1].set(
        xlabel="toroidal control distance",
        ylabel="phase-grid excess hitting cost",
        title="Measurement shortcuts slightly split distance two",
    )

    x = np.arange(3)
    for target_z in range(3):
        offset = (target_z - 1) * 0.08
        axes[2].plot(
            x + offset,
            coin_counter[:, target_z],
            "o--",
            color="#999999",
            alpha=0.65,
            label="independent coin" if target_z == 0 else None,
        )
        axes[2].plot(x + offset, qubit_counter[:, target_z], "o-", alpha=0.85, label=f"qubit, z={target_z}")
    axes[2].set(
        xticks=x,
        xlabel="required X+ count",
        ylabel="optimal expected actions",
        title="Qubit backaction distorts a 2D counter DFA",
    )
    axes[2].legend(fontsize=7)
    figure.savefig(FIGURES / "qutrit_and_counter_comparison.png", dpi=220)
    plt.close(figure)


def run(theta_count: int = 180) -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    qubit_rows = qubit_search(theta_count)
    eligible = [row for row in qubit_rows if row["min_trace_separation"] >= 0.12]
    best_row = max(eligible, key=lambda row: row["separation_adjusted_score"])
    write_csv(RESULTS / "qubit_rotation_search.csv", qubit_rows)

    best_states, _ = qubit_patch(float(best_row["theta"]))
    best_trace = pairwise_trace_distance(best_states)
    save_matrix(RESULTS / "qubit_selected_trace_distance.csv", best_trace)
    save_matrix(RESULTS / "automaton_open_manhattan.csv", open_manhattan())

    sic_states, x, z = hesse_sic()
    controls = [x, x.conj().T, z, z.conj().T]
    probabilities = sic_probabilities(sic_states)
    transitions = hesse_transitions(sic_states, controls)
    hitting_cost, policies, iterations = solve_sic_hitting_cost(probabilities, transitions)
    control_distance = shortest_control_distance(transitions)
    baseline = np.diag(hitting_cost)
    excess = hitting_cost - baseline[None, :]
    symmetric_excess = 0.5 * (excess + excess.T)
    hesse_coords, hesse_stress, hesse_eigenvalues = classical_mds(control_distance, 2)

    save_matrix(RESULTS / "qutrit_hesse_sic_probabilities.csv", probabilities)
    save_matrix(RESULTS / "qutrit_control_distance.csv", control_distance)
    save_matrix(RESULTS / "qutrit_goal_hitting_cost.csv", hitting_cost)
    save_matrix(RESULTS / "qutrit_goal_excess_cost.csv", symmetric_excess)
    save_matrix(RESULTS / "qutrit_optimal_policy.csv", policies)
    save_matrix(RESULTS / "qutrit_control_mds_coordinates.csv", hesse_coords)

    off_diagonal_overlaps = upper_values(np.abs(sic_states.conj() @ sic_states.T) ** 2)
    qutrit_summary = {
        "sic_completeness_residual": float(
            np.linalg.norm(sum(ket_density(state) / 3 for state in sic_states) - np.eye(3))
        ),
        "sic_off_diagonal_overlap_min": float(np.min(off_diagonal_overlaps)),
        "sic_off_diagonal_overlap_max": float(np.max(off_diagonal_overlaps)),
        "weyl_projective_commutator_residual": float(np.linalg.norm(x @ z - np.exp(-2j * np.pi / 3) * z @ x)),
        "bellman_iterations": iterations,
        "baseline_goal_cost": float(np.mean(baseline)),
        "control_distance_vs_excess_pearson": pearson(upper_values(control_distance), upper_values(symmetric_excess)),
        "control_distance_vs_excess_spearman": spearman(upper_values(control_distance), upper_values(symmetric_excess)),
        "control_distance_2d_mds_stress": hesse_stress,
        "control_distance_positive_gram_eigenvalues": [float(value) for value in hesse_eigenvalues],
        "distinct_excess_costs": [float(value) for value in np.unique(np.round(upper_values(symmetric_excess), 10))],
    }

    phase_states, u, v = qutrit_phase_grid()
    phase_controls = [u, u.conj().T, v, v.conj().T]
    phase_probabilities = sic_probabilities(phase_states)
    phase_transitions = hesse_transitions(phase_states, phase_controls)
    phase_cost, phase_policy, phase_iterations = solve_sic_hitting_cost(
        phase_probabilities, phase_transitions
    )
    phase_control_distance = shortest_control_distance(phase_transitions)
    phase_excess = phase_cost - np.diag(phase_cost)[None, :]
    phase_symmetric_excess = 0.5 * (phase_excess + phase_excess.T)
    phase_confirmations: dict[str, object] = {}
    for length in (1, 2, 3):
        confirmation_cost, confirmation_iterations = solve_confirmation_goal(
            phase_probabilities, phase_transitions, length
        )
        confirmation_excess = confirmation_cost - np.diag(confirmation_cost)[None, :]
        phase_confirmations[str(length)] = {
            "baseline": float(np.mean(np.diag(confirmation_cost))),
            "iterations": confirmation_iterations,
            "distance_correlation": pearson(
                upper_values(phase_control_distance),
                upper_values(0.5 * (confirmation_excess + confirmation_excess.T)),
            ),
        }
        save_matrix(
            RESULTS / f"qutrit_phase_confirmation_{length}_cost.csv", confirmation_cost
        )
    save_matrix(RESULTS / "qutrit_phase_probabilities.csv", phase_probabilities)
    save_matrix(RESULTS / "qutrit_phase_goal_hitting_cost.csv", phase_cost)
    save_matrix(RESULTS / "qutrit_phase_goal_excess_cost.csv", phase_symmetric_excess)
    save_matrix(RESULTS / "qutrit_phase_optimal_policy.csv", phase_policy)

    phase_overlap = np.abs(phase_states.conj() @ phase_states.T) ** 2
    phase_summary = {
        "povm_completeness_residual": float(
            np.linalg.norm(sum(ket_density(state) / 3 for state in phase_states) - np.eye(3))
        ),
        "commuting_control_residual": float(np.linalg.norm(u @ v - v @ u)),
        "distinct_off_diagonal_fidelities": [
            float(value) for value in np.unique(np.round(upper_values(phase_overlap), 10))
        ],
        "bellman_iterations": phase_iterations,
        "baseline_goal_cost": float(np.mean(np.diag(phase_cost))),
        "control_distance_vs_state_trace_pearson": pearson(
            upper_values(phase_control_distance),
            upper_values(pairwise_trace_distance(phase_states)),
        ),
        "control_distance_vs_excess_pearson": pearson(
            upper_values(phase_control_distance), upper_values(phase_symmetric_excess)
        ),
        "control_distance_vs_excess_spearman": spearman(
            upper_values(phase_control_distance), upper_values(phase_symmetric_excess)
        ),
        "distinct_excess_costs": [
            float(value) for value in np.unique(np.round(upper_values(phase_symmetric_excess), 10))
        ],
        "translated_confirmation_sequences": phase_confirmations,
    }

    coin_counter, qubit_counter = counter_surfaces()
    save_matrix(RESULTS / "coin_counter_goal_cost.csv", coin_counter)
    save_matrix(RESULTS / "qubit_counter_goal_cost.csv", qubit_counter)
    counter_summary = {
        "coin_cost_surface": coin_counter.tolist(),
        "qubit_cost_surface": qubit_counter.tolist(),
        "coin_additivity_residual": float(
            np.max(np.abs(coin_counter - coin_counter[:, :1] - coin_counter[:1, :]))
        ),
        "qubit_additivity_residual": float(
            np.max(np.abs(qubit_counter - qubit_counter[:, :1] - qubit_counter[:1, :]))
        ),
        "interpretation": "Both use the same external 3x3 counter DFA; only the qubit has backaction.",
    }
    automaton_summary = {
        "quantum_dimension": 2,
        "theta": 0.0,
        "number_of_goal_nodes": 9,
        "number_of_distinct_quantum_states": 1,
        "maximum_quantum_trace_distance": 0.0,
        "automaton_manhattan_2d_mds_stress": classical_mds(open_manhattan(), 2)[1],
        "warning": "The 3x3 geometry is wholly in external goal memory.",
    }
    summary = {
        "interpretive_contract": {
            "automaton_geometry": "distance induced by externally tracked goal progress",
            "state_geometry": "trace distance among conditional quantum states",
            "control_geometry": "shortest or optimal hitting cost under physical instruments",
        },
        "automaton_null": automaton_summary,
        "selected_qubit_patch": best_row,
        "qutrit_hesse": qutrit_summary,
        "qutrit_phase_grid": phase_summary,
        "sequence_counter_ablation": counter_summary,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_figure(qubit_rows, best_row, hesse_coords, symmetric_excess, control_distance)
    make_qutrit_comparison_figure(
        np.abs(sic_states.conj() @ sic_states.T) ** 2,
        phase_overlap,
        phase_symmetric_excess,
        phase_control_distance,
        coin_counter,
        qubit_counter,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-count", type=int, default=180)
    args = parser.parse_args()
    summary = run(theta_count=args.theta_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
