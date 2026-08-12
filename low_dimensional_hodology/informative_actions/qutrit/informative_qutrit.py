#!/usr/bin/env python3
"""Minimal informative qutrit actions and operational geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COORDS = tuple((x, y) for x in range(3) for y in range(3))


@lru_cache(maxsize=1)
def phase_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = np.exp(2j * np.pi / 3)
    states = np.array([[1, omega**x, omega**y] for x, y in COORDS], complex) / np.sqrt(3)
    u = np.diag([1, omega, 1]).astype(complex)
    v = np.diag([1, 1, omega]).astype(complex)
    return states, u, v


def rho(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def positive_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T


@lru_cache(maxsize=None)
def axis_instruments(eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Two three-outcome phase-axis POVMs and Lüders Kraus operators."""
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0,1]")
    omega = np.exp(2j * np.pi / 3)
    effects = np.zeros((2, 3, 3, 3), complex)
    for axis, blind in ((0, 2), (1, 1)):
        active = 1 if axis == 0 else 2
        for outcome in range(3):
            ket = np.zeros(3, complex)
            ket[0] = 1 / np.sqrt(2)
            ket[active] = omega**outcome / np.sqrt(2)
            sharp = (2 / 3) * rho(ket)
            sharp[blind, blind] += 1 / 3
            effects[axis, outcome] = eta * sharp + (1 - eta) * np.eye(3) / 3
    kraus = np.array([[positive_sqrt(effect) for effect in action] for action in effects])
    return effects, kraus


def phase_povm() -> np.ndarray:
    return np.array([rho(state) / 3 for state in phase_grid()[0]])


def hesse_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = np.exp(2j * np.pi / 3)
    x = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    z = np.diag([1, omega, omega**2]).astype(complex)
    fiducial = np.array([0, 1, -1], complex) / np.sqrt(2)
    states = np.array([np.linalg.matrix_power(x, m) @ np.linalg.matrix_power(z, n) @ fiducial for m in range(3) for n in range(3)])
    return states, np.array([rho(state) / 3 for state in states]), x, z


def hesse_povm() -> np.ndarray:
    return hesse_system()[1]


def hesse_kernel(unitary: np.ndarray | None = None) -> np.ndarray:
    """Outcome-to-next-outcome kernel of SIC preparation, control, SIC probe."""
    states, effects, _, _ = hesse_system()
    unitary = np.eye(3) if unitary is None else unitary
    return np.array([[born(rho(unitary @ state), effect) for effect in effects] for state in states])


def infer_kernel_permutation(kernel: np.ndarray) -> np.ndarray:
    """Infer an opaque control's action on predictive classes by unique maxima."""
    permutation = np.argmax(kernel, axis=1)
    if len(np.unique(permutation)) != len(permutation):
        raise ValueError("kernel does not induce a unique predictive-state permutation")
    return permutation


def compose_permutations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Composition applying ``first`` and then ``second``."""
    return second[first]


def permutation_order(permutation: np.ndarray) -> int:
    current = np.arange(len(permutation))
    for order in range(1, 100):
        current = permutation[current]
        if np.array_equal(current, np.arange(len(permutation))):
            return order
    raise ValueError("order exceeds search bound")


def permutation_group(generators: list[np.ndarray]) -> list[np.ndarray]:
    identity = np.arange(len(generators[0]))
    group = {tuple(identity): identity}
    frontier = [identity]
    while frontier:
        element = frontier.pop()
        for generator in generators:
            composed = compose_permutations(element, generator)
            key = tuple(composed)
            if key not in group:
                group[key] = composed
                frontier.append(composed)
    return list(group.values())


def bellman_hesse(tolerance: float = 1e-13) -> tuple[np.ndarray, int]:
    """Solve exact Hesse goal hitting with four opaque controls and SIC commit."""
    _, _, x, z = hesse_system()
    controls = [x, x.conj().T, z, z.conj().T]
    transitions = np.array([infer_kernel_permutation(hesse_kernel(control)) for control in controls])
    kernel = hesse_kernel()
    costs = np.zeros((9, 9))
    max_iterations = 0
    for goal in range(9):
        value = np.full(9, 10.0)
        for iteration in range(10000):
            move_q = 1 + value[transitions]
            probe_q = np.ones(9)
            mask = np.arange(9) != goal
            probe_q += kernel[:, mask] @ value[mask]
            updated = np.minimum(np.min(move_q, axis=0), probe_q)
            if np.max(np.abs(updated - value)) < tolerance:
                break
            value = updated
        costs[:, goal] = updated
        max_iterations = max(max_iterations, iteration + 1)
    return costs, max_iterations


def integrated_hesse_bellman(tolerance: float = 1e-13) -> tuple[np.ndarray, int]:
    """Bellman values when every translate action also reports and prepares.

    Action ``a`` has branches ``Pi_o U_a / sqrt(3)``. Outcome ``goal``
    terminates; otherwise the next predictive class is exactly ``o``.
    Identity reporting is included alongside the four local translations.
    """
    _, _, x, z = hesse_system()
    controls = [np.eye(3), x, x.conj().T, z, z.conj().T]
    kernels = np.array([hesse_kernel(control) for control in controls])
    costs = np.zeros((9, 9))
    max_iterations = 0
    for goal in range(9):
        value = np.full(9, 6.0)
        for iteration in range(10000):
            mask = np.arange(9) != goal
            q_values = np.array([1 + kernel[:, mask] @ value[mask] for kernel in kernels])
            updated = np.min(q_values, axis=0)
            if np.max(np.abs(updated - value)) < tolerance:
                break
            value = updated
        costs[:, goal] = updated
        max_iterations = max(max_iterations, iteration + 1)
    return costs, max_iterations


def weak_measure_prepare_kernel(unitary: np.ndarray, eta: float) -> np.ndarray:
    """Covariant noisy-report kernel whose outcome always prepares its SIC ray.

    This is implemented by measure-and-prepare Kraus refinements of effects
    ``eta Pi_o/3 + (1-eta) I/9``.  The unobserved refinement index is discarded.
    """
    return eta * hesse_kernel(unitary) + (1 - eta) * np.ones((9, 9)) / 9


def integrated_weak_bellman(eta: float, tolerance: float = 1e-13) -> np.ndarray:
    _, _, x, z = hesse_system()
    controls = [np.eye(3), x, x.conj().T, z, z.conj().T]
    kernels = np.array([weak_measure_prepare_kernel(control, eta) for control in controls])
    costs = np.zeros((9, 9))
    for goal in range(9):
        value = np.full(9, 9.0)
        for _ in range(20000):
            mask = np.arange(9) != goal
            updated = np.min(
                np.array([1 + kernel[:, mask] @ value[mask] for kernel in kernels]),
                axis=0,
            )
            if np.max(np.abs(updated - value)) < tolerance:
                break
            value = updated
        costs[:, goal] = updated
    return costs


def opaque_hesse_rows(seed: int, trials: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Infer the action group using only observed outcome transitions.

    A first SIC outcome prepares a predictive class. The learner sees only the
    opaque class token, opaque one-outcome action ID, and following SIC token.
    It never receives a hidden preparation coordinate or displacement label.
    """
    rng = np.random.default_rng(seed)
    _, _, x, z = hesse_system()
    controls = [np.eye(3), x, x.conj().T, z, z.conj().T]
    action_shuffle = rng.permutation(5)
    controls = [controls[i] for i in action_shuffle]
    outcome_shuffle = rng.permutation(9)
    inverse_outcome = np.argsort(outcome_shuffle)
    opaque_identity = hesse_kernel()[inverse_outcome][:, inverse_outcome]
    rows = []
    learned_permutations = []
    integrated_information = mutual_information(hesse_kernel()[:, None, :])
    for action_id, control in enumerate(controls):
        physical_kernel = hesse_kernel(control)
        opaque_kernel = physical_kernel[inverse_outcome][:, inverse_outcome]
        counts = np.full((9, 9), 0.5)
        for source_token in range(9):
            outcomes = rng.choice(9, size=trials, p=opaque_kernel[source_token])
            counts[source_token] += np.bincount(outcomes, minlength=9)
        estimate = counts / counts.sum(axis=1, keepdims=True)
        inferred = infer_kernel_permutation(estimate)
        exact = infer_kernel_permutation(opaque_kernel)
        learned_permutations.append(inferred)
        rows.append({"opaque_action": action_id, "physical_action_hidden": int(action_shuffle[action_id]), "permutation_accuracy": float(np.mean(inferred == exact)), "heldout_kernel_mae": float(np.mean(np.abs(estimate - opaque_kernel))), "order": permutation_order(inferred) if np.mean(inferred == exact) == 1 else -1, "immediate_action_outcome_mi_bits": integrated_information, "future_probe_effect_tv": float(np.mean(0.5 * np.sum(np.abs(opaque_kernel - opaque_identity), axis=1)))})
    group = permutation_group(learned_permutations)
    commute = all(np.array_equal(compose_permutations(a, b), compose_permutations(b, a)) for a in learned_permutations for b in learned_permutations)
    return rows, {"outcome_shuffle": outcome_shuffle.tolist(), "action_shuffle": action_shuffle.tolist(), "learned_group_order": len(group), "all_generators_commute": commute, "orbit_size_from_token_zero": len({int(element[0]) for element in group}), "identity_action_inferred_by_order_one": int(sum(row["order"] == 1 for row in rows) == 1)}


def graph_distance(permutations: list[np.ndarray]) -> np.ndarray:
    distance = np.full((9, 9), np.inf)
    for source in range(9):
        distance[source, source] = 0
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            for permutation in permutations:
                target = int(permutation[current])
                if not np.isfinite(distance[source, target]):
                    distance[source, target] = distance[source, current] + 1
                    frontier.append(target)
    return distance


def born(state: np.ndarray, effect: np.ndarray) -> float:
    return float(np.real(np.trace(effect @ state)))


def likelihoods(effects: np.ndarray) -> np.ndarray:
    """Return state x action x padded-outcome probabilities."""
    states = phase_grid()[0]
    if effects.ndim == 3:
        effects = effects[None, ...]
    return np.array([[[born(rho(state), effect) for effect in action] for action in effects] for state in states])


def mutual_information(table: np.ndarray) -> float:
    """I(hidden state; action,outcome) for uniform states/actions."""
    table = np.asarray(table, float)
    joint = table / (table.shape[0] * table.shape[1])
    outcome_action = joint.sum(axis=0, keepdims=True)
    prior = np.full((table.shape[0], 1, 1), 1 / table.shape[0])
    independent = prior * outcome_action
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))


def coordinate_information(table: np.ndarray, coordinate: int) -> float:
    """I(X or Y; action,outcome), marginalizing the other coordinate."""
    grouped = np.zeros((3, table.shape[1], table.shape[2]))
    for state, coord in enumerate(COORDS):
        grouped[coord[coordinate]] += table[state] / 3
    return mutual_information(grouped)


def average_survival(kraus: np.ndarray) -> float:
    result = []
    for ket in phase_grid()[0]:
        state = rho(ket)
        for action in kraus:
            channel = sum(operator @ state @ operator.conj().T for operator in action)
            result.append(born(channel, state))
    return float(np.mean(result))


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float), np.asarray(second, float)
    midpoint = 0.5 * (first + second)
    result = 0.0
    for distribution in (first, second):
        mask = distribution > 0
        result += 0.5 * np.sum(distribution[mask] * np.log2(distribution[mask] / midpoint[mask]))
    return float(result)


def predictive_distance(table: np.ndarray) -> np.ndarray:
    result = np.zeros((9, 9))
    for i in range(9):
        for j in range(i + 1, 9):
            squared = sum(js_divergence(table[i, action], table[j, action]) for action in range(table.shape[1]))
            result[i, j] = result[j, i] = np.sqrt(squared)
    return result


def classical_mds(distance: np.ndarray, dimensions: int = 2) -> tuple[np.ndarray, float, np.ndarray]:
    count = len(distance)
    center = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * center @ distance**2 @ center
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    coords = vectors[:, :dimensions] * np.sqrt(np.maximum(values[:dimensions], 0))
    fit = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    tri = np.triu_indices(count, 1)
    stress = float(np.sqrt(np.sum((distance[tri] - fit[tri]) ** 2) / np.sum(distance[tri] ** 2)))
    return coords, stress, values


def torus_hamming() -> np.ndarray:
    return np.array([[int(a[0] != b[0]) + int(a[1] != b[1]) for b in COORDS] for a in COORDS], float)


def pearson_matrices(first: np.ndarray, second: np.ndarray) -> float:
    tri = np.triu_indices(len(first), 1)
    return float(np.corrcoef(first[tri], second[tri])[0, 1])


def translate_index(index: int, dx: int, dy: int) -> int:
    x, y = COORDS[index]
    return ((x + dx) % 3) * 3 + (y + dy) % 3


def hidden_controls(seed: int = 10) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray]:
    _, u, v = phase_grid()
    named = [u, u.conj().T, v, v.conj().T]
    meanings = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    permutation = np.random.default_rng(seed).permutation(4)
    return [named[i] for i in permutation], [meanings[i] for i in permutation], permutation


def covariance_residual(unitary: np.ndarray, effects: np.ndarray, delta: tuple[int, int]) -> float:
    dx, dy = delta
    residuals = []
    for axis in range(2):
        shift = dx if axis == 0 else dy
        for outcome in range(3):
            transformed = unitary.conj().T @ effects[axis, outcome] @ unitary
            expected = effects[axis, (outcome - shift) % 3]
            residuals.append(np.linalg.norm(transformed - expected))
    return float(np.sqrt(np.mean(np.square(residuals))))


def infer_action_meanings(controls: list[np.ndarray], effects: np.ndarray) -> list[dict[str, object]]:
    candidates = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    rows = []
    for action, unitary in enumerate(controls):
        scores = [(covariance_residual(unitary, effects, delta), delta) for delta in candidates]
        scores.sort()
        rows.append({"action_id": action, "inferred_dx": scores[0][1][0], "inferred_dy": scores[0][1][1], "residual": scores[0][0], "margin": scores[1][0] - scores[0][0]})
    return rows


def haar_unitary(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    return q * (phases / np.abs(phases)).conj()


@dataclass
class Filter:
    weights: np.ndarray
    branches: np.ndarray

    @classmethod
    def uniform(cls) -> "Filter":
        return cls(np.full(9, 1 / 9), np.array([rho(state) for state in phase_grid()[0]]))

    def predict(self, effects: np.ndarray) -> np.ndarray:
        return np.array([sum(self.weights[s] * born(self.branches[s], effect) for s in range(9)) for effect in effects])

    def observe(self, action: int, outcome: int, effects: np.ndarray, kraus: np.ndarray) -> None:
        likes = np.array([born(state, effects[action, outcome]) for state in self.branches])
        self.weights *= likes
        self.weights /= self.weights.sum()
        for s in range(9):
            update = kraus[action, outcome] @ self.branches[s] @ kraus[action, outcome]
            self.branches[s] = update / np.trace(update)


def sample_measure(state: np.ndarray, action: int, effects: np.ndarray, kraus: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    probabilities = np.maximum([born(state, effect) for effect in effects[action]], 0)
    probabilities = np.asarray(probabilities) / np.sum(probabilities)
    outcome = int(rng.choice(3, p=probabilities))
    updated = kraus[action, outcome] @ state @ kraus[action, outcome]
    return outcome, updated / np.trace(updated)


def shortest_actions(source: int, goal: int, meanings: list[tuple[int, int]]) -> list[int]:
    sx, sy = COORDS[source]
    gx, gy = COORDS[goal]
    dx = (gx - sx) % 3
    dy = (gy - sy) % 3
    dx = 0 if dx == 0 else (1 if dx == 1 else -1)
    dy = 0 if dy == 0 else (1 if dy == 1 else -1)
    result = []
    if dx:
        result.append(meanings.index((dx, 0)))
    if dy:
        result.append(meanings.index((0, dy)))
    return result


def navigation(eta: float, senses: int, episodes: int, rng: np.random.Generator, *, known_start: bool = False) -> dict[str, float]:
    effects, kraus = axis_instruments(eta)
    controls, true_meanings, _ = hidden_controls()
    inferred_rows = infer_action_meanings(controls, effects if eta > 0 else axis_instruments(1)[0])
    inferred = [(int(row["inferred_dx"]), int(row["inferred_dy"])) for row in inferred_rows]
    success, fidelity, cost = [], [], []
    kets = phase_grid()[0]
    for _ in range(episodes):
        hidden, goal = int(rng.integers(9)), int(rng.integers(9))
        state = rho(kets[hidden])
        belief = Filter.uniform()
        if known_start:
            belief.weights[:] = 0
            belief.weights[hidden] = 1
        for t in range(senses):
            action = t % 2
            outcome, state = sample_measure(state, action, effects, kraus, rng)
            belief.observe(action, outcome, effects, kraus)
        estimate = int(np.flatnonzero(belief.weights >= belief.weights.max() - 1e-12)[0])
        actions = shortest_actions(estimate, goal, inferred)
        final_label = hidden
        for action in actions:
            unitary = controls[action]
            state = unitary @ state @ unitary.conj().T
            final_label = translate_index(final_label, *true_meanings[action])
        success.append(final_label == goal)
        fidelity.append(born(state, rho(kets[goal])))
        cost.append(senses + len(actions) + 1)
    values = np.asarray(success, float)
    fidelities = np.asarray(fidelity)
    return {"eta": eta, "senses": senses, "known_start": int(known_start), "episodes": episodes, "label_success": float(values.mean()), "label_success_se": float(values.std(ddof=1) / np.sqrt(episodes)), "target_fidelity": float(fidelities.mean()), "target_fidelity_se": float(fidelities.std(ddof=1) / np.sqrt(episodes)), "mean_cost": float(np.mean(cost))}


def sequence_prediction(eta: float, train: int, test: int, rng: np.random.Generator) -> dict[str, float]:
    """Held-out next-outcome prediction: marginal, learned Markov, exact filter."""
    effects, kraus = axis_instruments(eta)
    counts0 = np.full((2, 3), 0.5)
    counts1 = np.full((2, 3, 2, 3), 0.5)

    def episode(update: bool) -> tuple[float, float, float]:
        hidden = int(rng.integers(9))
        state = rho(phase_grid()[0][hidden])
        filt = Filter.uniform()
        previous = None
        losses = [0.0, 0.0, 0.0]
        scored = 0
        for _ in range(5):
            action = int(rng.integers(2))
            exact = filt.predict(effects[action])
            outcome, state = sample_measure(state, action, effects, kraus, rng)
            if not update:
                marginal = counts0[action] / counts0[action].sum()
                markov = marginal if previous is None else counts1[previous[0], previous[1], action] / counts1[previous[0], previous[1], action].sum()
                losses[0] -= np.log2(max(marginal[outcome], 1e-12))
                losses[1] -= np.log2(max(markov[outcome], 1e-12))
                losses[2] -= np.log2(max(exact[outcome], 1e-12))
                scored += 1
            else:
                counts0[action, outcome] += 1
                if previous is not None:
                    counts1[previous[0], previous[1], action, outcome] += 1
            filt.observe(action, outcome, effects, kraus)
            previous = (action, outcome)
        return *(loss / max(scored, 1) for loss in losses),

    for _ in range(train):
        episode(True)
    losses = np.array([episode(False) for _ in range(test)])
    return {"eta": eta, "train_sequences": train, "test_sequences": test, "marginal_nll_bits": float(losses[:, 0].mean()), "markov_nll_bits": float(losses[:, 1].mean()), "quantum_filter_nll_bits": float(losses[:, 2].mean())}


def candidate_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = []
    tradeoff = []
    for eta in np.linspace(0, 1, 21):
        effects, kraus = axis_instruments(float(eta))
        table = likelihoods(effects)
        tradeoff.append({"eta": eta, "joint_information_bits": mutual_information(table), "x_information_bits": coordinate_information(table, 0), "y_information_bits": coordinate_information(table, 1), "survival_fidelity": average_survival(kraus), "disturbance": 1 - average_survival(kraus)})
    named = {
        "null-axis-pair": axis_instruments(0)[0],
        "weak-axis-pair": axis_instruments(0.6)[0],
        "sharp-axis-pair": axis_instruments(1)[0],
        "phase-grid-povm": phase_povm(),
        "hesse-sic-povm": hesse_povm(),
        "computational-null": np.array([np.diag(np.eye(3)[i]) for i in range(3)]),
    }
    for name, effects in named.items():
        table = likelihoods(effects)
        flattened = table.reshape(9, -1)
        centered_rank = int(np.linalg.matrix_rank(flattened - flattened.mean(axis=0), tol=1e-10))
        candidates.append({"candidate": name, "actions": table.shape[1], "outcomes_per_action": table.shape[2], "joint_information_bits": mutual_information(table), "x_information_bits": coordinate_information(table, 0), "y_information_bits": coordinate_information(table, 1), "signature_rank": centered_rank, "distinct_signatures": len(np.unique(np.round(flattened, 10), axis=0))})
    return candidates, tradeoff


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_figures(candidates: list[dict[str, object]], tradeoff: list[dict[str, object]], geometry: dict[str, object], discovery: list[dict[str, object]], navigation_rows: list[dict[str, object]], prediction_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot([r["disturbance"] for r in tradeoff], [r["joint_information_bits"] for r in tradeoff], "o-")
    axes[0].set(xlabel="mean survival infidelity", ylabel="I(state; action,outcome), bits", title="Information--disturbance frontier")
    labels = [r["candidate"] for r in candidates]
    positions = np.arange(len(labels))
    axes[1].bar(positions - .18, [r["x_information_bits"] for r in candidates], .36, label="x information")
    axes[1].bar(positions + .18, [r["y_information_bits"] for r in candidates], .36, label="y information")
    axes[1].set(xticks=positions, xticklabels=labels, ylabel="bits", title="Candidate sensor meanings")
    axes[1].tick_params(axis="x", rotation=45); axes[1].legend(fontsize=8)
    coords = np.asarray(geometry["coordinates"])
    for i, (x, y) in enumerate(coords):
        axes[2].scatter(x, y, s=70, color="#235789"); axes[2].text(x+.01, y+.01, str(COORDS[i]), fontsize=8)
    axes[2].set_aspect("equal"); axes[2].set(title="MDS of operational predictions", xlabel="MDS 1", ylabel="MDS 2")
    fig.savefig(FIGURES / "candidate_search_and_geometry.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    axes[0].bar([str(row["action_id"]) for row in discovery], [row["residual"] for row in discovery])
    for index, row in enumerate(discovery): axes[0].text(index, row["residual"] + .005, f"({row['inferred_dx']},{row['inferred_dy']})", ha="center")
    axes[0].set(xlabel="hidden action ID", ylabel="covariance residual", title="Meanings inferred from future statistics")
    for eta in sorted(set(row["eta"] for row in navigation_rows)):
        selected = [row for row in navigation_rows if row["eta"] == eta and not row["known_start"]]
        axes[1].errorbar([r["mean_cost"] for r in selected], [r["label_success"] for r in selected], yerr=[r["label_success_se"] for r in selected], marker="o", label=fr"$\eta={eta}$")
    axes[1].axhline(1/9, color="gray", ls=":"); axes[1].set(xlabel="mean interventions", ylabel="goal-label success", title="Hidden-start navigation"); axes[1].legend(fontsize=8)
    axes[2].plot([r["eta"] for r in prediction_rows], [r["marginal_nll_bits"] for r in prediction_rows], "o-", label="marginal")
    axes[2].plot([r["eta"] for r in prediction_rows], [r["markov_nll_bits"] for r in prediction_rows], "o-", label="learned last-event")
    axes[2].plot([r["eta"] for r in prediction_rows], [r["quantum_filter_nll_bits"] for r in prediction_rows], "o-", label="exact predictive state")
    axes[2].set(xlabel=r"strength $\eta$", ylabel="held-out NLL (bits/outcome)", title="Future-action prediction"); axes[2].legend(fontsize=8)
    fig.savefig(FIGURES / "action_meanings_navigation_prediction.png", dpi=220); plt.close(fig)


def make_integrated_figure(
    weak_rows: list[dict[str, object]],
    word_distance: np.ndarray,
    integrated_cost: np.ndarray,
    separated_cost: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    image = axes[0].imshow(hesse_kernel(), cmap="Blues", vmin=0, vmax=1 / 3)
    axes[0].set(xlabel="next opaque outcome", ylabel="previous opaque outcome", title="Hesse measure--prepare kernel")
    fig.colorbar(image, ax=axes[0], label="probability")
    axes[1].plot([r["eta"] for r in weak_rows], [r["self_value"] for r in weak_rows], "o-", label="self")
    axes[1].plot([r["eta"] for r in weak_rows], [r["edge_value"] for r in weak_rows], "x--", lw=2, label="edge")
    axes[1].plot([r["eta"] for r in weak_rows], [r["diagonal_value"] for r in weak_rows], "o-", label="diagonal")
    axes[1].set(xlabel=r"informativeness $\eta$", ylabel="optimal report cost", title="Weakening never separates self from edge"); axes[1].legend()
    integrated_shell = [float(np.mean(np.diag(integrated_cost))), float(np.mean(integrated_cost[word_distance == 1])), float(np.mean(integrated_cost[word_distance == 2]))]
    separated_shell = [float(np.mean(np.diag(separated_cost))), float(np.mean(separated_cost[word_distance == 1])), float(np.mean(separated_cost[word_distance == 2]))]
    integrated_shell = np.array(integrated_shell) - integrated_shell[0]
    separated_shell = np.array(separated_shell) - separated_shell[0]
    locations = np.arange(3)
    axes[2].bar(locations - .18, integrated_shell, .36, label="integrated")
    axes[2].bar(locations + .18, separated_shell, .36, label="separated")
    axes[2].set(xticks=locations, xticklabels=("self", "edge", "diagonal"), ylabel="baseline-subtracted cost", title="Integration violates identity of indiscernibles"); axes[2].legend()
    fig.savefig(FIGURES / "integrated_action_no_go.png", dpi=220); plt.close(fig)


def run(episodes: int = 2000, seed: int = 20260812) -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    candidates, tradeoff = candidate_rows()
    effects = axis_instruments(0.6)[0]
    table = likelihoods(effects)
    distance = predictive_distance(table)
    coords, stress, eigenvalues = classical_mds(distance)
    geometry = {"eta": .6, "distance_vs_sqrt_hamming_correlation": pearson_matrices(distance, np.sqrt(torus_hamming())), "mds_2d_stress": stress, "positive_eigenvalues": [float(x) for x in eigenvalues if x > 1e-10], "coordinates": coords.tolist()}

    controls, true_meanings, permutation = hidden_controls()
    discovery = infer_action_meanings(controls, effects)
    for row, truth in zip(discovery, true_meanings): row.update({"true_dx": truth[0], "true_dy": truth[1], "correct": int((row["inferred_dx"], row["inferred_dy"]) == truth)})
    random_controls = [haar_unitary(rng) for _ in range(4)]
    random_discovery = infer_action_meanings(random_controls, effects)
    for row in random_discovery: row["control"] = "haar-random"
    for row in discovery: row["control"] = "phase-translation"

    opaque_rows, opaque_summary = opaque_hesse_rows(seed + 1, trials=2000)
    _, _, hesse_x, hesse_z = hesse_system()
    hesse_controls = [hesse_x, hesse_x.conj().T, hesse_z, hesse_z.conj().T]
    exact_permutations = [infer_kernel_permutation(hesse_kernel(control)) for control in hesse_controls]
    hesse_word_distance = graph_distance(exact_permutations)
    hesse_cost, hesse_iterations = bellman_hesse()
    hesse_excess = hesse_cost - np.diag(hesse_cost)[None, :]
    hesse_predictive_rows = hesse_kernel()
    hesse_predictive_rank = int(
        np.linalg.matrix_rank(
            hesse_predictive_rows - hesse_predictive_rows.mean(axis=0), tol=1e-10
        )
    )
    integrated_cost, integrated_iterations = integrated_hesse_bellman()
    weak_integrated_rows = []
    for weak_eta in np.linspace(0, 1, 11):
        weak_cost = integrated_weak_bellman(float(weak_eta))
        self_value = float(np.mean(np.diag(weak_cost)))
        edge_value = float(np.mean(weak_cost[hesse_word_distance == 1]))
        diagonal_value = float(np.mean(weak_cost[hesse_word_distance == 2]))
        weak_integrated_rows.append({"eta": weak_eta, "self_value": self_value, "edge_value": edge_value, "diagonal_value": diagonal_value, "self_edge_gap": edge_value - self_value, "edge_diagonal_gap": diagonal_value - edge_value, "identity_of_indiscernibles": int(edge_value - self_value > 1e-9)})
    hesse_summary = {
        "immediate_sic_information_bits": mutual_information(hesse_kernel()[:, None, :]),
        "predictive_equivalence_classes": len(np.unique(np.round(hesse_predictive_rows, 12), axis=0)),
        "predictive_signature_rank": hesse_predictive_rank,
        "learned_opaque_action_group": opaque_summary,
        "bellman_iterations": hesse_iterations,
        "bellman_baseline": float(np.mean(np.diag(hesse_cost))),
        "bellman_excess_vs_word_distance_correlation": pearson_matrices(
            0.5 * (hesse_excess + hesse_excess.T), hesse_word_distance
        ),
        "distinct_bellman_excess": np.unique(np.round(hesse_excess, 10)).tolist(),
        "word_metric_2d_mds_stress": classical_mds(hesse_word_distance)[1],
        "integrated_action_definition": "K[a,o] = Pi[o] U[a] / sqrt(3)",
        "integrated_bellman_iterations": integrated_iterations,
        "integrated_distinct_values": np.unique(np.round(integrated_cost, 10)).tolist(),
        "integrated_value_at_goal": float(np.mean(np.diag(integrated_cost))),
        "integrated_edge_value": float(np.mean(integrated_cost[hesse_word_distance == 1])),
        "integrated_diagonal_value": float(np.mean(integrated_cost[hesse_word_distance == 2])),
        "integrated_offdiagonal_shell_correlation_only": pearson_matrices(
            integrated_cost, hesse_word_distance
        ),
        "integrated_identity_of_indiscernibles_holds": False,
        "integrated_metric_verdict": "not a metric: self and nearest-neighbor values coincide",
    }
    haar_kernels = [hesse_kernel(unitary) for unitary in random_controls]
    future_benchmark = []
    for index, kernel in enumerate([hesse_kernel(control) for control in hesse_controls]):
        future_benchmark.append({"family": "weyl-translation", "opaque_action": index, "immediate_information_bits": 0.0, "future_effect_from_identity_tv": float(np.mean(0.5 * np.sum(np.abs(kernel - hesse_kernel()), axis=1))), "unique_future_kernel": 1})
    for index, kernel in enumerate(haar_kernels):
        future_benchmark.append({"family": "haar-random-unitary", "opaque_action": index, "immediate_information_bits": 0.0, "future_effect_from_identity_tv": float(np.mean(0.5 * np.sum(np.abs(kernel - hesse_kernel()), axis=1))), "unique_future_kernel": int(all(np.linalg.norm(kernel - other) > 1e-8 for j, other in enumerate(haar_kernels) if j != index))})

    navigation_rows = []
    for eta in (0.0, 0.3, 0.6, 1.0):
        for senses in (0, 2, 4, 6): navigation_rows.append(navigation(eta, senses, episodes, rng))
    navigation_rows.append(navigation(0.6, 0, episodes, rng, known_start=True))
    prediction_rows = [sequence_prediction(eta, 5000, 2000, rng) for eta in (0.0, 0.3, 0.6, 1.0)]

    write_csv(RESULTS / "candidate_instruments.csv", candidates)
    write_csv(RESULTS / "information_disturbance.csv", tradeoff)
    write_csv(RESULTS / "action_discovery.csv", discovery + random_discovery)
    write_csv(RESULTS / "opaque_hesse_action_learning.csv", opaque_rows)
    write_csv(RESULTS / "future_effect_controls.csv", future_benchmark)
    write_csv(RESULTS / "integrated_weak_scan.csv", weak_integrated_rows)
    control_rows = [
        {"control": "integrated-hesse", "immediate_information_bits": hesse_summary["immediate_sic_information_bits"], "predictive_classes": 9, "translation_group_order": 9, "metric_identity_holds": 0, "interpretation": "informative action topology, nonmetric Bellman value"},
        {"control": "separated-weyl-sic", "immediate_information_bits": 0.0, "predictive_classes": 9, "translation_group_order": 9, "metric_identity_holds": 1, "interpretation": "exact word hodology but movement outcome uninformative"},
        {"control": "haar-unitary-future-probe", "immediate_information_bits": 0.0, "predictive_classes": 9, "translation_group_order": 0, "metric_identity_holds": 0, "interpretation": "identifiable future effects without translation closure"},
        {"control": "null-quantum", "immediate_information_bits": 0.0, "predictive_classes": 1, "translation_group_order": 0, "metric_identity_holds": 0, "interpretation": "no quantum geometry"},
        {"control": "external-automaton-null-quantum", "immediate_information_bits": 0.0, "predictive_classes": 1, "translation_group_order": 0, "metric_identity_holds": 0, "interpretation": "nine hand-coded DFA nodes, quantum rank zero"},
    ]
    write_csv(RESULTS / "dialectical_controls.csv", control_rows)
    write_csv(RESULTS / "navigation.csv", navigation_rows)
    write_csv(RESULTS / "heldout_prediction.csv", prediction_rows)
    np.savetxt(RESULTS / "predictive_distance.csv", distance, delimiter=",")
    np.savetxt(RESULTS / "predictive_mds.csv", coords, delimiter=",")
    np.savetxt(RESULTS / "hesse_measure_prepare_kernel.csv", hesse_kernel(), delimiter=",")
    np.savetxt(RESULTS / "hesse_word_distance.csv", hesse_word_distance, delimiter=",")
    np.savetxt(RESULTS / "hesse_bellman_cost.csv", hesse_cost, delimiter=",")
    np.savetxt(RESULTS / "hesse_integrated_action_cost.csv", integrated_cost, delimiter=",")

    null_table = likelihoods(axis_instruments(0)[0])
    controls_permuted = [controls[i] for i in (2, 0, 3, 1)]
    permuted_discovery = infer_action_meanings(controls_permuted, effects)
    manifest = {"seed": seed, "episodes_per_navigation_condition": episodes, "total_navigation_episodes": episodes * len(navigation_rows), "geometry": geometry, "hesse_measure_prepare": hesse_summary, "action_permutation": permutation.tolist(), "all_translation_meanings_recovered": all(row["correct"] for row in discovery), "translation_max_residual": max(row["residual"] for row in discovery), "haar_random_min_residual": min(row["residual"] for row in random_discovery), "null_predictive_rank": int(np.linalg.matrix_rank(null_table.reshape(9, -1) - null_table.reshape(9, -1).mean(axis=0), tol=1e-10)), "external_automaton_quantum_rank": 0, "external_automaton_nodes": 9, "permuted_action_meanings": [(row["inferred_dx"], row["inferred_dy"]) for row in permuted_discovery], "interpretation": "Action meanings are automorphisms of future outcome laws; action and outcome names are gauge."}
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    make_figures(candidates, tradeoff, geometry, discovery, navigation_rows, prediction_rows)
    make_integrated_figure(weak_integrated_rows, hesse_word_distance, integrated_cost, hesse_cost)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000); parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args(); print(json.dumps(run(args.episodes, args.seed), indent=2))


if __name__ == "__main__": main()
