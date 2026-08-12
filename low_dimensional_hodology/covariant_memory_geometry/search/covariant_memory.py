"""Exact covariant qutrit memory instruments and deterministic diagnostics.

The finite predictive states are the nine Hesse-SIC rays.  Each integrated
action has one observed full-operator-rank memory branch and nine observed reset
branches.  The memory branch translates the current ray without measuring it;
the reset branches report information and prepare a reported Hesse ray.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log2
from typing import Mapping, Sequence

import numpy as np


COORDS = tuple(product(range(3), repeat=2))
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def density(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def hesse_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = np.exp(2j * np.pi / 3)
    x = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    z = np.diag([1, omega, omega**2]).astype(complex)
    fiducial = np.array([0, 1, -1], complex) / np.sqrt(2)
    states = np.array(
        [np.linalg.matrix_power(x, m) @ np.linalg.matrix_power(z, n) @ fiducial for m, n in COORDS]
    )
    projectors = np.array([density(state) for state in states])
    return states, projectors, x, z


def translate_index(index: int, displacement: tuple[int, int]) -> int:
    x, y = COORDS[index]
    dx, dy = displacement
    return ((x + dx) % 3) * 3 + (y + dy) % 3


def translation_unitary(displacement: tuple[int, int]) -> np.ndarray:
    _, _, x, z = hesse_system()
    return np.linalg.matrix_power(x, displacement[0] % 3) @ np.linalg.matrix_power(z, displacement[1] % 3)


def sharp_hesse_kernel(displacement: tuple[int, int]) -> np.ndarray:
    """P(report | Hesse state, displacement), shape state x report."""
    states, projectors, _, _ = hesse_system()
    unitary = translation_unitary(displacement)
    return np.array(
        [[np.real(np.trace(projectors[o] @ density(unitary @ states[s]))) / 3 for o in range(9)] for s in range(9)]
    )


def reset_kernel(displacement: tuple[int, int], sharpness: float) -> np.ndarray:
    if not 0 <= sharpness <= 1:
        raise ValueError("sharpness must be in [0,1]")
    return sharpness * sharp_hesse_kernel(displacement) + (1.0 - sharpness) / 9.0


@dataclass(frozen=True)
class MemoryInstrument:
    """Group-covariant integrated action with retained-memory probability mu.

    Outcome ``memory`` has Kraus sqrt(mu) U_a and maps a Hesse predictive class
    exactly to its translate. Outcome ``reset:o`` has a measure-and-prepare CP
    map with effect (1-mu)[xi Pi_o/3 + (1-xi)I/9] and output Pi_o.
    For mu<1 and xi<1 each reset branch has Choi/Kraus rank three; for mu>0
    the memory branch has Choi rank one and a full-rank Kraus operator.
    """

    memory: float
    sharpness: float

    def __post_init__(self) -> None:
        if not 0 <= self.memory <= 1 or not 0 <= self.sharpness <= 1:
            raise ValueError("parameters must be in [0,1]")

    def transition_branches(self, displacement: tuple[int, int]) -> np.ndarray:
        """P(outcome,next_state | state), outcomes memory, reset:0..8."""
        result = np.zeros((9, 10, 9))
        for state in range(9):
            result[state, 0, translate_index(state, displacement)] = self.memory
            for outcome, probability in enumerate(reset_kernel(displacement, self.sharpness)[state]):
                result[state, outcome + 1, outcome] = (1.0 - self.memory) * probability
        return result

    def kraus(self, displacement: tuple[int, int]) -> Mapping[str, tuple[np.ndarray, ...]]:
        states, projectors, _, _ = hesse_system()
        unitary = translation_unitary(displacement)
        branches: dict[str, tuple[np.ndarray, ...]] = {
            "memory": (np.sqrt(self.memory) * unitary,)
        }
        identity = np.eye(3, dtype=complex)
        for outcome in range(9):
            effect = (1.0 - self.memory) * (
                self.sharpness * projectors[outcome] / 3.0
                + (1.0 - self.sharpness) * identity / 9.0
            )
            values, vectors = np.linalg.eigh(effect)
            sqrt_effect = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T
            operators = []
            for basis in np.eye(3, dtype=complex):
                operators.append(np.outer(states[outcome], basis.conj()) @ sqrt_effect @ unitary)
            branches[f"reset:{outcome}"] = tuple(operators)
        return branches

    def immediate_information_bits(self) -> float:
        """I(pre-action class; observed branch) under a uniform class prior."""
        table = self.transition_branches((0, 0)).sum(axis=2)
        joint = table / 9.0
        marginal = joint.sum(axis=0, keepdims=True)
        mask = joint > 0
        independent = np.ones((9, 1)) / 9.0 * marginal
        return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))


def state_hitting_bellman(model: MemoryInstrument, tolerance: float = 1e-13) -> tuple[np.ndarray, int]:
    """Optimal costs with V_g(g)=0 and all five integrated actions unit cost."""
    transitions = [model.transition_branches(move).sum(axis=1) for move in MOVES]
    values = np.zeros((9, 9))
    maximum_iterations = 0
    for goal in range(9):
        value = np.zeros(9)
        value[np.arange(9) != goal] = 8.0
        for iteration in range(50000):
            q = np.array([1.0 + kernel @ value for kernel in transitions])
            updated = np.min(q, axis=0)
            updated[goal] = 0.0
            if np.max(np.abs(updated - value)) < tolerance:
                value = updated
                break
            value = updated
        values[:, goal] = value
        maximum_iterations = max(maximum_iterations, iteration + 1)
    return values, maximum_iterations


def analytic_shell_values(model: MemoryInstrument) -> tuple[float, float]:
    """Closed-form edge and diagonal costs for the covariant policy.

    Let b=(4-xi)/36 be any nonpeak reset probability.  Moving an edge toward
    the goal and a diagonal toward an edge reduces Bellman optimality to two
    linear equations.  Numerical Bellman iteration independently verifies that
    these actions are optimal throughout the searched parameter rectangle.
    """
    mu, xi = model.memory, model.sharpness
    c = (1.0 - mu) * (4.0 - xi) / 9.0
    h = mu + (1.0 - mu) * (16.0 + 5.0 * xi) / 36.0
    common = 1.0 - c
    edge = 1.0 / (common**2 - c * h)
    diagonal = (1.0 + h * edge) / common
    return edge, diagonal


def analytic_distance(model: MemoryInstrument) -> np.ndarray:
    edge, diagonal = analytic_shell_values(model)
    target = torus_distance()
    return np.where(target == 0, 0.0, np.where(target == 1, edge, diagonal))


def torus_distance() -> np.ndarray:
    return np.array(
        [[int(a[0] != b[0]) + int(a[1] != b[1]) for b in COORDS] for a in COORDS], dtype=float
    )


def metric_diagnostics(distance: np.ndarray) -> dict[str, float | bool]:
    symmetric = float(np.max(np.abs(distance - distance.T)))
    diagonal = float(np.max(np.abs(np.diag(distance))))
    off = distance[~np.eye(len(distance), dtype=bool)]
    minimum_off = float(np.min(off))
    triangle = 0.0
    for i, j, k in product(range(len(distance)), repeat=3):
        triangle = max(triangle, float(distance[i, k] - distance[i, j] - distance[j, k]))
    return {
        "symmetry_residual": symmetric,
        "diagonal_residual": diagonal,
        "minimum_offdiagonal": minimum_off,
        "maximum_triangle_violation": triangle,
        "is_metric": symmetric < 1e-10 and diagonal < 1e-10 and minimum_off > 1e-10 and triangle < 1e-10,
    }


def torus_diagnostics(distance: np.ndarray) -> dict[str, float]:
    target = torus_distance()
    mask = np.triu(np.ones_like(target, dtype=bool), 1)
    x, y = target[mask], distance[mask]
    scale = float(np.dot(x, y) / np.dot(x, x))
    fitted = scale * x
    relative_rmse = float(np.sqrt(np.mean((y - fitted) ** 2)) / np.mean(y))
    correlation = float(np.corrcoef(x, y)[0, 1])
    shell1 = y[x == 1]
    shell2 = y[x == 2]
    return {
        "torus_correlation": correlation,
        "best_scale": scale,
        "relative_rmse_after_scale": relative_rmse,
        "edge_mean": float(np.mean(shell1)),
        "edge_spread": float(np.ptp(shell1)),
        "diagonal_mean": float(np.mean(shell2)),
        "diagonal_spread": float(np.ptp(shell2)),
        "strict_shell_margin": float(np.min(shell2) - np.max(shell1)),
        "heldout_diagonal_additivity_error": float(np.mean(np.abs(shell2 - 2.0 * np.mean(shell1))) / np.mean(shell2)),
    }


def schoenberg(distance: np.ndarray, dimensions: int = 2) -> dict[str, float | int]:
    n = len(distance)
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ distance**2 @ center
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = np.maximum(values[:dimensions], 0.0)
    coordinates = vectors[:, :dimensions] * np.sqrt(positive)
    fitted = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    mask = np.triu(np.ones_like(distance, dtype=bool), 1)
    stress = float(np.sqrt(np.sum((distance[mask] - fitted[mask]) ** 2) / np.sum(distance[mask] ** 2)))
    mass = np.sum(np.abs(values))
    return {
        "negative_eigenmass_fraction": float(np.sum(np.abs(values[values < -1e-10])) / mass),
        "positive_dimension": int(np.sum(values > 1e-10)),
        "mds_2d_stress": stress,
        "minimum_gram_eigenvalue": float(np.min(values)),
    }


def covariance_residual(model: MemoryInstrument) -> float:
    """Classical predictive covariance of every outcome-labelled branch."""
    worst = 0.0
    for group in COORDS:
        state_perm = np.array([translate_index(s, group) for s in range(9)])
        outcome_perm = state_perm
        for move in MOVES:
            kernel = model.transition_branches(move)
            for state in range(9):
                expected = np.zeros((10, 9))
                expected[0, state_perm[np.argmax(kernel[state, 0])]] = kernel[state, 0].sum()
                for outcome in range(9):
                    expected[outcome_perm[outcome] + 1, outcome_perm[outcome]] = kernel[state, outcome + 1].sum()
                worst = max(worst, float(np.max(np.abs(model.transition_branches(move)[state_perm[state]] - expected))))
    return worst


def kraus_residual(model: MemoryInstrument) -> float:
    worst = 0.0
    for move in MOVES:
        branches = model.kraus(move)
        gram = sum((operator.conj().T @ operator for operators in branches.values() for operator in operators), np.zeros((3, 3), complex))
        worst = max(worst, float(np.max(np.abs(gram - np.eye(3)))))
    return worst


def branch_rank_diagnostics(
    model: MemoryInstrument, displacement: tuple[int, int] = (0, 0), tolerance: float = 1e-7
) -> dict[str, int]:
    """Separate minimal branch Choi rank from individual Kraus-matrix rank."""
    branches = model.kraus(displacement)

    def choi_rank(operators: Sequence[np.ndarray]) -> int:
        vectors = np.stack([operator.reshape(-1) for operator in operators], axis=1)
        return int(np.linalg.matrix_rank(vectors, tol=tolerance))

    def maximum_operator_rank(operators: Sequence[np.ndarray]) -> int:
        return max(int(np.linalg.matrix_rank(operator, tol=tolerance)) for operator in operators)

    resets = [operators for name, operators in branches.items() if name.startswith("reset:")]
    reset_choi = [choi_rank(operators) for operators in resets]
    reset_operator = [maximum_operator_rank(operators) for operators in resets]
    return {
        "memory_branch_choi_rank": choi_rank(branches["memory"]),
        "memory_branch_maximum_operator_rank": maximum_operator_rank(branches["memory"]),
        "reset_branch_minimum_choi_rank": min(reset_choi),
        "reset_branch_maximum_choi_rank": max(reset_choi),
        "reset_branch_maximum_operator_rank": max(reset_operator),
    }


def geodesic_path_closure(model: MemoryInstrument) -> dict[str, float | int]:
    """Nonselective channel closure for equal-length words with same displacement.

    Covariance makes all equal-length same-displacement words identical. Words
    with cancelling detours have extra common noise and are intentionally
    audited separately.
    """
    states, projectors, _, _ = hesse_system()
    effects = model.sharpness * projectors / 3.0 + (1.0 - model.sharpness) * np.eye(3) / 9.0
    unitaries = {move: translation_unitary(move) for move in MOVES[1:]}

    def channel(state: np.ndarray, move: tuple[int, int]) -> np.ndarray:
        unitary = unitaries[move]
        translated = unitary @ state @ unitary.conj().T
        reset = sum(
            (np.real(np.trace(effect @ translated)) * projector for effect, projector in zip(effects, projectors)),
            np.zeros((3, 3), complex),
        )
        return model.memory * translated + (1.0 - model.memory) * reset

    words = [tuple(word) for length in range(5) for word in product(MOVES[1:], repeat=length)]
    groups: dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]] = {}
    displacement_groups: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}
    for word in words:
        displacement = (sum(a[0] for a in word) % 3, sum(a[1] for a in word) % 3)
        groups.setdefault((displacement[0], displacement[1], len(word)), []).append(word)
        displacement_groups.setdefault(displacement, []).append(word)

    def endpoint(word) -> np.ndarray:
        state = density(states[0])
        for move in word:
            state = channel(state, move)
        return state

    equal_length_worst = 0.0
    for members in groups.values():
        endpoints = [endpoint(word) for word in members]
        for first in endpoints:
            for second in endpoints:
                equal_length_worst = max(equal_length_worst, float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(first - second)))))
    all_length_worst = 0.0
    for members in displacement_groups.values():
        endpoints = [endpoint(word) for word in members]
        for first in endpoints:
            for second in endpoints:
                all_length_worst = max(all_length_worst, float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(first - second)))))
    return {
        "equal_length_same_displacement_classes": len(groups),
        "equal_length_path_closure_residual": equal_length_worst,
        "all_length_path_closure_residual": all_length_worst,
    }


def search_grid() -> list[dict[str, float | bool]]:
    rows = []
    for memory in np.linspace(0.0, 0.95, 20):
        for sharpness in np.linspace(0.1, 1.0, 19):
            model = MemoryInstrument(float(memory), float(sharpness))
            distance = analytic_distance(model)
            iterations = 0
            metric = metric_diagnostics(distance)
            torus = torus_diagnostics(distance)
            mi = model.immediate_information_bits()
            # Reward retained memory, information, shell separation, and low
            # torus distortion without privileging hidden coordinate names.
            score = (
                2.0 * mi
                + 0.35 * memory
                + 0.12 * torus["strict_shell_margin"]
                - 3.0 * torus["relative_rmse_after_scale"]
            )
            rows.append(
                {
                    "memory": float(memory), "sharpness": float(sharpness),
                    "immediate_mi_bits": mi, "score": score,
                    "bellman_iterations": iterations, **metric, **torus,
                }
            )
    return sorted(rows, key=lambda row: (-float(row["score"]), -float(row["immediate_mi_bits"])))


def exhaustive_bellman_verification() -> list[dict[str, float | bool | int]]:
    """Independently iterate Bellman equations at every point in the 380-point grid."""
    rows = []
    for memory in np.linspace(0.0, 0.95, 20):
        for sharpness in np.linspace(0.1, 1.0, 19):
            model = MemoryInstrument(float(memory), float(sharpness))
            numerical, iterations = state_hitting_bellman(model)
            error = float(np.max(np.abs(numerical - analytic_distance(model))))
            rows.append(
                {
                    "memory": float(memory),
                    "sharpness": float(sharpness),
                    "analytic_bellman_max_error": error,
                    "bellman_iterations": iterations,
                    "is_metric": bool(metric_diagnostics(numerical)["is_metric"]),
                }
            )
    return rows


def oracle_labelled_action_audit(
    model: MemoryInstrument, trials_per_state_action: int, seed: int
) -> tuple[float, float, list[np.ndarray]]:
    """Oracle benchmark using latent source/successor labels, not agent observations."""
    rng = np.random.default_rng(seed)
    action_shuffle = rng.permutation(len(MOVES))
    token_shuffle = rng.permutation(9)
    inverse_token = np.argsort(token_shuffle)
    learned = []
    accuracies, maes = [], []
    for opaque_action in range(len(MOVES)):
        move = MOVES[action_shuffle[opaque_action]]
        exact = np.array([translate_index(s, move) for s in range(9)])
        counts = np.full((9, 9), 0.25)
        for opaque_source in range(9):
            physical_source = inverse_token[opaque_source]
            target = exact[physical_source]
            memory_trials = rng.binomial(trials_per_state_action, model.memory)
            counts[opaque_source, token_shuffle[target]] += memory_trials
        estimate = counts / counts.sum(axis=1, keepdims=True)
        inferred = np.argmax(estimate, axis=1)
        opaque_exact = token_shuffle[exact[inverse_token]]
        learned.append(inferred)
        accuracies.append(np.mean(inferred == opaque_exact))
        target_kernel = np.zeros((9, 9)); target_kernel[np.arange(9), opaque_exact] = 1.0
        maes.append(np.mean(np.abs(estimate - target_kernel)))
    return float(np.mean(accuracies)), float(np.mean(maes)), learned


def _compose_permutations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return second[first]


def _generated_permutation_group(generators: Sequence[np.ndarray]) -> list[np.ndarray]:
    identity = np.arange(9)
    found = {tuple(identity): identity}
    frontier = [identity]
    while frontier:
        element = frontier.pop()
        for generator in generators:
            product_permutation = _compose_permutations(element, generator)
            key = tuple(int(x) for x in product_permutation)
            if key not in found:
                found[key] = product_permutation
                frontier.append(product_permutation)
    return list(found.values())


def _permutation_order(permutation: np.ndarray) -> int:
    identity = np.arange(9)
    power = identity
    for candidate in range(1, 28):
        power = _compose_permutations(power, permutation)
        if np.array_equal(power, identity):
            return candidate
    return -1


def generate_observable_action_probe_records(
    model: MemoryInstrument, attempts_per_anchor_action: int, seed: int
) -> tuple[list[tuple[int, int, int]], dict[str, np.ndarray | int]]:
    """Generate learner-visible anchor/action/future-report triples.

    Operational protocol for each trial:

    1. a sharp Hesse preparation reports an opaque anchor token and prepares
       that Hesse ray;
    2. an opaque integrated action is applied and only trials with the observed
       ``memory`` branch are retained;
    3. the same sharp Hesse instrument (identity control) is used as a common
       future probe, producing one opaque report token.

    The learner-visible records never contain the translated state.  Persistent
    action/token permutations and the physical maps are returned separately for
    offline gauge-aware scoring only.
    """
    if attempts_per_anchor_action <= 0:
        raise ValueError("attempts_per_anchor_action must be positive")
    if model.memory <= 0:
        raise ValueError("observable memory-branch probing requires memory > 0")
    rng = np.random.default_rng(seed)
    action_shuffle = rng.permutation(len(MOVES))  # opaque action -> physical action
    token_shuffle = rng.permutation(9)  # physical Hesse token -> opaque token
    inverse_token = np.argsort(token_shuffle)
    probe_kernel = sharp_hesse_kernel((0, 0))
    records: list[tuple[int, int, int]] = []
    attempts = 0
    for opaque_anchor in range(9):
        physical_anchor = int(inverse_token[opaque_anchor])
        for opaque_action in range(len(MOVES)):
            move = MOVES[int(action_shuffle[opaque_action])]
            translated_state = translate_index(physical_anchor, move)
            for _ in range(attempts_per_anchor_action):
                attempts += 1
                if rng.random() >= model.memory:
                    continue
                physical_report = int(rng.choice(9, p=probe_kernel[translated_state]))
                records.append((opaque_anchor, opaque_action, int(token_shuffle[physical_report])))
    private_audit = {
        "action_shuffle": action_shuffle,
        "token_shuffle": token_shuffle,
        "attempts": attempts,
        "accepted_memory_events": len(records),
    }
    return records, private_audit


def learn_action_maps_from_observable_probes(
    records: Sequence[tuple[int, int, int]], pseudocount: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Infer action maps using only opaque anchor/action/future-report triples."""
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    counts = np.full((len(MOVES), 9, 9), pseudocount)
    for anchor_token, action_token, report_token in records:
        counts[action_token, anchor_token, report_token] += 1.0
    laws = counts / counts.sum(axis=2, keepdims=True)
    return np.argmax(laws, axis=2), laws


def observable_action_probe_audit(
    model: MemoryInstrument, attempts_per_anchor_action: int, seed: int
) -> tuple[dict[str, float | int | bool | str], list[dict[str, float | int]]]:
    """Run observable-only action learning; use hidden labels only for final scoring."""
    records, private = generate_observable_action_probe_records(model, attempts_per_anchor_action, seed)
    inferred, laws = learn_action_maps_from_observable_probes(records)
    action_shuffle = np.asarray(private["action_shuffle"])
    token_shuffle = np.asarray(private["token_shuffle"])
    inverse_token = np.argsort(token_shuffle)
    exact = []
    per_action = []
    for opaque_action in range(len(MOVES)):
        move = MOVES[int(action_shuffle[opaque_action])]
        physical_targets = np.array([translate_index(int(s), move) for s in inverse_token])
        exact_map = token_shuffle[physical_targets]
        exact.append(exact_map)
        accuracy = float(np.mean(inferred[opaque_action] == exact_map))
        per_action.append(
            {
                "opaque_action": opaque_action,
                "accepted_probe_records": int(sum(record[1] == opaque_action for record in records)),
                "offline_map_accuracy": accuracy,
                "inferred_permutation_order": _permutation_order(inferred[opaque_action]),
                "unique_inferred_targets": int(len(np.unique(inferred[opaque_action]))),
                "mean_peak_probability": float(np.mean(np.max(laws[opaque_action], axis=1))),
            }
        )
    valid = all(len(np.unique(permutation)) == 9 for permutation in inferred)
    commute = bool(valid and all(
        np.array_equal(_compose_permutations(first, second), _compose_permutations(second, first))
        for first in inferred for second in inferred
    ))
    group = _generated_permutation_group(inferred) if valid else []
    orbit = len({int(permutation[0]) for permutation in group}) if group else 1
    summary: dict[str, float | int | bool | str] = {
        "attempts": int(private["attempts"]),
        "accepted_memory_events": int(private["accepted_memory_events"]),
        "memory_acceptance_rate": float(int(private["accepted_memory_events"]) / int(private["attempts"])),
        "offline_mean_map_accuracy": float(np.mean([row["offline_map_accuracy"] for row in per_action])),
        "valid_permutation_action_set": bool(valid),
        "commuting": commute,
        "learned_group_order": len(group),
        "orbit_size": orbit,
        "learner_fields": "opaque_anchor_token,opaque_action_token,opaque_future_probe_token",
    }
    return summary, per_action


def oracle_labelled_transition_nll(
    model: MemoryInstrument, train_per_state_action: int, strings: int, length: int, seed: int
) -> dict[str, float]:
    """Score joint outcome/latent-next-state strings with oracle state labels."""
    rng = np.random.default_rng(seed)
    counts = np.full((9, 5, 10, 9), 0.25)
    exact = np.array([model.transition_branches(move) for move in MOVES]).transpose(1, 0, 2, 3)
    for state in range(9):
        for action in range(5):
            flat = exact[state, action].reshape(-1)
            draws = rng.multinomial(train_per_state_action, flat)
            counts[state, action] += draws.reshape(10, 9)
    learned = counts / counts.sum(axis=(2, 3), keepdims=True)
    exact_nll = learned_nll = marginal_nll = 0.0
    observations = 0
    marginal = exact.mean(axis=0)
    for _ in range(strings):
        state = int(rng.integers(9))
        for _ in range(length):
            action = int(rng.integers(5))
            flat = exact[state, action].reshape(-1)
            event = int(rng.choice(90, p=flat))
            outcome, next_state = divmod(event, 9)
            exact_nll -= log2(max(flat[event], 1e-300))
            learned_nll -= log2(max(learned[state, action, outcome, next_state], 1e-300))
            marginal_nll -= log2(max(marginal[action, outcome, next_state], 1e-300))
            state = next_state; observations += 1
    return {
        "exact_nll_bits": exact_nll / observations,
        "learned_nll_bits": learned_nll / observations,
        "marginal_nll_bits": marginal_nll / observations,
        "heldout_strings": strings, "heldout_length": length,
    }
