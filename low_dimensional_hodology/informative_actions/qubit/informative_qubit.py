"""Informative qubit instruments and coordinate-free predictive reconstruction.

The public learner-facing objects use opaque button names and observed
frequencies. Hidden lattice coordinates appear only in evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import atan2, cos, pi, sin, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np


BUTTONS = ("amber", "blue", "crimson", "dune")
COORDINATES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def wrap(angle: float) -> float:
    return float((angle + pi) % (2.0 * pi) - pi)


def circular_error(first: float, second: float) -> float:
    return abs(wrap(first - second))


def phase_unitary(angle: float) -> np.ndarray:
    return np.diag(np.array([np.exp(-0.5j * angle), np.exp(0.5j * angle)]))


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T


@dataclass(frozen=True)
class InformativeQubit:
    """Four opaque nonunitary buttons acting on an equatorial qubit.

    Each button first applies its phase rotation and then performs the same
    weak X measurement. The immediate outcome is informative, while common
    terminal X/Y probes make signed phase effects operationally identifiable.
    """

    alpha: float
    beta: float
    strength: float

    @property
    def phases(self) -> Mapping[str, float]:
        return dict(zip(BUTTONS, (self.alpha, -self.alpha, self.beta, -self.beta)))

    @property
    def hidden_coordinates(self) -> Mapping[str, tuple[int, int]]:
        return dict(zip(BUTTONS, COORDINATES))

    @property
    def coherence_retention(self) -> float:
        return sqrt(1.0 - self.strength**2)

    @staticmethod
    def reset_state() -> np.ndarray:
        plus = np.array([1.0, 1.0], dtype=complex) / sqrt(2.0)
        return np.outer(plus, plus.conj())

    @staticmethod
    def pauli() -> Mapping[str, np.ndarray]:
        return {
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

    @property
    def weak_effects(self) -> Mapping[int, np.ndarray]:
        identity = np.eye(2, dtype=complex)
        x = self.pauli()["X"]
        return {
            1: 0.5 * (identity + self.strength * x),
            -1: 0.5 * (identity - self.strength * x),
        }

    def instrument(self, button: str) -> Mapping[int, np.ndarray]:
        unitary = phase_unitary(self.phases[button])
        return {outcome: matrix_sqrt_psd(effect) @ unitary for outcome, effect in self.weak_effects.items()}

    def apply(self, state: np.ndarray, button: str, outcome: int) -> tuple[float, np.ndarray]:
        kraus = self.instrument(button)[outcome]
        unnormalized = kraus @ state @ kraus.conj().T
        probability = float(np.real(np.trace(unnormalized)))
        return probability, unnormalized / probability

    def channel(self, state: np.ndarray, button: str) -> np.ndarray:
        return sum((k @ state @ k.conj().T for k in self.instrument(button).values()), np.zeros((2, 2), complex))

    def probe_probability(self, state: np.ndarray, axis: str, outcome: int = 1) -> float:
        pauli = self.pauli()[axis]
        return float(0.5 * (1.0 + outcome * np.real(np.trace(state @ pauli))))

    def exact_signature(self, button: str) -> dict[str, float]:
        state = self.reset_state()
        immediate = self.apply(state, button, 1)[0]
        post = self.channel(state, button)
        return {"action_plus": immediate, "probe_X": self.probe_probability(post, "X"), "probe_Y": self.probe_probability(post, "Y")}


def ideal_goal_phases(alpha: float, beta: float) -> dict[tuple[int, int], float]:
    return {(i, j): wrap(i * alpha + j * beta) for i, j in product(range(3), repeat=2)}


def minimum_pairwise_phase(values: Iterable[float]) -> float:
    values = tuple(values)
    return min(circular_error(a, b) for index, a in enumerate(values) for b in values[index + 1 :])


def bounded_alias_margin(alpha: float, beta: float, horizon: int = 8) -> float:
    candidates = []
    for i in range(-horizon, horizon + 1):
        for j in range(-horizon, horizon + 1):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= horizon:
                candidates.append(circular_error(i * alpha + j * beta, 0.0))
    return min(candidates)


def candidate_score(alpha: float, beta: float, strength: float) -> dict[str, float]:
    goal_separation = minimum_pairwise_phase(ideal_goal_phases(alpha, beta).values())
    action_separation = minimum_pairwise_phase((alpha, -alpha, beta, -beta))
    alias_margin = bounded_alias_margin(alpha, beta)
    outcome_axis_gap = strength * abs(cos(alpha) - cos(beta))
    signed_probe_gap = sqrt(1.0 - strength**2) * min(abs(sin(alpha)), abs(sin(beta)))
    information_margin = min(outcome_axis_gap, signed_probe_gap)
    score = (
        1.7 * goal_separation
        + 0.7 * action_separation
        + 1.2 * alias_margin
        + 1.5 * information_margin
        - 0.18 * (1.0 - sqrt(1.0 - strength**2)) * 4.0
        - 12.0 * max(0.0, 0.08 - alias_margin)
    )
    return {
        "score": score,
        "goal_separation": goal_separation,
        "action_separation": action_separation,
        "alias_margin_h8": alias_margin,
        "outcome_axis_gap": outcome_axis_gap,
        "signed_probe_gap": signed_probe_gap,
        "information_margin": information_margin,
    }


def deterministic_search() -> list[dict[str, float]]:
    """Reproducible finite search balancing semantics and disturbance."""
    records = []
    for alpha in np.linspace(0.52, 1.18, 23):
        for beta in np.linspace(1.48, 2.65, 27):
            if beta <= alpha:
                continue
            for strength in np.linspace(0.18, 0.72, 19):
                metrics = candidate_score(float(alpha), float(beta), float(strength))
                records.append({"alpha": float(alpha), "beta": float(beta), "strength": float(strength), **metrics})
    return sorted(records, key=lambda row: (-row["score"], row["alpha"], row["beta"], row["strength"]))


def sample_signatures(model: InformativeQubit, trials: int, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Estimate each opaque button only through observed common tests."""
    estimates = {}
    for button in BUTTONS:
        exact = model.exact_signature(button)
        estimates[button] = {
            key: float(rng.binomial(trials, probability) / trials)
            for key, probability in exact.items()
        }
    return estimates


def reconstruct_phases(signatures: Mapping[str, Mapping[str, float]], retention: float) -> dict[str, float]:
    phases = {}
    for button, signature in signatures.items():
        x = 2.0 * signature["probe_X"] - 1.0
        y = (2.0 * signature["probe_Y"] - 1.0) / retention
        phases[button] = wrap(atan2(y, x))
    return phases


def infer_coordinate_chart(phases: Mapping[str, float]) -> dict[str, tuple[int, int]]:
    """Infer inverse pairs and axes, fixing only an arbitrary gauge.

    All 24 assignments of opaque buttons to cardinal coordinates are scored by
    inverse-phase and within-axis consistency. A deterministic lexicographic
    tie-break chooses one representative of the D4 gauge orbit.
    """
    best = None
    for ordered in permutations(BUTTONS):
        assignment = dict(zip(ordered, COORDINATES))
        phase_by_coordinate = {assignment[button]: phases[button] for button in BUTTONS}
        score = (
            circular_error(phase_by_coordinate[(1, 0)] + phase_by_coordinate[(-1, 0)], 0.0) ** 2
            + circular_error(phase_by_coordinate[(0, 1)] + phase_by_coordinate[(0, -1)], 0.0) ** 2
        )
        key = (score, ordered)
        if best is None or key < best[0]:
            best = (key, assignment)
    assert best is not None
    return best[1]


def charts_equivalent_under_d4(
    inferred: Mapping[str, tuple[int, int]], truth: Mapping[str, tuple[int, int]]
) -> bool:
    transforms = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                def transform(point, swap=swap, sx=sx, sy=sy):
                    x, y = point
                    if swap:
                        x, y = y, x
                    return sx * x, sy * y
                transforms.append(transform)
    return any(all(inferred[b] == transform(truth[b]) for b in BUTTONS) for transform in transforms)


def canonical_goal_word(chart: Mapping[str, tuple[int, int]], goal: tuple[int, int]) -> tuple[str, ...]:
    inverse = {coordinate: button for button, coordinate in chart.items()}
    return (
        (inverse[(1, 0)],) * goal[0]
        + (inverse[(0, 1)],) * goal[1]
    )


def sequence_displacement(word: Sequence[str], chart: Mapping[str, tuple[int, int]]) -> tuple[int, int]:
    return tuple(sum(chart[button][axis] for button in word) for axis in (0, 1))  # type: ignore[return-value]


def manhattan_goal_matrix() -> np.ndarray:
    goals = tuple(product(range(3), repeat=2))
    return np.array([[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in goals] for a in goals], float)


def conditional_trajectory(model: InformativeQubit, buttons: Sequence[str], outcomes: Sequence[int]) -> np.ndarray:
    state = model.reset_state()
    for button, outcome in zip(buttons, outcomes):
        _, state = model.apply(state, button, outcome)
    return state


def reconstruct_state_from_probes(probabilities: Mapping[str, float]) -> np.ndarray:
    bloch = np.array([2.0 * probabilities[axis] - 1.0 for axis in ("X", "Y", "Z")])
    norm = np.linalg.norm(bloch)
    if norm > 1.0:
        bloch /= norm
    pauli = InformativeQubit.pauli()
    return 0.5 * (np.eye(2) + sum(bloch[index] * pauli[axis] for index, axis in enumerate(("X", "Y", "Z"))))


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    singular = np.linalg.eigvalsh(first - second)
    return float(0.5 * np.sum(np.abs(singular)))


def nonselective_word_state(model: InformativeQubit, word: Sequence[str]) -> np.ndarray:
    state = model.reset_state()
    for button in word:
        state = model.channel(state, button)
    return state


def common_future_signature(model: InformativeQubit, state: np.ndarray) -> np.ndarray:
    return np.array([model.probe_probability(state, axis) for axis in ("X", "Y", "Z")])


def word_equivalence_audit(model: InformativeQubit) -> list[dict[str, float | int | str]]:
    """Compare distinct paths assigned the same external word displacement."""
    chart = model.hidden_coordinates
    words = [tuple(word) for length in range(5) for word in product(BUTTONS, repeat=length)]
    groups: dict[tuple[int, int], list[tuple[str, ...]]] = {}
    for word in words:
        groups.setdefault(sequence_displacement(word, chart), []).append(word)
    rows = []
    for displacement, members in sorted(groups.items()):
        signatures = [common_future_signature(model, nonselective_word_state(model, word)) for word in members]
        maximum = 0.0
        witness = (members[0], members[0])
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                difference = float(np.linalg.norm(signatures[first] - signatures[second]))
                if difference > maximum:
                    maximum = difference
                    witness = members[first], members[second]
        rows.append(
            {
                "dx": displacement[0], "dy": displacement[1], "path_count": len(members),
                "max_common_probe_difference": maximum,
                "equivalent_at_1e-10": int(maximum < 1e-10),
                "witness_word_1": "-".join(witness[0]) or "identity",
                "witness_word_2": "-".join(witness[1]) or "identity",
            }
        )
    return rows


def random_unitary_signatures(model: InformativeQubit) -> dict[str, dict[str, float]]:
    result = {}
    for button, phase in model.phases.items():
        state = phase_unitary(phase) @ model.reset_state() @ phase_unitary(phase).conj().T
        result[button] = {
            "action_plus": 0.5,
            "probe_X": model.probe_probability(state, "X"),
            "probe_Y": model.probe_probability(state, "Y"),
        }
    return result


def null_signatures() -> dict[str, dict[str, float]]:
    return {button: {"action_plus": 0.5, "probe_X": 1.0, "probe_Y": 0.5} for button in BUTTONS}
