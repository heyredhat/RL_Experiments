"""Exactly solvable low-dimensional hodological lattices.

The positive construction uses one qubit and two commuting phase rotations.  Its
quantum state is low dimensional, but the action semigroup is a faithful copy
of Z^2.  A finite set of goals therefore inherits an exact square-lattice word
metric without assigning one orthogonal basis state to each goal.

Only NumPy is required.  Plotting and experiment orchestration live in
``run_exact_experiments.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, log, pi, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np


Goal = tuple[int, int]


@dataclass(frozen=True)
class PhaseLattice:
    """A qubit representation of a two-generator translation group.

    The analytically certified default angles are

        alpha = 2*pi/9 + epsilon*sqrt(2)
        beta  = 2*pi/3 + epsilon*sqrt(3).

    For any nonzero rational epsilon, alpha, beta, and 2*pi are rationally
    independent.  Consequently U**a V**b is projectively the identity only for
    a=b=0, and the representation of Z^2 on the equatorial fiducial is faithful.
    """

    epsilon: float = 0.01

    @property
    def alpha(self) -> float:
        return 2.0 * pi / 9.0 + self.epsilon * sqrt(2.0)

    @property
    def beta(self) -> float:
        return 2.0 * pi / 3.0 + self.epsilon * sqrt(3.0)

    @staticmethod
    def fiducial() -> np.ndarray:
        return np.array([1.0, 1.0], dtype=complex) / sqrt(2.0)

    @staticmethod
    def phase_unitary(theta: float) -> np.ndarray:
        return np.diag(np.array([1.0, np.exp(1j * theta)], dtype=complex))

    @property
    def actions(self) -> Mapping[str, tuple[np.ndarray, ...]]:
        """Unitary Kraus instruments for the four cardinal actions."""
        u = self.phase_unitary(self.alpha)
        v = self.phase_unitary(self.beta)
        return {
            "E": (u,),
            "W": (u.conj().T,),
            "N": (v,),
            "S": (v.conj().T,),
        }

    def phase(self, goal: Goal) -> float:
        i, j = goal
        return i * self.alpha + j * self.beta

    def ket(self, goal: Goal) -> np.ndarray:
        return self.phase_unitary(self.phase(goal)) @ self.fiducial()

    def density(self, goal: Goal) -> np.ndarray:
        psi = self.ket(goal)
        return np.outer(psi, psi.conj())

    def verifier(self, goal: Goal) -> Mapping[str, np.ndarray]:
        """Two-outcome goal-verification instrument {P_g, I-P_g}."""
        p = self.density(goal)
        return {"accept": p, "reject": np.eye(2, dtype=complex) - p}

    def fidelity(self, first: Goal, second: Goal) -> float:
        return float(abs(np.vdot(self.ket(first), self.ket(second))) ** 2)

    def apply_word(self, state: np.ndarray, word: Iterable[str]) -> np.ndarray:
        rho = np.array(state, dtype=complex, copy=True)
        for symbol in word:
            (kraus,) = self.actions[symbol]
            rho = kraus @ rho @ kraus.conj().T
        return rho


def goals_3x3() -> tuple[Goal, ...]:
    return tuple(product(range(3), repeat=2))


def displacement(source: Goal, target: Goal) -> Goal:
    return target[0] - source[0], target[1] - source[1]


def canonical_word(source: Goal, target: Goal) -> tuple[str, ...]:
    """One shortest cardinal word taking source exactly to target."""
    di, dj = displacement(source, target)
    horizontal = ("E",) * max(di, 0) + ("W",) * max(-di, 0)
    vertical = ("N",) * max(dj, 0) + ("S",) * max(-dj, 0)
    return horizontal + vertical


def exact_word_distance(source: Goal, target: Goal) -> int:
    """The exact word metric, proved by faithfulness of the representation."""
    di, dj = displacement(source, target)
    return abs(di) + abs(dj)


def word_exponents(word: Iterable[str]) -> Goal:
    """Return the two signed generator counts of an action sequence."""
    counts = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
    i = j = 0
    for symbol in word:
        di, dj = counts[symbol]
        i += di
        j += dj
    return i, j


def bounded_sequence_goal_accepts(
    word: Sequence[str], source: Goal, target: Goal, horizon: int = 4
) -> bool:
    """Membership in a finite, hence regular, sequence-goal language.

    L(source,target;H) contains every word of length at most H whose signed
    generator counts equal target-source.  H=4 retains a shortest realization
    of every transition in the 3x3 patch.  A finite-language trie or the bounded
    counter DFA described in RESULTS_EXACT.md recognizes it exactly.
    """
    if len(word) > horizon:
        return False
    return word_exponents(word) == displacement(source, target)


def exact_euclidean_macro_cost(source: Goal, target: Goal) -> float:
    """Cost when every translation is an available norm-priced macro-action.

    The direct Kraus operator U**di V**dj costs sqrt(di**2+dj**2).  The
    Euclidean triangle inequality proves that no concatenation is cheaper.
    """
    di, dj = displacement(source, target)
    return float(np.hypot(di, dj))


def macro_kraus(model: PhaseLattice, delta: Goal) -> tuple[np.ndarray, ...]:
    theta = delta[0] * model.alpha + delta[1] * model.beta
    return (model.phase_unitary(theta),)


def euclidean_waiting_instrument(
    model: PhaseLattice, delta: Goal
) -> Mapping[str, np.ndarray]:
    """A unit-cost binary instrument with Euclidean expected translation cost.

    For nonzero integer delta, success probability p=1/||delta||_2.  The
    success branch applies the translation and the failure branch is the
    identity, so retry-until-success has mean intervention count ||delta||_2.
    """
    norm = float(np.hypot(*delta))
    if norm < 1.0:
        raise ValueError("delta must be a nonzero integer displacement")
    probability = 1.0 / norm
    unitary = macro_kraus(model, delta)[0]
    return {
        "success": sqrt(probability) * unitary,
        "failure": sqrt(1.0 - probability) * np.eye(2, dtype=complex),
    }


def euclidean_success_probability(delta: Goal) -> float:
    norm = float(np.hypot(*delta))
    if norm < 1.0:
        raise ValueError("delta must be a nonzero integer displacement")
    return 1.0 / norm


def kraus_gram(kraus: Sequence[np.ndarray]) -> np.ndarray:
    return sum((k.conj().T @ k for k in kraus), np.zeros_like(kraus[0]))


def dephasing_kraus(eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Kraus representation of a channel multiplying coherence by eta."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    identity = np.eye(2, dtype=complex)
    z = np.diag(np.array([1.0, -1.0], dtype=complex))
    return sqrt((1.0 + eta) / 2.0) * identity, sqrt((1.0 - eta) / 2.0) * z


def apply_channel(state: np.ndarray, kraus: Sequence[np.ndarray]) -> np.ndarray:
    return sum((k @ state @ k.conj().T for k in kraus), np.zeros_like(state))


def noisy_goal_fidelity(path_length: int, eta: float) -> float:
    """Closed form after path_length rotations, each followed by dephasing."""
    return 0.5 * (1.0 + eta**path_length)


def repeated_verification_count(false_accept: float, error_tolerance: float) -> int:
    """Independent accept tests needed to suppress a false goal below delta."""
    if not 0.0 < error_tolerance < 1.0:
        raise ValueError("error_tolerance must lie in (0, 1)")
    if false_accept <= 0.0:
        return 1
    if false_accept >= 1.0:
        raise ValueError("indistinguishable goals cannot be verified")
    return max(1, ceil(log(error_tolerance) / log(false_accept)))


def angular_separation(model: PhaseLattice, first: Goal, second: Goal) -> float:
    raw = (model.phase(first) - model.phase(second)) % (2.0 * pi)
    return float(min(raw, 2.0 * pi - raw))


def minimum_goal_separation(model: PhaseLattice) -> float:
    goals = goals_3x3()
    return min(
        angular_separation(model, first, second)
        for index, first in enumerate(goals)
        for second in goals[index + 1 :]
    )


def approximate_shortest_distance(
    model: PhaseLattice,
    source: Goal,
    target: Goal,
    infidelity_tolerance: float,
    search_radius: int = 18,
) -> int:
    """Shortest word accepted by a finite-tolerance target verifier.

    The enumeration is finite and intended as a robustness diagnostic, not as
    part of the exact theorem.  The exact canonical displacement is always in
    the search region for the 3x3 patch.
    """
    threshold = 1.0 - infidelity_tolerance
    source_phase = model.phase(source)
    target_phase = model.phase(target)
    candidates = sorted(
        product(range(-search_radius, search_radius + 1), repeat=2),
        key=lambda pair: (abs(pair[0]) + abs(pair[1]), pair),
    )
    for a, b in candidates:
        phase_error = source_phase + a * model.alpha + b * model.beta - target_phase
        fidelity = np.cos(phase_error / 2.0) ** 2
        if fidelity + 1e-14 >= threshold:
            return abs(a) + abs(b)
    raise RuntimeError("increase search_radius")


def qutrit_weyl_operators() -> tuple[np.ndarray, np.ndarray]:
    """The finite-order qutrit Weyl pair X,Z used by the torus counterexample."""
    omega = np.exp(2j * pi / 3.0)
    x = np.roll(np.eye(3, dtype=complex), shift=1, axis=0)
    z = np.diag(np.array([1.0, omega, omega**2], dtype=complex))
    return x, z


def qutrit_torus_distance(source: Goal, target: Goal) -> int:
    """Cayley distance on Z_3 x Z_3 under X^+-, Z^+-."""
    di = abs(target[0] - source[0]) % 3
    dj = abs(target[1] - source[1]) % 3
    return min(di, 3 - di) + min(dj, 3 - dj)


def distance_matrix(distance_function) -> np.ndarray:
    goals = goals_3x3()
    return np.array(
        [[distance_function(first, second) for second in goals] for first in goals],
        dtype=float,
    )


def classical_mds(distance: np.ndarray, dimensions: int = 2) -> tuple[np.ndarray, float]:
    """Classical MDS coordinates and normalized raw stress."""
    count = distance.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:dimensions], 0.0)
    coordinates = vectors[:, :dimensions] * np.sqrt(positive)
    embedded = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(distance, dtype=bool), k=1)
    stress = float(np.sqrt(np.sum((embedded[mask] - distance[mask]) ** 2) / np.sum(distance[mask] ** 2)))
    return coordinates, stress
