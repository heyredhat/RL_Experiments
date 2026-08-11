"""Exact Euclidean hodology on a two-phase qutrit orbit.

The physical states are nonorthogonal qutrit phase states, while observed
success/failure outcomes update a finite history coordinate in lockstep with
the hidden state.  Random-unitary retry instruments turn Euclidean displacement
length into expected unit-intervention cost.  The construction is intended as
an analytic benchmark, not as a claim that the inverse-distance probability
law has itself emerged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class QutritPhaseLattice:
    """A faithful square phase torus with a local open 3x3 chart."""

    order: int = 11

    def __post_init__(self) -> None:
        if self.order < 5 or self.order % 2 == 0:
            raise ValueError("order must be an odd integer at least five")

    @staticmethod
    def fiducial() -> np.ndarray:
        return np.array([np.sqrt(3 / 8), 0.5, np.sqrt(3 / 8)], dtype=complex)

    @staticmethod
    def generators() -> tuple[np.ndarray, np.ndarray]:
        return np.diag([0.0, 1.0, 0.0]), np.diag([0.0, 0.5, 1.0])

    @property
    def epsilon(self) -> float:
        return 4.0 * np.pi / self.order

    @property
    def unitaries(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.generators()
        return np.diag(np.exp(1j * self.epsilon * np.diag(a))), np.diag(
            np.exp(1j * self.epsilon * np.diag(b))
        )

    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(product(range(self.order), repeat=2))

    @staticmethod
    def patch_coordinates() -> tuple[Coordinate, ...]:
        return tuple(product(range(3), repeat=2))

    def ket(self, coordinate: Coordinate) -> np.ndarray:
        x, y = coordinate
        u, v = self.unitaries
        return (
            np.linalg.matrix_power(u, x % self.order)
            @ np.linalg.matrix_power(v, y % self.order)
            @ self.fiducial()
        )

    def density(self, coordinate: Coordinate) -> np.ndarray:
        ket = self.ket(coordinate)
        return np.outer(ket, ket.conj())

    def centered(self, value: int) -> int:
        residue = value % self.order
        return residue if residue <= self.order // 2 else residue - self.order

    def displacement(self, source: Coordinate, target: Coordinate) -> Coordinate:
        return (
            self.centered(target[0] - source[0]),
            self.centered(target[1] - source[1]),
        )

    def distance(self, source: Coordinate, target: Coordinate) -> float:
        return float(np.hypot(*self.displacement(source, target)))

    def displacement_unitary(self, displacement: Coordinate) -> np.ndarray:
        dx, dy = displacement
        u, v = self.unitaries
        return np.linalg.matrix_power(u, dx % self.order) @ np.linalg.matrix_power(
            v, dy % self.order
        )

    def retry_kraus(self, displacement: Coordinate) -> tuple[np.ndarray, np.ndarray]:
        """Success moves by delta; failure is identity; every attempt costs one."""
        length = float(np.hypot(*displacement))
        if length < 1.0:
            raise ValueError("a retry displacement must have Euclidean length at least one")
        probability = 1.0 / length
        return (
            np.sqrt(probability) * self.displacement_unitary(displacement),
            np.sqrt(1.0 - probability) * np.eye(3, dtype=complex),
        )

    def fubini_study_metric(self) -> np.ndarray:
        """Covariance metric of the two phase generators at the fiducial."""
        probabilities = np.abs(self.fiducial()) ** 2
        a, b = (np.diag(operator).real for operator in self.generators())
        centered_a = a - probabilities @ a
        centered_b = b - probabilities @ b
        return np.array(
            [
                [probabilities @ centered_a**2, probabilities @ (centered_a * centered_b)],
                [probabilities @ (centered_a * centered_b), probabilities @ centered_b**2],
            ]
        )


def distance_matrix(
    model: QutritPhaseLattice, coordinates: tuple[Coordinate, ...]
) -> np.ndarray:
    return np.array(
        [[model.distance(source, target) for target in coordinates] for source in coordinates],
        dtype=float,
    )


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(first - second)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def trace_distance_matrix(
    model: QutritPhaseLattice, coordinates: tuple[Coordinate, ...]
) -> np.ndarray:
    states = [model.density(coordinate) for coordinate in coordinates]
    return np.array(
        [[trace_distance(first, second) for second in states] for first in states]
    )


def schoenberg_gram(distance: np.ndarray) -> np.ndarray:
    count = len(distance)
    centering = np.eye(count) - np.ones((count, count)) / count
    return -0.5 * centering @ distance**2 @ centering


def bellman_residual(model: QutritPhaseLattice, goal: Coordinate) -> float:
    """Maximum residual in the analytic all-displacement Bellman equations."""
    coordinates = model.coordinates()
    half = model.order // 2
    actions = tuple(
        displacement
        for displacement in product(range(-half, half + 1), repeat=2)
        if displacement != (0, 0)
    )
    residual = 0.0
    for source in coordinates:
        if source == goal:
            continue
        value = model.distance(source, goal)
        candidates = []
        for displacement in actions:
            length = float(np.hypot(*displacement))
            probability = 1.0 / length
            moved = (
                (source[0] + displacement[0]) % model.order,
                (source[1] + displacement[1]) % model.order,
            )
            candidates.append(
                1.0
                + (1.0 - probability) * value
                + probability * model.distance(moved, goal)
            )
        residual = max(residual, abs(min(candidates) - value))
    return residual

