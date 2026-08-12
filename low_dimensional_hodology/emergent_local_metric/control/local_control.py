"""Local translation-covariant qutrit control and metric diagnostics.

No action depends on the current location or requested goal.  Each repertoire
is one finite stencil reused throughout the plane.  A binary random-unitary
instrument implements every stencil displacement; its geometric waiting cost
is the edge cost used by the Bellman solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from math import acos, gcd, hypot, pi, sqrt
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


Vector = tuple[int, int]


@dataclass(frozen=True)
class LocalQutritModel:
    """Equilateral phase encoding with locally isotropic FS metric."""

    phase_scale: float = 0.15

    @property
    def wave_vectors(self) -> np.ndarray:
        return np.array(
            [[1.0, 0.0], [-0.5, sqrt(3.0) / 2.0], [-0.5, -sqrt(3.0) / 2.0]],
            dtype=float,
        )

    def ket(self, point: Sequence[float]) -> np.ndarray:
        phases = self.phase_scale * (self.wave_vectors @ np.asarray(point, dtype=float))
        return np.exp(1j * phases) / sqrt(3.0)

    def density(self, point: Sequence[float]) -> np.ndarray:
        ket = self.ket(point)
        return np.outer(ket, ket.conj())

    def translation(self, displacement: Sequence[float]) -> np.ndarray:
        phases = self.phase_scale * (self.wave_vectors @ np.asarray(displacement, dtype=float))
        return np.diag(np.exp(1j * phases))

    def instrument(self, displacement: Vector, expected_cost: float) -> Mapping[str, np.ndarray]:
        """Unit-cost attempt with mean waiting time ``expected_cost``."""
        if expected_cost < 1.0:
            raise ValueError("a unit-cost attempt requires expected_cost >= 1")
        probability = 1.0 / expected_cost
        return {
            "success": sqrt(probability) * self.translation(displacement),
            "failure": sqrt(1.0 - probability) * np.eye(3, dtype=complex),
        }

    def fubini_study(self, first: Sequence[float], second: Sequence[float]) -> float:
        overlap = abs(np.vdot(self.ket(first), self.ket(second)))
        return acos(float(np.clip(overlap, 0.0, 1.0)))

    def scaled_fubini_study_displacement(self, displacement: Sequence[float]) -> float:
        """FS distance rescaled to agree with Euclidean distance infinitesimally."""
        return sqrt(2.0) * self.fubini_study((0.0, 0.0), displacement) / self.phase_scale


def primitive_stencil(max_component: int) -> tuple[Vector, ...]:
    """All primitive integer directions in a square stencil."""
    vectors = []
    for x, y in product(range(-max_component, max_component + 1), repeat=2):
        if (x, y) == (0, 0):
            continue
        if gcd(abs(x), abs(y)) == 1:
            vectors.append((x, y))
    return tuple(sorted(vectors))


def repertoire(name: str) -> tuple[Vector, ...]:
    levels = {"D4": 0, "D8": 1, "D16": 2, "D32": 3}
    if name not in levels:
        raise KeyError(name)
    if name == "D4":
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    full = primitive_stencil(levels[name])
    if name == "D8":
        return full
    if name == "D16":
        return tuple(v for v in full if max(abs(v[0]), abs(v[1])) <= 2)
    return full


def direction_class(vector: Vector) -> Vector:
    values = sorted((abs(vector[0]), abs(vector[1])), reverse=True)
    return values[0], values[1]


def classes_for(actions: Sequence[Vector]) -> tuple[Vector, ...]:
    return tuple(sorted({direction_class(action) for action in actions}))


def costs_from_parameters(actions: Sequence[Vector], parameters: Mapping[Vector, float]) -> dict[Vector, float]:
    return {action: float(parameters[direction_class(action)]) for action in actions}


def euclidean(displacement: Sequence[float]) -> float:
    return float(hypot(displacement[0], displacement[1]))


def grid_displacements(radius: int) -> tuple[Vector, ...]:
    return tuple(
        (x, y)
        for x, y in product(range(-radius, radius + 1), repeat=2)
        if (x, y) != (0, 0) and hypot(x, y) <= radius + 1e-12
    )


def bellman_distances(
    actions: Sequence[Vector], costs: Mapping[Vector, float], radius: int
) -> dict[Vector, float]:
    """Exact deterministic shortest expected costs on a bounded lattice."""
    margin = max(max(abs(x), abs(y)) for x, y in actions) + 1
    bound = radius + margin
    distances: dict[Vector, float] = {(0, 0): 0.0}
    queue: list[tuple[float, Vector]] = [(0.0, (0, 0))]
    while queue:
        distance, point = heappop(queue)
        if distance != distances.get(point):
            continue
        for action in actions:
            neighbor = point[0] + action[0], point[1] + action[1]
            if abs(neighbor[0]) > bound or abs(neighbor[1]) > bound:
                continue
            candidate = distance + costs[action]
            if candidate + 1e-14 < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heappush(queue, (candidate, neighbor))
    return distances


def relative_mse(
    parameters: Mapping[Vector, float],
    actions: Sequence[Vector],
    training: Sequence[Vector],
    target: Callable[[Vector], float],
) -> float:
    costs = costs_from_parameters(actions, parameters)
    radius = int(np.ceil(max(euclidean(point) for point in training)))
    distances = bellman_distances(actions, costs, radius)
    errors = [(distances[point] / target(point) - 1.0) ** 2 for point in training]
    return float(np.mean(errors))


def optimize_class_costs(
    actions: Sequence[Vector],
    training: Sequence[Vector],
    target: Callable[[Vector], float],
    iterations: int = 42,
) -> tuple[dict[Vector, float], list[float]]:
    """Deterministic projected coordinate search over symmetry-tied costs."""
    classes = classes_for(actions)
    parameters = {key: max(1.0, 1.18 * euclidean(key)) for key in classes}
    history = [relative_mse(parameters, actions, training, target)]
    steps = {key: 0.24 * euclidean(key) for key in classes}
    for _ in range(iterations):
        for key in classes:
            current = parameters[key]
            best_value = current
            best_objective = history[-1]
            for sign in (-1.0, 1.0):
                candidate = max(1.0, current + sign * steps[key])
                proposal = dict(parameters)
                proposal[key] = candidate
                objective = relative_mse(proposal, actions, training, target)
                if objective + 1e-14 < best_objective:
                    best_objective = objective
                    best_value = candidate
            parameters[key] = best_value
            if best_value == current:
                steps[key] *= 0.55
            history.append(best_objective)
        if max(steps.values()) < 2e-5:
            break
    return parameters, history


def bellman_residual(
    distances: Mapping[Vector, float], actions: Sequence[Vector], costs: Mapping[Vector, float], radius: int
) -> float:
    residuals = []
    for point in grid_displacements(radius):
        candidates = []
        for action in actions:
            predecessor = point[0] - action[0], point[1] - action[1]
            if predecessor in distances:
                candidates.append(costs[action] + distances[predecessor])
        residuals.append(abs(distances[point] - min(candidates)))
    return float(max(residuals, default=0.0))


def anisotropy_by_radius(
    distances: Mapping[Vector, float], radii: Iterable[int]
) -> list[dict[str, float]]:
    rows = []
    for radius in radii:
        points = [p for p in grid_displacements(radius) if radius - 1 < euclidean(p) <= radius]
        ratios = np.array([distances[p] / euclidean(p) for p in points])
        rows.append(
            {
                "radius": radius,
                "count": len(points),
                "mean_ratio": float(np.mean(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "anisotropy_range": float((np.max(ratios) - np.min(ratios)) / np.mean(ratios)),
                "angular_cv": float(np.std(ratios) / np.mean(ratios)),
            }
        )
    return rows


def pairwise_metric(points: Sequence[Vector], distances: Mapping[Vector, float]) -> np.ndarray:
    return np.array(
        [[distances[(b[0] - a[0], b[1] - a[1])] for b in points] for a in points],
        dtype=float,
    )


def schoenberg_diagnostics(distance: np.ndarray, dimensions: int = 2) -> dict[str, float]:
    count = len(distance)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    negative_mass = float(np.sum(np.abs(eigenvalues[eigenvalues < -1e-10])))
    absolute_mass = float(np.sum(np.abs(eigenvalues)))
    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coordinates = eigenvectors[:, :dimensions] * np.sqrt(positive)
    embedded = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(distance, dtype=bool), 1)
    stress = float(np.sqrt(np.sum((embedded[mask] - distance[mask]) ** 2) / np.sum(distance[mask] ** 2)))
    return {
        "negative_eigenmass_fraction": negative_mass / absolute_mass if absolute_mass else 0.0,
        "positive_dimension": int(np.sum(eigenvalues > 1e-10)),
        "mds_2d_stress": stress,
        "minimum_gram_eigenvalue": float(np.min(eigenvalues)),
    }


def translation_covariance_residual(model: LocalQutritModel, points: Sequence[Vector], actions: Sequence[Vector]) -> float:
    residual = 0.0
    for point in points:
        rho = model.density(point)
        for action in actions:
            unitary = model.translation(action)
            translated = unitary @ rho @ unitary.conj().T
            expected = model.density((point[0] + action[0], point[1] + action[1]))
            residual = max(residual, float(np.max(np.abs(translated - expected))))
    return residual


def kraus_completeness_residual(model: LocalQutritModel, actions: Sequence[Vector], costs: Mapping[Vector, float]) -> float:
    residual = 0.0
    for action in actions:
        instrument = model.instrument(action, costs[action])
        gram = sum((k.conj().T @ k for k in instrument.values()), np.zeros((3, 3), dtype=complex))
        residual = max(residual, float(np.max(np.abs(gram - np.eye(3)))))
    return residual
