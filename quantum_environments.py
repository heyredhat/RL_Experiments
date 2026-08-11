"""Configurable hidden quantum-measurement environments.

The classes in this module own the simulator's privileged quantum description.
Agents should depend only on ``n_actions``, ``n_outcomes``, ``reset()``, and
``step(action)``.  Kraus operators and density matrices deliberately remain
private implementation details.

The catalog spans several operationally different worlds: sharp and unsharp
qubit measurements, a four-outcome qubit SIC measurement, and mutually
unbiased qutrit measurements.  This makes it possible to ask whether learned
predictive states and goal geometries change with the agent's intervention
repertoire rather than merely with its random seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Measurement:
    """One named quantum instrument represented by outcome Kraus operators."""

    name: str
    kraus: tuple[Array, ...]


@dataclass(frozen=True)
class EnvironmentDefinition:
    """Validated ingredients used to instantiate a hidden environment."""

    name: str
    description: str
    measurements: tuple[Measurement, ...]
    initial_states: Mapping[str, Array]
    default_initial_state: str

    @property
    def dimension(self) -> int:
        return int(self.measurements[0].kraus[0].shape[0])


def _ket_density(ket: Array) -> Array:
    ket = np.asarray(ket, dtype=complex)
    ket = ket / np.linalg.norm(ket)
    return np.outer(ket, ket.conj())


def _projective_measurement(name: str, basis: tuple[Array, ...]) -> Measurement:
    return Measurement(name, tuple(_ket_density(ket) for ket in basis))


def _unsharp_qubit_measurement(name: str, axis: Array, q: float) -> Measurement:
    """Lüders instrument for effects ``q P+ + (1-q) P-`` and its complement."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    pauli_axis = np.array(
        [[z, x - 1j * y], [x + 1j * y, -z]],
        dtype=complex,
    )
    p_plus = (np.eye(2, dtype=complex) + pauli_axis) / 2.0
    p_minus = np.eye(2, dtype=complex) - p_plus
    k0 = np.sqrt(q) * p_plus + np.sqrt(1.0 - q) * p_minus
    k1 = np.sqrt(1.0 - q) * p_plus + np.sqrt(q) * p_minus
    return Measurement(name, (k0, k1))


def _qubit_states() -> dict[str, Array]:
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    return {
        "zero": _ket_density(zero),
        "one": _ket_density(one),
        "plus": _ket_density(zero + one),
        "minus": _ket_density(zero - one),
        "plus-i": _ket_density(zero + 1j * one),
        "minus-i": _ket_density(zero - 1j * one),
        "mixed": np.eye(2, dtype=complex) / 2.0,
    }


def _qubit_bases() -> dict[str, tuple[Array, ...]]:
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    return {
        "Z": (zero, one),
        "X": ((zero + one) / np.sqrt(2.0), (zero - one) / np.sqrt(2.0)),
        "Y": (
            (zero + 1j * one) / np.sqrt(2.0),
            (zero - 1j * one) / np.sqrt(2.0),
        ),
    }


def _qubit_zx_weak(weak_q: float) -> EnvironmentDefinition:
    bases = _qubit_bases()
    return EnvironmentDefinition(
        name="qubit-zx-weak",
        description="Projective Z/X measurements plus an unsharp Z instrument.",
        measurements=(
            _projective_measurement("Z", bases["Z"]),
            _projective_measurement("X", bases["X"]),
            _unsharp_qubit_measurement("weak-Z", (0, 0, 1), weak_q),
        ),
        initial_states=_qubit_states(),
        default_initial_state="one",
    )


def _qubit_pauli(_: float) -> EnvironmentDefinition:
    bases = _qubit_bases()
    return EnvironmentDefinition(
        name="qubit-pauli",
        description="Three incompatible projective Pauli measurements Z, X, and Y.",
        measurements=tuple(
            _projective_measurement(name, bases[name]) for name in ("Z", "X", "Y")
        ),
        initial_states=_qubit_states(),
        default_initial_state="plus-i",
    )


def _qubit_unsharp(weak_q: float) -> EnvironmentDefinition:
    return EnvironmentDefinition(
        name="qubit-unsharp",
        description="Three partially informative and partially disturbing Pauli instruments.",
        measurements=(
            _unsharp_qubit_measurement("weak-Z", (0, 0, 1), weak_q),
            _unsharp_qubit_measurement("weak-X", (1, 0, 0), weak_q),
            _unsharp_qubit_measurement("weak-Y", (0, 1, 0), weak_q),
        ),
        initial_states=_qubit_states(),
        default_initial_state="mixed",
    )


def _sic_measurement() -> Measurement:
    # Tetrahedral Bloch vectors.  E_i = (I + r_i.sigma) / 4 = Pi_i / 2,
    # so the positive square-root Kraus operator is Pi_i / sqrt(2).
    vectors = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)
    eye = np.eye(2, dtype=complex)
    kraus = []
    for x, y, z in vectors:
        sigma = np.array([[z, x - 1j * y], [x + 1j * y, -z]], dtype=complex)
        projector = (eye + sigma) / 2.0
        kraus.append(projector / np.sqrt(2.0))
    return Measurement("tetra-SIC", tuple(kraus))


def _qubit_pauli_sic(_: float) -> EnvironmentDefinition:
    bases = _qubit_bases()
    return EnvironmentDefinition(
        name="qubit-pauli-sic",
        description="Pauli projective instruments together with a four-outcome qubit SIC.",
        measurements=(
            _projective_measurement("Z", bases["Z"]),
            _projective_measurement("X", bases["X"]),
            _projective_measurement("Y", bases["Y"]),
            _sic_measurement(),
        ),
        initial_states=_qubit_states(),
        default_initial_state="mixed",
    )


def _qutrit_mub(_: float) -> EnvironmentDefinition:
    dimension = 3
    computational = tuple(np.eye(dimension, dtype=complex)[:, i] for i in range(dimension))
    omega = np.exp(2j * np.pi / dimension)
    fourier = tuple(
        np.array([omega ** (j * k) for j in range(dimension)], dtype=complex)
        / np.sqrt(dimension)
        for k in range(dimension)
    )
    phase_fourier = tuple(
        np.array([omega ** (j * k + j * j) for j in range(dimension)], dtype=complex)
        / np.sqrt(dimension)
        for k in range(dimension)
    )
    equal = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    states = {
        "zero": _ket_density(computational[0]),
        "one": _ket_density(computational[1]),
        "two": _ket_density(computational[2]),
        "plus": _ket_density(equal),
        "mixed": np.eye(dimension, dtype=complex) / dimension,
    }
    return EnvironmentDefinition(
        name="qutrit-mub",
        description="A qutrit observed through three mutually unbiased projective bases.",
        measurements=(
            _projective_measurement("Z3", computational),
            _projective_measurement("F3", fourier),
            _projective_measurement("phase-F3", phase_fourier),
        ),
        initial_states=states,
        default_initial_state="two",
    )


_BUILDERS: dict[str, Callable[[float], EnvironmentDefinition]] = {
    "qubit-zx-weak": _qubit_zx_weak,
    "qubit-pauli": _qubit_pauli,
    "qubit-unsharp": _qubit_unsharp,
    "qubit-pauli-sic": _qubit_pauli_sic,
    "qutrit-mub": _qutrit_mub,
}


DEFAULT_GOALS_BY_ENVIRONMENT: dict[str, str] = {
    "qubit-zx-weak": (
        "Z0=0:0;Z1=0:1;X0=1:0;X1=1:1;"
        "Z0_X0=0:0,1:0;X0_Z0=1:0,0:0;weakZ0_Z0=2:0,0:0"
    ),
    "qubit-pauli": (
        "Z0=0:0;X0=1:0;Y0=2:0;Z0_X0=0:0,1:0;"
        "X0_Z0=1:0,0:0;Y0_X1=2:0,1:1;Z0_X0_Y0=0:0,1:0,2:0"
    ),
    "qubit-unsharp": (
        "wZ0=0:0;wX0=1:0;wY0=2:0;wZ0_wX0=0:0,1:0;"
        "wX0_wZ0=1:0,0:0;wZ0_wX0_wY0=0:0,1:0,2:0"
    ),
    "qubit-pauli-sic": (
        "Z0=0:0;X0=1:0;Y0=2:0;SIC0=3:0;SIC1=3:1;"
        "SIC2=3:2;SIC3=3:3;Z0_SIC0=0:0,3:0;SIC0_X0=3:0,1:0"
    ),
    "qutrit-mub": (
        "Z0=0:0;Z1=0:1;Z2=0:2;F0=1:0;F1=1:1;"
        "Z0_F0=0:0,1:0;F0_Z0=1:0,0:0;Z0_F0_P0=0:0,1:0,2:0"
    ),
}


def available_environments() -> tuple[str, ...]:
    """Return stable names accepted by the command-line interfaces."""
    return tuple(_BUILDERS)


def environment_definition(name: str, weak_q: float = 0.80) -> EnvironmentDefinition:
    """Build and validate a catalog definition without starting an episode."""
    if name not in _BUILDERS:
        choices = ", ".join(available_environments())
        raise ValueError(f"unknown environment {name!r}; choose from {choices}")
    if not 0.5 < float(weak_q) < 1.0:
        raise ValueError("weak_q must lie strictly between 0.5 and 1.0")
    definition = _BUILDERS[name](float(weak_q))
    _validate_definition(definition)
    return definition


def _validate_definition(definition: EnvironmentDefinition) -> None:
    if not definition.measurements:
        raise ValueError("an environment needs at least one measurement")
    dimension = definition.dimension
    eye = np.eye(dimension, dtype=complex)
    for measurement in definition.measurements:
        if not measurement.kraus:
            raise ValueError(f"measurement {measurement.name!r} has no outcomes")
        completeness = np.zeros_like(eye)
        for operator in measurement.kraus:
            if operator.shape != (dimension, dimension):
                raise ValueError(f"invalid Kraus shape in {measurement.name!r}")
            completeness += operator.conj().T @ operator
        if not np.allclose(completeness, eye, atol=1e-9):
            raise ValueError(f"Kraus operators for {measurement.name!r} are incomplete")
    for state_name, rho in definition.initial_states.items():
        if rho.shape != (dimension, dimension):
            raise ValueError(f"invalid state shape for {state_name!r}")
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise ValueError(f"initial state {state_name!r} is not Hermitian")
        if not np.isclose(np.trace(rho).real, 1.0, atol=1e-10):
            raise ValueError(f"initial state {state_name!r} is not normalized")


class QuantumEnvironment:
    """Hidden finite-dimensional quantum system with a minimal agent API."""

    def __init__(
        self,
        initial_state: str | None = None,
        weak_q: float = 0.80,
        seed: int = 0,
        environment: str = "qubit-zx-weak",
    ):
        definition = environment_definition(environment, weak_q)
        selected_state = initial_state or definition.default_initial_state
        if selected_state not in definition.initial_states:
            choices = ", ".join(definition.initial_states)
            raise ValueError(
                f"initial_state {selected_state!r} is unavailable for {environment}; "
                f"choose from {choices}"
            )

        self.environment_name = definition.name
        self.description = definition.description
        self.initial_state = selected_state
        self.weak_q = float(weak_q)
        self.rng = np.random.default_rng(seed)
        self.action_names = tuple(m.name for m in definition.measurements)
        self.action_outcome_counts = tuple(len(m.kraus) for m in definition.measurements)
        self.n_actions = len(definition.measurements)
        # Neural heads have a common width; outcomes unavailable for a given
        # action are simply never sampled or used as goal checkpoints.
        self.n_outcomes = max(self.action_outcome_counts)
        self.dimension = definition.dimension

        self._initial_states = {
            name: rho.copy() for name, rho in definition.initial_states.items()
        }
        self._kraus = tuple(m.kraus for m in definition.measurements)
        self._rho: Array | None = None
        self.reset()

    @property
    def initial_state_names(self) -> tuple[str, ...]:
        """Names are metadata for experiment configuration, not agent input."""
        return tuple(self._initial_states)

    def reset(self) -> None:
        self._rho = self._initial_states[self.initial_state].copy()

    def step(self, action: int) -> int:
        """Apply a hidden instrument and return only its classical outcome."""
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"invalid action {action}")
        assert self._rho is not None
        operators = self._kraus[int(action)]
        probs = np.array(
            [np.trace(k @ self._rho @ k.conj().T).real for k in operators],
            dtype=float,
        )
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0.0:
            raise RuntimeError("quantum instrument produced zero total probability")
        probs /= total
        outcome = int(self.rng.choice(len(operators), p=probs))
        operator = operators[outcome]
        unnormalized = operator @ self._rho @ operator.conj().T
        self._rho = unnormalized / np.trace(unnormalized).real
        # Keep roundoff from accumulating during long experiments.
        self._rho = (self._rho + self._rho.conj().T) / 2.0
        self._rho /= np.trace(self._rho).real
        return outcome

