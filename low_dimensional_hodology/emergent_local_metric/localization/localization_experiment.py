#!/usr/bin/env python3
"""Bayesian localization and navigation on a nonorthogonal qutrit phase orbit.

The hidden preparation label is one of nine phase-grid coordinates.  A weak,
covariant Lüders instrument produces observations while disturbing the qutrit.
An exact nine-hypothesis quantum filter tracks both posterior label weights and
the conditional density matrix under every preparation hypothesis.

Two terminal criteria are deliberately kept separate:

* label navigation: did the inferred translation move the hidden preparation
  label to the target coordinate?
* operational state navigation: fidelity of the final disturbed state with the
  target phase-grid state.

The gap between them diagnoses measurement-induced preparation masquerading as
localization.
"""

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
    kets = np.array(
        [[1.0, omega**x, omega**y] for x, y in COORDS], dtype=complex
    ) / np.sqrt(3.0)
    move_x = np.diag([1.0, omega, 1.0]).astype(complex)
    move_y = np.diag([1.0, 1.0, omega]).astype(complex)
    return kets, move_x, move_y


def density(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def positive_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T


@lru_cache(maxsize=None)
def weak_instrument(eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return covariant effects and their positive-square-root Kraus operators.

    ``eta=0`` is a null nine-outcome coin and ``eta=1`` is the rank-one phase
    POVM.  Intermediate values continuously interpolate the effects:

        E_o = eta Pi_o/3 + (1-eta) I/9.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    kets, _, _ = phase_grid()
    effects = np.array(
        [eta * density(ket) / 3.0 + (1.0 - eta) * np.eye(3) / 9.0 for ket in kets]
    )
    kraus = np.array([positive_sqrt(effect) for effect in effects])
    return effects, kraus


def translate_index(index: int, dx: int, dy: int) -> int:
    x, y = COORDS[index]
    return ((x + dx) % 3) * 3 + ((y + dy) % 3)


def signed_step(delta: int) -> int:
    delta %= 3
    return 0 if delta == 0 else (1 if delta == 1 else -1)


def shortest_displacement(source: int, target: int) -> tuple[int, int]:
    sx, sy = COORDS[source]
    tx, ty = COORDS[target]
    return signed_step(tx - sx), signed_step(ty - sy)


def torus_distance(source: int, target: int) -> int:
    dx, dy = shortest_displacement(source, target)
    return abs(dx) + abs(dy)


def move_unitary(dx: int, dy: int) -> np.ndarray:
    _, move_x, move_y = phase_grid()
    return np.linalg.matrix_power(move_x, dx % 3) @ np.linalg.matrix_power(
        move_y, dy % 3
    )


def born_probability(rho: np.ndarray, effect: np.ndarray) -> float:
    return float(np.real(np.trace(effect @ rho)))


def normalize_density(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.real(np.trace(rho))


def entropy_bits(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log2(positive)))


def stable_argmax(values: np.ndarray, tolerance: float = 1e-12) -> int:
    """Choose the lowest label among numerically tied maximizers."""
    values = np.asarray(values, dtype=float)
    return int(np.flatnonzero(values >= np.max(values) - tolerance)[0])


@dataclass
class QuantumLabelFilter:
    """Exact filter over nine initial preparation hypotheses."""

    weights: np.ndarray
    branch_states: np.ndarray

    @classmethod
    def uniform(cls) -> "QuantumLabelFilter":
        kets, _, _ = phase_grid()
        return cls(
            weights=np.full(9, 1.0 / 9.0),
            branch_states=np.array([density(ket) for ket in kets]),
        )

    def predictive_outcomes(self, effects: np.ndarray) -> np.ndarray:
        result = np.zeros(9)
        for hypothesis in range(9):
            for outcome in range(9):
                result[outcome] += self.weights[hypothesis] * born_probability(
                    self.branch_states[hypothesis], effects[outcome]
                )
        return result / result.sum()

    def observe(self, outcome: int, effects: np.ndarray, kraus: np.ndarray) -> None:
        likelihoods = np.array(
            [
                born_probability(self.branch_states[hypothesis], effects[outcome])
                for hypothesis in range(9)
            ]
        )
        evidence = float(self.weights @ likelihoods)
        if evidence <= 0.0:
            raise RuntimeError("impossible observation")
        self.weights = self.weights * likelihoods / evidence
        for hypothesis in range(9):
            state = kraus[outcome] @ self.branch_states[hypothesis] @ kraus[outcome]
            self.branch_states[hypothesis] = normalize_density(state)

    def ensemble_state(self) -> np.ndarray:
        return np.einsum("h,hij->ij", self.weights, self.branch_states)


@dataclass(frozen=True)
class Strategy:
    name: str
    fixed_senses: int | None = None
    threshold: float | None = None
    max_senses: int = 6

    def should_stop(self, senses: int, confidence: float) -> bool:
        if self.fixed_senses is not None:
            return senses >= self.fixed_senses
        assert self.threshold is not None
        return confidence >= self.threshold or senses >= self.max_senses


STRATEGIES = (
    Strategy("no-sense", fixed_senses=0),
    Strategy("fixed-1", fixed_senses=1),
    Strategy("fixed-3", fixed_senses=3),
    Strategy("fixed-5", fixed_senses=5),
    Strategy("adaptive-0.25", threshold=0.25, max_senses=6),
    Strategy("adaptive-0.32", threshold=0.32, max_senses=6),
    Strategy("adaptive-0.50", threshold=0.50, max_senses=6),
)


def sample_outcome(
    rho: np.ndarray, effects: np.ndarray, rng: np.random.Generator
) -> int:
    probabilities = np.array([born_probability(rho, effect) for effect in effects])
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()
    return int(rng.choice(9, p=probabilities))


def run_episode(
    eta: float,
    strategy: Strategy,
    rng: np.random.Generator,
    *,
    known_start: bool = False,
    target: int = 0,
) -> dict[str, object]:
    effects, kraus = weak_instrument(eta)
    kets, _, _ = phase_grid()
    hidden = int(rng.integers(9))
    actual_state = density(kets[hidden])
    belief = QuantumLabelFilter.uniform()
    if known_start:
        belief.weights[:] = 0.0
        belief.weights[hidden] = 1.0

    observations: list[int] = []
    entropies = [entropy_bits(belief.weights)]
    while not strategy.should_stop(len(observations), float(np.max(belief.weights))):
        outcome = sample_outcome(actual_state, effects, rng)
        probability = born_probability(actual_state, effects[outcome])
        actual_state = normalize_density(kraus[outcome] @ actual_state @ kraus[outcome])
        belief.observe(outcome, effects, kraus)
        observations.append(outcome)
        entropies.append(entropy_bits(belief.weights))
        if probability <= 0.0:
            raise RuntimeError("sampled impossible outcome")

    target_projector = density(kets[target])
    estimate = stable_argmax(belief.weights)
    dx, dy = shortest_displacement(estimate, target)
    label_unitary = move_unitary(dx, dy)
    label_final_state = label_unitary @ actual_state @ label_unitary.conj().T
    label_hidden_final = translate_index(hidden, dx, dy)
    operational_score = born_probability(label_final_state, target_projector)
    label_success = int(label_hidden_final == target)
    movement_cost = abs(dx) + abs(dy)

    # Counterfactual controller on the same history: navigate the present
    # conditioned quantum state rather than the inferred preparation label.
    ensemble = belief.ensemble_state()
    state_scores = np.array(
        [born_probability(ensemble, density(candidate)) for candidate in kets]
    )
    state_estimate = stable_argmax(state_scores)
    state_dx, state_dy = shortest_displacement(state_estimate, target)
    state_unitary = move_unitary(state_dx, state_dy)
    state_final = state_unitary @ actual_state @ state_unitary.conj().T
    state_hidden_final = translate_index(hidden, state_dx, state_dy)
    state_movement_cost = abs(state_dx) + abs(state_dy)
    senses = len(observations)
    return {
        "eta": eta,
        "strategy": strategy.name,
        "known_start": known_start,
        "hidden": hidden,
        "target": target,
        "estimate": estimate,
        "state_estimate": state_estimate,
        "observations": observations,
        "posterior": belief.weights.tolist(),
        "entropies": entropies,
        "senses": senses,
        "moves": movement_cost,
        "total_cost": senses + movement_cost + 1,
        "confidence": float(np.max(belief.weights)),
        "posterior_entropy": entropy_bits(belief.weights),
        "label_success": label_success,
        "operational_score": operational_score,
        "state_label_success": int(state_hidden_final == target),
        "state_operational_score": born_probability(state_final, target_projector),
        "state_moves": state_movement_cost,
        "state_total_cost": senses + state_movement_cost + 1,
    }


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str, bool], list[dict[str, object]]] = {}
    for record in records:
        key = (float(record["eta"]), str(record["strategy"]), bool(record["known_start"]))
        groups.setdefault(key, []).append(record)
    rows = []
    for (eta, strategy, known_start), group in sorted(groups.items()):
        def mean(field: str) -> float:
            return float(np.mean([float(item[field]) for item in group]))

        def standard_error(field: str) -> float:
            values = np.array([float(item[field]) for item in group])
            return float(np.std(values, ddof=1) / np.sqrt(len(values)))

        success = np.array([float(item["label_success"]) for item in group])
        rows.append(
            {
                "eta": eta,
                "strategy": strategy,
                "known_start": int(known_start),
                "episodes": len(group),
                "label_success": mean("label_success"),
                "label_success_se": float(np.std(success, ddof=1) / np.sqrt(len(success))),
                "operational_score": mean("operational_score"),
                "operational_score_se": standard_error("operational_score"),
                "state_label_success": mean("state_label_success"),
                "state_label_success_se": standard_error("state_label_success"),
                "state_operational_score": mean("state_operational_score"),
                "state_operational_score_se": standard_error("state_operational_score"),
                "mean_confidence": mean("confidence"),
                "posterior_entropy_bits": mean("posterior_entropy"),
                "mean_senses": mean("senses"),
                "mean_moves": mean("moves"),
                "mean_total_cost": mean("total_cost"),
                "state_mean_moves": mean("state_moves"),
                "state_mean_total_cost": mean("state_total_cost"),
            }
        )
    return rows


def calibration_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for eta in sorted({float(record["eta"]) for record in records}):
        for strategy in sorted({str(record["strategy"]) for record in records}):
            subset = [
                record
                for record in records
                if float(record["eta"]) == eta
                and str(record["strategy"]) == strategy
                and not bool(record["known_start"])
            ]
            for low, high in zip(bins[:-1], bins[1:]):
                selected = [
                    record
                    for record in subset
                    if low <= float(record["confidence"]) < high
                    or (high == 1.0 and float(record["confidence"]) == 1.0)
                ]
                if selected:
                    rows.append(
                        {
                            "eta": eta,
                            "strategy": strategy,
                            "bin_low": low,
                            "bin_high": high,
                            "count": len(selected),
                            "mean_confidence": float(
                                np.mean([float(record["confidence"]) for record in selected])
                            ),
                            "empirical_success": float(
                                np.mean([float(record["label_success"]) for record in selected])
                            ),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analytic_one_shot(eta: float) -> tuple[float, float]:
    """Exact label accuracy and operational score after outcome-directed move."""
    return (1.0 + 2.0 * eta) / 9.0, (1.0 + 2.0 * eta) / 3.0


def make_figures(rows: list[dict[str, object]], records: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    unknown = [row for row in rows if not int(row["known_start"])]
    strategies = [strategy.name for strategy in STRATEGIES]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(strategies)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for strategy, color in zip(strategies, colors):
        selected = sorted(
            [row for row in unknown if row["strategy"] == strategy], key=lambda row: row["eta"]
        )
        axes[0].plot(
            [row["eta"] for row in selected],
            [row["label_success"] for row in selected],
            "o-",
            label=strategy,
            color=color,
        )
        axes[1].plot(
            [row["mean_total_cost"] for row in selected],
            [row["label_success"] for row in selected],
            "o-",
            label=strategy,
            color=color,
        )
    eta_grid = np.linspace(0, 1, 100)
    axes[0].plot(eta_grid, [(1 + 2 * eta) / 9 for eta in eta_grid], "k--", label="one-shot exact")
    axes[0].axhline(1 / 9, color="gray", ls=":", label="chance")
    axes[0].set(
        xlabel=r"measurement strength $\eta$",
        ylabel="hidden-label navigation success",
        title="Localization of the preparation label",
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(
        xlabel="mean total interventions",
        ylabel="hidden-label navigation success",
        title="Accuracy--cost frontier",
    )

    for strategy, color in zip(strategies, colors):
        selected = [row for row in unknown if row["strategy"] == strategy]
        axes[2].scatter(
            [row["label_success"] for row in selected],
            [row["operational_score"] for row in selected],
            label=strategy,
            color=color,
            alpha=0.85,
        )
        axes[2].scatter(
            [row["state_label_success"] for row in selected],
            [row["state_operational_score"] for row in selected],
            marker="^",
            facecolors="none",
            edgecolors=color,
            alpha=0.85,
        )
    axes[2].scatter([], [], marker="o", color="black", label="label-MAP move")
    axes[2].scatter([], [], marker="^", facecolors="none", edgecolors="black", label="predictive-state move")
    axes[2].plot([0, 1], [0, 1], "k:", label="equal criteria")
    axes[2].set(
        xlabel="hidden-label success",
        ylabel="terminal target-state fidelity",
        title="Goal definition selects the navigation policy",
    )
    axes[2].legend(fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "localization_performance.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for eta, color in zip((0.0, 0.5, 1.0), ("#777777", "#f18f01", "#235789")):
        entropy_by_sense = []
        for sense_count in range(6):
            subset = [
                record
                for record in records
                if float(record["eta"]) == eta
                and str(record["strategy"]) == "fixed-5"
            ]
            entropy_by_sense.append(
                float(np.mean([record["entropies"][sense_count] for record in subset]))
            )
        axes[0].plot(range(6), entropy_by_sense, "o-", color=color, label=fr"$\eta={eta}$")
    axes[0].axhline(np.log2(9), color="black", ls=":", lw=1)
    axes[0].set(
        xlabel="weak measurements",
        ylabel="posterior entropy (bits)",
        title="Information retained about initial coordinate",
    )
    axes[0].legend()

    example = next(
        record
        for record in records
        if float(record["eta"]) == 0.8
        and str(record["strategy"]) == "fixed-5"
        and int(record["label_success"]) == 1
    )
    posterior = np.array(example["posterior"]).reshape(3, 3)
    image = axes[1].imshow(posterior, origin="lower", cmap="magma", vmin=0, vmax=posterior.max())
    for x in range(3):
        for y in range(3):
            axes[1].text(y, x, f"{posterior[x,y]:.2f}", ha="center", va="center", color="white")
    axes[1].set(
        xticks=range(3),
        yticks=range(3),
        xlabel="phase coordinate y",
        ylabel="phase coordinate x",
        title=f"Example posterior; observations={example['observations']}",
    )
    fig.colorbar(image, ax=axes[1], label="posterior probability")
    fig.savefig(FIGURES / "localization_beliefs.png", dpi=220)
    plt.close(fig)


def run(episodes: int = 3000, seed: int = 20260811) -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    etas = (0.0, 0.2, 0.5, 0.8, 1.0)
    records: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for eta in etas:
        for strategy in STRATEGIES:
            for episode in range(episodes):
                record = run_episode(eta, strategy, rng)
                records.append(record)
                if episode < 2:
                    examples.append(record)
        # Known-start control needs no sensing; it is independent of eta apart
        # from the unused instrument and is retained at each eta for plotting.
        for _ in range(episodes):
            records.append(run_episode(eta, STRATEGIES[0], rng, known_start=True))

    rows = aggregate(records)
    calibration = calibration_rows(records)
    write_csv(RESULTS / "localization_summary.csv", rows)
    write_csv(RESULTS / "posterior_calibration.csv", calibration)

    analytic = [
        {
            "eta": eta,
            "label_accuracy": analytic_one_shot(eta)[0],
            "operational_score": analytic_one_shot(eta)[1],
        }
        for eta in etas
    ]
    write_csv(RESULTS / "analytic_one_shot.csv", analytic)
    (RESULTS / "example_episodes.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(rows, records)

    one_shot_errors = []
    for row in rows:
        if row["strategy"] == "fixed-1" and not int(row["known_start"]):
            exact_label, exact_operational = analytic_one_shot(float(row["eta"]))
            one_shot_errors.extend(
                [
                    abs(float(row["label_success"]) - exact_label),
                    abs(float(row["operational_score"]) - exact_operational),
                ]
            )
    manifest = {
        "seed": seed,
        "episodes_per_condition": episodes,
        "etas": etas,
        "strategies": [strategy.__dict__ for strategy in STRATEGIES],
        "total_episodes": len(records),
        "maximum_monte_carlo_error_against_one_shot_formula": max(one_shot_errors),
        "criteria": {
            "label_success": "whether inferred translation maps hidden preparation label to target",
            "operational_score": "final state fidelity with target phase-grid projector",
            "state_controller": "counterfactual move centered on current Bayesian ensemble state",
        },
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    print(json.dumps(run(args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
