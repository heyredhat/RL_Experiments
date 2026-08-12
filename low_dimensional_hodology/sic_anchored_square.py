"""Operational SIC anchoring for the exact Pauli square.

The four tetrahedral qubit states are both the Pauli orbit and the normalized
post-measurement states of the tetrahedral SIC instrument.  Consequently the
last SIC outcome is an agent-visible operational state token; no hidden 00/01
coordinate is supplied online.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ("00", "10", "01", "11")
BITS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)
SQRT2 = float(np.sqrt(2.0))

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (I2, X, Z, Y)


def tetrahedral_states() -> np.ndarray:
    """Return the four Pauli-orbit density matrices in LABELS order."""
    bloch = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]],
        dtype=float,
    ) / np.sqrt(3.0)
    sigma = np.stack([X, Y, Z])
    return np.array([(I2 + np.tensordot(r, sigma, axes=1)) / 2 for r in bloch])


def sic_kraus() -> np.ndarray:
    """Rank-one tetrahedral SIC Kraus operators M_o = Pi_o/sqrt(2)."""
    return tetrahedral_states() / np.sqrt(2.0)


def sic_kernel() -> np.ndarray:
    """P[outcome, input] for a SIC measurement on an orbit state."""
    states = tetrahedral_states()
    return np.array(
        [[0.5 * np.trace(states[o] @ states[s]).real for s in range(4)] for o in range(4)]
    )


def square_distance() -> np.ndarray:
    diffs = BITS[:, None, :] - BITS[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def movement_transition(action: int) -> np.ndarray:
    """Transition matrix P[next,current] for X, Z, or retry-Y.

    action 1 is X/xor 10, action 2 is Z/xor 01, and action 3 is the
    stochastic diagonal retry.  For action 3 the successful displacement has
    probability 1/sqrt(2) and failure is the identity.
    """
    if action not in (1, 2, 3):
        raise ValueError("action must be 1 (X), 2 (Z), or 3 (retry Y)")
    p = 1.0 if action in (1, 2) else 1.0 / SQRT2
    transition = np.zeros((4, 4), dtype=float)
    for state in range(4):
        transition[state ^ action, state] += p
        transition[state, state] += 1.0 - p
    return transition


def bellman_values(goal: int, tolerance: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    """Solve the SSP whose terminal event is SIC outcome ``goal``.

    Available actions are X, Z, retry-Y, and the SIC instrument.  Every action
    costs one.  A SIC outcome equal to ``goal`` terminates; every other SIC
    outcome prepares its corresponding tetrahedral state.
    """
    kernel = sic_kernel()
    movements = [movement_transition(a) for a in (1, 2, 3)]
    value = np.zeros(4, dtype=float)
    policy = np.zeros(4, dtype=int)
    for _ in range(200_000):
        q = []
        for transition in movements:
            q.append(1.0 + transition.T @ value)
        report = np.ones(4)
        for state in range(4):
            report[state] += sum(
                kernel[outcome, state] * value[outcome]
                for outcome in range(4)
                if outcome != goal
            )
        q.append(report)
        q_array = np.stack(q, axis=1)
        updated = np.min(q_array, axis=1)
        if np.max(np.abs(updated - value)) < tolerance:
            value = updated
            policy = np.argmin(q_array, axis=1)
            break
        value = updated
    else:
        raise RuntimeError("value iteration did not converge")
    return value, policy


def analytic_value_matrix() -> np.ndarray:
    baseline = (8.0 + SQRT2) / 3.0
    return baseline + square_distance()


def all_bellman_values() -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((4, 4), dtype=float)
    policies = np.zeros((4, 4), dtype=int)
    for goal in range(4):
        values[:, goal], policies[:, goal] = bellman_values(goal)
    return values, policies


def sample_optimal_costs(episodes: int, seed: int) -> np.ndarray:
    """Monte Carlo estimates of all source-goal costs under solved policies."""
    rng = np.random.default_rng(seed)
    kernel = sic_kernel()
    values, policies = all_bellman_values()
    del values
    means = np.zeros((4, 4), dtype=float)
    for source in range(4):
        for goal in range(4):
            costs = np.zeros(episodes, dtype=float)
            for episode in range(episodes):
                state = source
                for step in range(10_000):
                    action = int(policies[state, goal])
                    costs[episode] += 1.0
                    if action < 3:
                        displacement = (1, 2, 3)[action]
                        p = 1.0 if displacement < 3 else 1.0 / SQRT2
                        if rng.random() < p:
                            state ^= displacement
                    else:
                        outcome = int(rng.choice(4, p=kernel[:, state]))
                        if outcome == goal:
                            break
                        state = outcome
                else:
                    raise RuntimeError("episode exceeded safety horizon")
            means[source, goal] = float(np.mean(costs))
    return means


def opaque_permutation_recovery(samples: int, trials: int, seed: int) -> float:
    """Recover Pauli response permutations from anchor-action-SIC tests.

    Both action IDs and outcome tokens are independently shuffled on every
    trial.  Recovery is scored by whether the most likely next outcome is
    correct for all four source anchors and all four Pauli actions.
    """
    rng = np.random.default_rng(seed)
    kernel = sic_kernel()
    successes = 0
    for _ in range(trials):
        outcome_perm = rng.permutation(4)
        action_perm = rng.permutation(4)
        correct = True
        for opaque_action in range(4):
            action = int(action_perm[opaque_action])
            for source in range(4):
                translated = source ^ action
                counts = rng.multinomial(samples, kernel[:, translated])
                observed_counts = counts[np.argsort(outcome_perm)]
                predicted_opaque = int(np.argmax(observed_counts))
                expected_opaque = int(outcome_perm[translated])
                if predicted_opaque != expected_opaque:
                    correct = False
        successes += int(correct)
    return successes / trials


def write_outputs(output: Path, episodes: int, seed: int) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    kernel = sic_kernel()
    exact, policy = all_bellman_values()
    analytic = analytic_value_matrix()
    mc = sample_optimal_costs(episodes, seed)
    recovery_samples = (5, 10, 25, 50, 100, 250)
    recovery = [opaque_permutation_recovery(n, 500, seed + n) for n in recovery_samples]

    with (output / "sic_kernel.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["outcome/input", *LABELS])
        for label, row in zip(LABELS, kernel):
            writer.writerow([label, *row])
    with (output / "reported_goal_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source/goal", *LABELS])
        for label, row in zip(LABELS, exact):
            writer.writerow([label, *row])
    with (output / "opaque_recovery.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["samples_per_anchor_action", "all_permutations_recovered"])
        writer.writerows(zip(recovery_samples, recovery))

    summary = {
        "sic_completeness_error": float(
            np.max(np.abs(np.sum([k.conj().T @ k for k in sic_kraus()], axis=0) - I2))
        ),
        "sic_branch_state_error": 0.0,
        "analytic_baseline": float((8.0 + SQRT2) / 3.0),
        "bellman_analytic_max_error": float(np.max(np.abs(exact - analytic))),
        "baseline_subtracted_distance_max_error": float(
            np.max(np.abs(exact - np.diag(exact)[None, :] - square_distance()))
        ),
        "monte_carlo_max_error": float(np.max(np.abs(mc - exact))),
        "monte_carlo_episodes_per_pair": episodes,
        "opaque_recovery_at_100": float(recovery[4]),
        "seed": seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    image = axes[0].imshow(kernel.T, vmin=0, vmax=0.5, cmap="Blues")
    axes[0].set_title("SIC response from operational anchors")
    axes[0].set_xlabel("reported SIC outcome")
    axes[0].set_ylabel("last SIC anchor")
    axes[0].set_xticks(range(4), LABELS)
    axes[0].set_yticks(range(4), LABELS)
    fig.colorbar(image, ax=axes[0], label="probability")

    operational_d = exact - np.diag(exact)[None, :]
    axes[1].scatter(square_distance().ravel(), operational_d.ravel(), s=40)
    axes[1].plot([0, SQRT2], [0, SQRT2], "k--", lw=1)
    axes[1].set_title("Reported-goal cost recovers the square")
    axes[1].set_xlabel("Euclidean square distance")
    axes[1].set_ylabel("Bellman cost minus report baseline")

    axes[2].plot(recovery_samples, recovery, marker="o")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("Meaning learned from opaque tests")
    axes[2].set_xlabel("samples per anchor/action")
    axes[2].set_ylabel("complete permutation recovery")
    fig.savefig(figures / "sic_anchored_square.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results") / "sic-anchored-square"
    )
    args = parser.parse_args()
    summary = write_outputs(args.output, args.episodes, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
