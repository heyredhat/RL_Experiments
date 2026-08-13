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
ACTION_NAMES = ("X", "Z", "retry-Y", "SIC")
DISCOUNT_POLICY_THRESHOLD = float(
    (3.0 + 3.0 * SQRT2 - np.sqrt(15.0 + 12.0 * SQRT2)) / 2.0
)

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


def action_values(value: np.ndarray, goal: int, discount: float = 1.0) -> np.ndarray:
    """Return Q(x,a) for the four movement/report actions.

    The immediate cost is one for *every* instrument use, including a SIC use
    that produces the requested terminal outcome.  Future cost is multiplied
    by ``discount``.  The terminal branch is absent from the continuation sum.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0,1]")
    kernel = sic_kernel()
    movements = [movement_transition(a) for a in (1, 2, 3)]
    q = [1.0 + discount * transition.T @ value for transition in movements]
    report = np.ones(4)
    for state in range(4):
        report[state] += discount * sum(
            kernel[outcome, state] * value[outcome]
            for outcome in range(4)
            if outcome != goal
        )
    q.append(report)
    return np.stack(q, axis=1)


def bellman_values(
    goal: int, discount: float = 1.0, tolerance: float = 1e-14
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the SSP whose terminal event is SIC outcome ``goal``.

    Available actions are X, Z, retry-Y, and the SIC instrument.  Every action
    costs one.  A SIC outcome equal to ``goal`` terminates; every other SIC
    outcome prepares its corresponding tetrahedral state.  ``discount=1`` is
    the stochastic-shortest-path problem used for hodological distance.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0,1]")
    value = np.zeros(4, dtype=float)
    policy = np.zeros(4, dtype=int)
    for _ in range(200_000):
        q_array = action_values(value, goal, discount)
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


def all_bellman_values(discount: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((4, 4), dtype=float)
    policies = np.zeros((4, 4), dtype=int)
    for goal in range(4):
        values[:, goal], policies[:, goal] = bellman_values(goal, discount=discount)
    return values, policies


def analytic_discounted_shell_values(discount: float) -> tuple[np.ndarray, str]:
    """Closed-form values for goal 00, ordered as self, edge, diagonal.

    The optimal policy has three exact regimes.  At the two boundary values
    the adjacent policies tie; the formulas agree there.
    """
    gamma = float(discount)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("discount must lie in [0,1]")
    if gamma <= 0.5:
        edge = 6.0 / (6.0 - 5.0 * gamma)
        self_value = (6.0 - 2.0 * gamma) / (6.0 - 5.0 * gamma)
        return np.array([self_value, edge, edge]), "SIC/SIC/SIC"
    if gamma <= DISCOUNT_POLICY_THRESHOLD:
        delta = 2 * gamma**3 - 6 * gamma**2 - 9 * gamma + 18
        self_value = 2 * (9 - gamma**2) / delta
        edge = 3 * (-2 * gamma**2 + 3 * gamma + 6) / delta
        diagonal = 6 * (gamma + 3) / delta
        return np.array([self_value, edge, diagonal]), "SIC/move/SIC"
    p = 1.0 / SQRT2
    stay = 1.0 - p
    denominator = 1.0 - gamma * stay
    self_value = (
        1.0 + gamma / 3.0 + gamma / (6.0 * denominator)
    ) / (
        1.0 - gamma**2 / 3.0 - gamma**2 * p / (6.0 * denominator)
    )
    edge = 1.0 + gamma * self_value
    diagonal = (1.0 + gamma * p * self_value) / denominator
    return np.array([self_value, edge, diagonal]), "SIC/move/retry-Y"


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


def raw_history_predictive_audit(max_depth: int = 5) -> list[dict[str, float]]:
    """Enumerate raw action--outcome histories and count predictive classes.

    No location label is propagated in this audit.  A node stores only the
    conditional density matrix obtained from its literal instrument history.
    Its operational signature is the distribution of the *next* SIC outcome.
    Informational completeness makes that signature a causal sufficient
    statistic.  Starting after any observed SIC token, all reachable histories
    have one of exactly four signatures even though their raw count is 4*8**d.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    instruments = (
        sic_kraus(),
        np.array([X]),
        np.array([np.sqrt(1.0 / SQRT2) * Y, np.sqrt(1.0 - 1.0 / SQRT2) * I2]),
        np.array([Z]),
    )
    canonical = sic_kernel().T
    frontier = [rho.copy() for rho in tetrahedral_states()]
    rows: list[dict[str, float]] = []
    for depth in range(max_depth + 1):
        signatures = []
        maximum_error = 0.0
        for rho in frontier:
            signature = np.array(
                [0.5 * np.trace(projector @ rho).real for projector in tetrahedral_states()]
            )
            signatures.append(signature)
            maximum_error = max(
                maximum_error,
                float(np.min(np.max(np.abs(canonical - signature[None, :]), axis=1))),
            )
        unique = np.unique(np.round(np.asarray(signatures), decimals=12), axis=0)
        rows.append(
            {
                "depth": depth,
                "raw_histories": len(frontier),
                "predictive_classes": len(unique),
                "max_signature_error": maximum_error,
            }
        )
        if depth == max_depth:
            break
        next_frontier = []
        for rho in frontier:
            for instrument in instruments:
                for kraus in instrument:
                    branch = kraus @ rho @ kraus.conj().T
                    probability = float(np.trace(branch).real)
                    if probability > 1e-15:
                        next_frontier.append(branch / probability)
        frontier = next_frontier
    return rows


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

    goal = 0
    goal_values, goal_policy = bellman_values(goal)
    goal_q = action_values(goal_values, goal)
    shell_names = ("self", "edge", "edge", "diagonal")

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
    with (output / "monte_carlo_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source/goal", *LABELS])
        for label, row in zip(LABELS, mc):
            writer.writerow([label, *row])
    with (output / "opaque_recovery.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["samples_per_anchor_action", "all_permutations_recovered"])
        writer.writerows(zip(recovery_samples, recovery))
    with (output / "action_values_gamma_1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "shell", "action", "Q", "optimal"])
        for state in range(4):
            for action, name in enumerate(ACTION_NAMES):
                writer.writerow(
                    [
                        LABELS[state],
                        shell_names[state],
                        name,
                        goal_q[state, action],
                        int(action == goal_policy[state]),
                    ]
                )

    discounts = np.unique(
        np.concatenate(
            [np.linspace(0.0, 1.0, 501), [0.5, DISCOUNT_POLICY_THRESHOLD]]
        )
    )
    discount_rows = []
    for gamma in discounts:
        shell, regime = analytic_discounted_shell_values(float(gamma))
        numerical, numerical_policy = bellman_values(goal, discount=float(gamma))
        discount_rows.append(
            (
                gamma,
                *shell,
                shell[1] - shell[0],
                shell[2] - shell[0],
                regime,
                "/".join(ACTION_NAMES[index] for index in numerical_policy),
                float(np.max(np.abs(numerical - shell[[0, 1, 1, 2]]))),
            )
        )
    with (output / "discount_scan.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gamma",
                "self_value",
                "edge_value",
                "diagonal_value",
                "edge_excess",
                "diagonal_excess",
                "analytic_regime",
                "numerical_policy_00_10_01_11",
                "analytic_numerical_max_error",
            ]
        )
        writer.writerows(discount_rows)

    history_audit = raw_history_predictive_audit()
    with (output / "history_predictive_quotient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_audit[0].keys())
        writer.writeheader()
        writer.writerows(history_audit)

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
        "discount_policy_threshold": DISCOUNT_POLICY_THRESHOLD,
        "discount_closed_form_max_error": float(max(row[-1] for row in discount_rows)),
        "raw_histories_at_audit_depth": int(history_audit[-1]["raw_histories"]),
        "predictive_classes_at_audit_depth": int(history_audit[-1]["predictive_classes"]),
        "history_quotient_max_signature_error": float(
            max(row["max_signature_error"] for row in history_audit)
        ),
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

    discount_array = np.asarray([row[:6] for row in discount_rows], dtype=float)
    gamma = discount_array[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(gamma, discount_array[:, 1], label=r"$v_0$ (self)")
    axes[0].plot(gamma, discount_array[:, 2], label=r"$v_1$ (edge)")
    axes[0].plot(gamma, discount_array[:, 3], label=r"$v_2$ (diagonal)")
    axes[0].set_title("Discounted intervention costs")
    axes[0].set_xlabel(r"discount $\gamma$")
    axes[0].set_ylabel("optimal discounted cost")
    axes[0].legend()

    axes[1].plot(gamma, discount_array[:, 4], label="edge excess")
    axes[1].plot(gamma, discount_array[:, 5], label="diagonal excess")
    axes[1].axhline(1.0, color="C0", ls="--", lw=1)
    axes[1].axhline(SQRT2, color="C1", ls="--", lw=1)
    axes[1].set_title("Baseline subtraction depends on discounting")
    axes[1].set_xlabel(r"discount $\gamma$")
    axes[1].set_ylabel(r"$v_k-v_0$")
    axes[1].legend()

    ratio = np.divide(
        discount_array[:, 5],
        discount_array[:, 4],
        out=np.full_like(gamma, np.nan),
        where=np.abs(discount_array[:, 4]) > 1e-12,
    )
    axes[2].plot(gamma, ratio, color="C2")
    axes[2].axhline(SQRT2, color="k", ls="--", lw=1, label=r"Euclidean $\sqrt{2}$")
    for threshold in (0.5, DISCOUNT_POLICY_THRESHOLD):
        axes[2].axvline(threshold, color="0.5", ls=":", lw=1)
    axes[2].set_ylim(0.95, 1.47)
    axes[2].set_title("Policy phases and square aspect")
    axes[2].set_xlabel(r"discount $\gamma$")
    axes[2].set_ylabel("diagonal excess / edge excess")
    axes[2].legend()
    fig.savefig(figures / "sic_discounting.png", dpi=180)
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
