"""Actively choose whether to sense, move, or commit in a predictive atlas.

This successor to :mod:`predictive_atlas` removes the fixed 48-probe scan.
Beacon likelihoods and blind movement transitions are calibrated from delayed
landmark experiments.  At run time the controller begins from a uniform prior
and gives every beacon, movement, and terminal commitment the same unit cost.
A goal-aware value-of-information rule decides which intervention is worth its
cost.  Coordinates, Kraus operators, exact response fields, and the current
place are reserved for offline validation.

Production run::

    conda run --no-capture-output -n qbist_spacetime \
      python active_predictive_atlas.py --output results/active-atlas
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from predictive_atlas import (
    BEACON_ACTIONS,
    DIAGONAL_SUCCESS,
    N_MOVES,
    N_SITES,
    PLACE_NAMES,
    TERMINAL_PROBE,
    _set_site,
    _true_site,
    beacon_fields,
    learned_transition_model,
    planning_values,
    update_belief,
)
from quantum_environments import QuantumEnvironment, environment_definition
from spatial_hodology import distance_correlation, exact_movement_costs, geometry_metrics, metric_mds


SENSE_COST = 1.0
MOVE_COST = 1.0
COMMIT_COST = 1.0


@dataclass
class ActiveSummary:
    condition: str
    seed: int
    all_pairs_success: float
    reset_success: float
    mean_senses: float
    mean_moves: float
    mean_total_interventions: float
    mean_commit_confidence: float
    mean_initial_entropy: float
    mean_commit_entropy: float
    beacon_mae: float
    transition_tv_error: float
    movement_exact_correlation: float
    movement_stress_1d: float
    movement_stress_2d: float
    movement_stress_3d: float
    movement_procrustes_r2: float
    total_stress_2d: float
    total_stress_3d: float
    total_procrustes_r2: float
    sensing_distance_correlation: float
    sensing_additive_r2: float


def _workspace_path(path: str) -> Path:
    workspace = Path(__file__).resolve().parent
    candidate = (workspace / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"output must stay inside {workspace}")
    return candidate


def entropy(belief: np.ndarray) -> float:
    safe = np.clip(np.asarray(belief, dtype=float), 1e-15, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def calibrate_beacons(
    *,
    environment: str,
    trials_per_site_beacon: int,
    seed: int,
    prior: float = 0.5,
) -> np.ndarray:
    """Estimate P(outcome | landmark, beacon) from delayed-label surveys."""
    env = QuantumEnvironment(environment=environment, weak_q=DIAGONAL_SUCCESS, seed=seed)
    counts = np.full((len(BEACON_ACTIONS), N_SITES, 2), prior, dtype=float)
    for site in range(N_SITES):
        for beacon_index, action in enumerate(BEACON_ACTIONS):
            for _ in range(trials_per_site_beacon):
                _set_site(env, site)
                outcome = env.step(action)
                # The terminal report verifies the chart label after sensing;
                # it never supplies an input to the learned likelihood table.
                label = env.step(TERMINAL_PROBE)
                if label != site:
                    raise RuntimeError("QND beacon changed the landmark")
                counts[beacon_index, label, outcome] += 1.0
    return counts / counts.sum(axis=2, keepdims=True)


def exact_beacon_model(environment: str = "qudit-grid-3x3-beacons") -> np.ndarray:
    fields = beacon_fields(environment)
    return np.stack((1.0 - fields, fields), axis=2)


def beacon_model_mae(learned: np.ndarray, environment: str) -> float:
    return float(np.mean(np.abs(learned - exact_beacon_model(environment))))


def exact_joint_model(environment: str) -> np.ndarray:
    """Privileged movement kernel used only to audit the learned survey."""
    definition = environment_definition(environment, weak_q=DIAGONAL_SUCCESS)
    joint = np.zeros((N_MOVES, 2, N_SITES, N_SITES), dtype=float)
    for action, measurement in enumerate(definition.measurements[:N_MOVES]):
        for outcome, kraus_events in enumerate(measurement.outcome_kraus):
            for source in range(N_SITES):
                ket = np.eye(N_SITES, dtype=complex)[:, source]
                for operator in kraus_events:
                    branch = operator @ ket
                    joint[action, outcome, source] += np.abs(branch) ** 2
    return joint


def joint_model_tv_error(learned: np.ndarray, environment: str) -> float:
    exact = exact_joint_model(environment)
    return float(0.5 * np.abs(learned - exact).sum(axis=(1, 3)).mean())


def update_beacon_belief(
    belief: np.ndarray,
    likelihoods: np.ndarray,
    beacon_index: int,
    outcome: int,
) -> np.ndarray:
    posterior = np.asarray(belief, dtype=float) * likelihoods[beacon_index, :, outcome]
    normalizer = posterior.sum()
    if normalizer <= 1e-15:
        return np.full(N_SITES, 1.0 / N_SITES)
    return posterior / normalizer


def beacon_branches(
    belief: np.ndarray,
    likelihoods: np.ndarray,
    beacon_index: int,
) -> list[tuple[float, np.ndarray, int]]:
    branches = []
    for outcome in (0, 1):
        probability = float(belief @ likelihoods[beacon_index, :, outcome])
        if probability > 1e-15:
            branches.append(
                (
                    probability,
                    update_beacon_belief(belief, likelihoods, beacon_index, outcome),
                    outcome,
                )
            )
    return branches


def expected_information_gain(
    belief: np.ndarray,
    likelihoods: np.ndarray,
    beacon_index: int,
) -> float:
    expected = sum(
        probability * entropy(posterior)
        for probability, posterior, _ in beacon_branches(belief, likelihoods, beacon_index)
    )
    return entropy(belief) - expected


def choose_entropy_action(
    belief: np.ndarray,
    goal: int,
    likelihoods: np.ndarray,
    q_values: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[str, int, dict[str, float]]:
    """Goal-agnostic active-sensing control used as a matched baseline."""
    if float(belief.max()) < confidence_threshold:
        gains = np.array(
            [expected_information_gain(belief, likelihoods, b) for b in range(4)]
        )
        return "sense", int(np.argmax(gains)), {
            f"eig:{index}": float(value) for index, value in enumerate(gains)
        }
    if int(np.argmax(belief)) == goal:
        return "commit", -1, {"confidence": float(belief[goal])}
    expected_q = np.einsum("s,sa->a", belief, q_values[:, goal, :])
    return "move", int(np.argmin(expected_q)), {
        f"move:{index}": float(value) for index, value in enumerate(expected_q)
    }


def policy_partition(
    belief: np.ndarray,
    goal: int,
    q_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Push a place belief forward to the next-decision equivalence classes.

    Decision label 8 denotes terminal commitment; labels 0--7 denote the
    learned optimal blind movement.  Sites requiring the same next action are
    deliberately pooled.  Active sensing therefore resolves only distinctions
    relevant to the present goal.
    """
    decisions = np.argmin(q_values[:, goal, :], axis=1).astype(int)
    decisions[goal] = N_MOVES
    masses = np.zeros(N_MOVES + 1, dtype=float)
    for site, decision in enumerate(decisions):
        masses[decision] += float(belief[site])
    return decisions, masses


def choose_policy_partition_action(
    belief: np.ndarray,
    goal: int,
    likelihoods: np.ndarray,
    q_values: np.ndarray,
    *,
    decision_error_penalty: float,
    sensing_lookahead: int,
    value_cache: dict[tuple[int, int, tuple[float, ...]], float] | None = None,
    action_cache: dict[tuple[int, tuple[float, ...]], tuple[str, int, dict[str, float]]] | None = None,
) -> tuple[str, int, dict[str, float]]:
    """Buy a beacon only when finite-horizon goal-relevant risk justifies it.

    The Bayes risk for the next decision is ``1 - max_d P(d | belief, goal)``.
    A probe has unit cost.  A short exact lookahead over possible beacon
    outcomes compares stopping with ``penalty * decision Bayes risk`` against
    buying one or more observations.  Multi-step lookahead matters because one
    weak outcome may not change the modal decision even when a short sequence
    would.  This puts sensing and acting in common intervention units without
    a fixed confidence threshold.
    """
    def stop_cost(current: np.ndarray) -> float:
        _, current_masses = policy_partition(current, goal, q_values)
        return decision_error_penalty * (1.0 - float(current_masses.max()))

    belief_key = tuple(np.round(belief, 8))
    action_key = (goal, belief_key)
    if action_cache is not None and action_key in action_cache:
        return action_cache[action_key]
    memo = value_cache if value_cache is not None else {}

    def risk_value(current: np.ndarray, depth: int) -> float:
        key = (goal, depth, tuple(np.round(current, 8)))
        if key in memo:
            return memo[key]
        best = stop_cost(current)
        if depth > 0:
            for beacon_index in range(len(BEACON_ACTIONS)):
                candidate = SENSE_COST + sum(
                    probability * risk_value(posterior, depth - 1)
                    for probability, posterior, _ in beacon_branches(
                        current, likelihoods, beacon_index
                    )
                )
                best = min(best, candidate)
        memo[key] = best
        return best

    current_stop = stop_cost(belief)
    candidates = []
    for beacon_index in range(len(BEACON_ACTIONS)):
        candidates.append(
            SENSE_COST
            + sum(
                probability * risk_value(posterior, max(0, sensing_lookahead - 1))
                for probability, posterior, _ in beacon_branches(
                    belief, likelihoods, beacon_index
                )
            )
        )
    best_beacon = int(np.argmin(candidates))
    scores = {
        f"sense-cost:{index}": float(value) for index, value in enumerate(candidates)
    }
    scores["stop-cost"] = current_stop
    if candidates[best_beacon] + 1e-10 < current_stop:
        result = ("sense", best_beacon, scores)
        if action_cache is not None:
            action_cache[action_key] = result
        return result
    _, masses = policy_partition(belief, goal, q_values)
    decision = int(np.argmax(masses))
    if decision == N_MOVES:
        result = ("commit", -1, scores)
    else:
        result = ("move", decision, scores)
    if action_cache is not None:
        action_cache[action_key] = result
    return result


def run_episode(
    *,
    env: QuantumEnvironment,
    source: int,
    goal: int,
    condition: str,
    likelihoods: np.ndarray,
    transition_joint: np.ndarray,
    q_values: np.ndarray,
    max_interventions: int,
    failure_penalty: float,
    confidence_threshold: float,
    sensing_lookahead: int,
    value_cache: dict[tuple[int, int, tuple[float, ...]], float] | None = None,
    action_cache: dict[tuple[int, tuple[float, ...]], tuple[str, int, dict[str, float]]] | None = None,
) -> dict[str, object]:
    _set_site(env, source)
    belief = np.eye(N_SITES)[source].copy() if condition == "oracle" else np.full(N_SITES, 1 / N_SITES)
    beliefs = [belief.tolist()]
    true_sites = [source]
    actions: list[int] = []
    outcomes: list[int] = []
    kinds: list[str] = []
    decision_margins: list[float] = []
    senses = 0
    moves = 0
    committed = False
    terminal_outcome = -1

    if condition == "fixed-12":
        for _ in range(12):
            for beacon_index, action in enumerate(BEACON_ACTIONS):
                outcome = env.step(action)
                belief = update_beacon_belief(belief, likelihoods, beacon_index, outcome)
                senses += 1
                actions.append(action)
                outcomes.append(outcome)
                kinds.append("fixed-sense")
                beliefs.append(belief.tolist())
                true_sites.append(_true_site(env))

    for _ in range(max(0, max_interventions - len(actions) - 1)):
        if condition == "oracle":
            if int(np.argmax(belief)) == goal:
                kind, index, scores = "commit", -1, {"confidence": float(belief[goal])}
            else:
                expected_q = np.einsum("s,sa->a", belief, q_values[:, goal, :])
                kind, index = "move", int(np.argmin(expected_q))
                scores = {f"move:{action}": float(value) for action, value in enumerate(expected_q)}
        elif condition == "fixed-12":
            if int(np.argmax(belief)) == goal:
                kind, index, scores = "commit", -1, {"confidence": float(belief[goal])}
            else:
                expected_q = np.einsum("s,sa->a", belief, q_values[:, goal, :])
                kind, index = "move", int(np.argmin(expected_q))
                scores = {f"move:{action}": float(value) for action, value in enumerate(expected_q)}
        elif condition == "entropy":
            kind, index, scores = choose_entropy_action(
                belief,
                goal,
                likelihoods,
                q_values,
                confidence_threshold=confidence_threshold,
            )
        elif condition == "active":
            kind, index, scores = choose_policy_partition_action(
                belief,
                goal,
                likelihoods,
                q_values,
                decision_error_penalty=failure_penalty,
                sensing_lookahead=sensing_lookahead,
                value_cache=value_cache,
                action_cache=action_cache,
            )
        else:
            raise ValueError(f"unknown evaluation condition: {condition}")
        ordered = sorted(scores.values())
        decision_margins.append(float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0)
        if kind == "commit":
            terminal_outcome = env.step(TERMINAL_PROBE)
            actions.append(TERMINAL_PROBE)
            outcomes.append(terminal_outcome)
            kinds.append("commit")
            committed = True
            break
        if kind == "sense":
            action = BEACON_ACTIONS[index]
            outcome = env.step(action)
            belief = update_beacon_belief(belief, likelihoods, index, outcome)
            senses += 1
        else:
            action = index
            outcome = env.step(action)
            belief = update_belief(belief, transition_joint, action, outcome)
            moves += 1
        actions.append(action)
        outcomes.append(outcome)
        kinds.append(kind)
        beliefs.append(belief.tolist())
        true_sites.append(_true_site(env))

    if not committed:
        terminal_outcome = env.step(TERMINAL_PROBE)
        actions.append(TERMINAL_PROBE)
        outcomes.append(terminal_outcome)
        kinds.append("forced-commit")
    success = terminal_outcome == goal
    return {
        "source": source,
        "goal": goal,
        "condition": condition,
        "success": bool(success),
        "senses": senses,
        "moves": moves,
        "total_interventions": len(actions),
        "commit_confidence": float(belief[goal]),
        "initial_entropy": float(entropy(np.asarray(beliefs[0]))),
        "commit_entropy": float(entropy(belief)),
        "actions": actions,
        "outcomes": outcomes,
        "kinds": kinds,
        "beliefs": beliefs,
        "true_sites": true_sites,
        "terminal_outcome": terminal_outcome,
        "mean_decision_margin": float(np.mean(decision_margins)) if decision_margins else 0.0,
    }


def additive_matrix_r2(matrix: np.ndarray) -> float:
    """Variance explained by source and goal main effects in a pair matrix."""
    matrix = np.asarray(matrix, dtype=float)
    overall = float(matrix.mean())
    fitted = matrix.mean(axis=1, keepdims=True) + matrix.mean(axis=0, keepdims=True) - overall
    residual = float(np.sum((matrix - fitted) ** 2))
    total = float(np.sum((matrix - overall) ** 2))
    return 1.0 - residual / total if total > 1e-15 else 1.0


def off_diagonal_correlation(first: np.ndarray, second: np.ndarray) -> float:
    mask = ~np.eye(len(first), dtype=bool)
    x = np.asarray(first, dtype=float)[mask]
    y = np.asarray(second, dtype=float)[mask]
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _restricted_matrix(
    records: list[dict[str, object]],
    key: str,
    failure_cost: float,
    *,
    censor_failures: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate pairwise costs, optionally censoring failed trials.

    Movement and total intervention matrices are restricted hitting costs, so a
    failed trial receives the relevant deadline. Sensing is instead a resource
    decomposition: it reports the beacon actions actually taken even when the
    eventual commit is wrong or times out.
    """
    matrix = np.zeros((N_SITES, N_SITES), dtype=float)
    success = np.zeros_like(matrix)
    for source in range(N_SITES):
        for goal in range(N_SITES):
            rows = [r for r in records if r["source"] == source and r["goal"] == goal]
            values = [
                float(r[key]) if r["success"] or not censor_failures else failure_cost
                for r in rows
            ]
            matrix[source, goal] = float(np.mean(values))
            success[source, goal] = float(np.mean([r["success"] for r in rows]))
    np.fill_diagonal(matrix, 0.0)
    return matrix, success


def evaluate_condition(
    *,
    condition: str,
    environment: str,
    likelihoods: np.ndarray,
    transition_joint: np.ndarray,
    q_values: np.ndarray,
    episodes_per_pair: int,
    max_interventions: int,
    max_movement_cost: int,
    failure_penalty: float,
    confidence_threshold: float,
    sensing_lookahead: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    env = QuantumEnvironment(environment=environment, weak_q=DIAGONAL_SUCCESS, seed=seed)
    records = []
    value_cache: dict[tuple[int, int, tuple[float, ...]], float] = {}
    action_cache: dict[
        tuple[int, tuple[float, ...]], tuple[str, int, dict[str, float]]
    ] = {}
    for source in range(N_SITES):
        for goal in range(N_SITES):
            for _ in range(episodes_per_pair):
                records.append(
                    run_episode(
                        env=env,
                        source=source,
                        goal=goal,
                        condition=condition,
                        likelihoods=likelihoods,
                        transition_joint=transition_joint,
                        q_values=q_values,
                        max_interventions=max_interventions,
                        failure_penalty=failure_penalty,
                        confidence_threshold=confidence_threshold,
                        sensing_lookahead=sensing_lookahead,
                        value_cache=value_cache,
                        action_cache=action_cache,
                    )
                )
    movement, success = _restricted_matrix(records, "moves", float(max_movement_cost))
    total, _ = _restricted_matrix(records, "total_interventions", float(max_interventions))
    sensing, _ = _restricted_matrix(
        records,
        "senses",
        float(max_interventions),
        censor_failures=False,
    )
    return {"movement": movement, "total": total, "sensing": sensing, "success": success}, records


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def summarize_condition(
    *,
    condition: str,
    seed: int,
    matrices: dict[str, np.ndarray],
    records: list[dict[str, object]],
    beacon_mae: float,
    transition_error: float,
) -> ActiveSummary:
    movement_metrics = geometry_metrics(matrices["movement"])
    total_metrics = geometry_metrics(matrices["total"])
    exact = exact_movement_costs(DIAGONAL_SUCCESS)
    movement_symmetric = (matrices["movement"] + matrices["movement"].T) / 2
    sensing_symmetric = (matrices["sensing"] + matrices["sensing"].T) / 2
    return ActiveSummary(
        condition=condition,
        seed=seed,
        all_pairs_success=float(matrices["success"].mean()),
        reset_success=float(matrices["success"][4].mean()),
        mean_senses=float(np.mean([r["senses"] for r in records])),
        mean_moves=float(np.mean([r["moves"] for r in records])),
        mean_total_interventions=float(np.mean([r["total_interventions"] for r in records])),
        mean_commit_confidence=float(np.mean([r["commit_confidence"] for r in records])),
        mean_initial_entropy=float(np.mean([r["initial_entropy"] for r in records])),
        mean_commit_entropy=float(np.mean([r["commit_entropy"] for r in records])),
        beacon_mae=beacon_mae,
        transition_tv_error=transition_error,
        movement_exact_correlation=distance_correlation(movement_symmetric, exact),
        movement_stress_1d=movement_metrics["stress_1d"],
        movement_stress_2d=movement_metrics["stress_2d"],
        movement_stress_3d=movement_metrics["stress_3d"],
        movement_procrustes_r2=movement_metrics["coordinate_procrustes_r2"],
        total_stress_2d=total_metrics["stress_2d"],
        total_stress_3d=total_metrics["stress_3d"],
        total_procrustes_r2=total_metrics["coordinate_procrustes_r2"],
        sensing_distance_correlation=off_diagonal_correlation(sensing_symmetric, exact),
        sensing_additive_r2=additive_matrix_r2(matrices["sensing"]),
    )


def run(args: argparse.Namespace) -> None:
    output = _workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    matrices_dir = output / "matrices"
    matrices_dir.mkdir(exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    saved_trajectories: list[dict[str, object]] = []
    conditions = (
        ("active", "qudit-grid-3x3-reversible-beacons", "learned"),
        ("active-exact-sensors", "qudit-grid-3x3-reversible-beacons", "exact"),
        ("fixed-12", "qudit-grid-3x3-reversible-beacons", "learned"),
        ("entropy", "qudit-grid-3x3-reversible-beacons", "learned"),
        ("active-null", "qudit-grid-3x3-reversible-null-beacons", "null"),
        ("oracle", "qudit-grid-3x3-reversible-beacons", "exact"),
    )

    for seed in args.seeds:
        informative = calibrate_beacons(
            environment="qudit-grid-3x3-reversible-beacons",
            trials_per_site_beacon=args.beacon_trials,
            seed=seed + 40_000,
        )
        null = calibrate_beacons(
            environment="qudit-grid-3x3-reversible-null-beacons",
            trials_per_site_beacon=args.beacon_trials,
            seed=seed + 50_000,
        )
        exact = exact_beacon_model("qudit-grid-3x3-reversible-beacons")
        transition_joint = learned_transition_model(
            trials_per_source_action=args.transition_trials,
            seed=seed + 60_000,
            environment="qudit-grid-3x3-reversible-beacons",
        )
        transition_error = joint_model_tv_error(
            transition_joint, "qudit-grid-3x3-reversible-beacons"
        )
        _, q_values = planning_values(transition_joint)
        calibration_rows.extend(
            {
                "seed": seed,
                "environment": name,
                "beacon": beacon,
                "site": site,
                "learned_p_one": float(model[beacon, site, 1]),
                "exact_p_one": float(exact_beacon_model(name)[beacon, site, 1]),
            }
            for name, model in (
                ("qudit-grid-3x3-reversible-beacons", informative),
                ("qudit-grid-3x3-reversible-null-beacons", null),
            )
            for beacon in range(4)
            for site in range(N_SITES)
        )

        for condition, environment, model_kind in conditions:
            print(f"[seed {seed}] evaluating {condition}", flush=True)
            likelihoods = {"learned": informative, "exact": exact, "null": null}[model_kind]
            controller_condition = "active" if condition in ("active", "active-exact-sensors", "active-null") else condition
            matrices, records = evaluate_condition(
                condition=controller_condition,
                environment=environment,
                likelihoods=likelihoods,
                transition_joint=transition_joint,
                q_values=q_values,
                episodes_per_pair=args.pair_episodes,
                max_interventions=args.max_interventions,
                max_movement_cost=args.max_movement_cost,
                failure_penalty=args.failure_penalty,
                confidence_threshold=args.confidence_threshold,
                sensing_lookahead=args.sensing_lookahead,
                seed=seed + 70_000 + len(summary_rows),
            )
            # Restore the public condition name after using the shared active controller.
            for record in records:
                record["condition"] = condition
                record["seed"] = seed
            summary = summarize_condition(
                condition=condition,
                seed=seed,
                matrices=matrices,
                records=records,
                beacon_mae=beacon_model_mae(likelihoods, environment),
                transition_error=transition_error,
            )
            summary_rows.append(asdict(summary))
            for matrix_name, matrix in matrices.items():
                _save_matrix(matrices_dir / f"{matrix_name}__{condition}__seed{seed}.csv", matrix)
            for goal in (0, 2, 6, 8):
                saved_trajectories.append(
                    next(
                        record
                        for record in records
                        if record["source"] == 4 and record["goal"] == goal
                    )
                )

    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "beacon_calibration.csv", calibration_rows)
    (output / "trajectories.json").write_text(json.dumps(saved_trajectories, indent=2))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "equal-cost active predictive atlas",
        "unit_costs": {"sense": SENSE_COST, "move": MOVE_COST, "commit": COMMIT_COST},
        "matrix_contract": {
            "movement": "failures receive max_movement_cost",
            "total": "failures receive max_interventions",
            "sensing": "actual beacon actions, including failed episodes",
            "success": "empirical terminal-probe accuracy",
        },
        "beacon_trials_per_site_action": args.beacon_trials,
        "transition_trials_per_source_action": args.transition_trials,
        "pair_episodes": args.pair_episodes,
        "max_interventions": args.max_interventions,
        "max_movement_cost": args.max_movement_cost,
        "failure_penalty": args.failure_penalty,
        "confidence_threshold": args.confidence_threshold,
        "sensing_lookahead": args.sensing_lookahead,
        "seeds": args.seeds,
        "online_received": [
            "goal landmark",
            "chosen intervention and observed outcome",
            "learned beacon likelihood table",
            "learned outcome-conditioned movement table",
        ],
        "withheld_online": [
            "current landmark",
            "density matrix and Kraus operators",
            "exact beacon fields and transition kernel",
            "concealed coordinates",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary_rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/active-atlas")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--beacon-trials", type=int, default=200)
    parser.add_argument("--transition-trials", type=int, default=100)
    parser.add_argument("--pair-episodes", type=int, default=100)
    parser.add_argument("--max-interventions", type=int, default=60)
    parser.add_argument("--max-movement-cost", type=int, default=12)
    parser.add_argument("--failure-penalty", type=float, default=100.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--sensing-lookahead", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if min(args.beacon_trials, args.transition_trials, args.pair_episodes) < 1:
        raise ValueError("all survey and evaluation counts must be positive")
    if args.max_interventions < 2 or args.max_movement_cost < 1:
        raise ValueError("intervention and movement limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
