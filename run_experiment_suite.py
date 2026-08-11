"""Run reproducible cross-environment, cross-backend experiments.

The runner intentionally writes tidy data before any plotting.  Neural runs use
the documented ``qbist_spacetime`` Conda environment; the resulting CSV bundle
can then be plotted by a lightweight Python with Matplotlib installed.

Example
-------
conda run -n qbist_spacetime python run_experiment_suite.py \
    --episodes 1000 --eval-episodes 100 --seeds 0,1 \
    --output results/standard
python plot_results.py results/standard
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare_backends import make_agent
from goal_geometry import collect_geometry_diagnostics, save_geometry_diagnostics
from quantum_environments import DEFAULT_GOALS_BY_ENVIRONMENT, QuantumEnvironment
from quantum_rl_common import (
    EpisodeResult,
    evaluate_agent,
    parse_goals,
    summarize_by_goal,
    summarize_results,
    train_agent,
    validate_goals,
)


DEFAULT_SCENARIOS = ",".join(
    (
        "qubit-zx-weak:one",
        "qubit-zx-weak:plus",
        "qubit-pauli:plus-i",
        "qubit-unsharp:mixed",
        "qubit-pauli-sic:mixed",
        "qutrit-mub:two",
    )
)


def _workspace_path(path: str) -> Path:
    """Resolve an output path and enforce the project's write boundary."""
    workspace = Path(__file__).resolve().parent
    candidate = (
        (workspace / path).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"output must stay inside {workspace}")
    return candidate


def parse_scenarios(text: str) -> list[tuple[str, str | None]]:
    scenarios: list[tuple[str, str | None]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        environment, separator, state = token.partition(":")
        scenarios.append((environment, state if separator else None))
    if not scenarios:
        raise ValueError("at least one scenario is required")
    return scenarios


def _agent_args(backend: str, seed: int, cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend=backend,
        history_length=cli.history_length,
        alpha=cli.alpha,
        gamma=cli.gamma,
        epsilon_start=cli.epsilon_start,
        epsilon_end=cli.epsilon_end,
        epsilon_decay=cli.epsilon_decay,
        interaction_embedding_dim=cli.interaction_embedding_dim,
        hidden_dim=cli.hidden_dim,
        goal_dim=cli.goal_dim,
        learning_rate=cli.learning_rate,
        prediction_weight=cli.prediction_weight,
        cost_weight=cli.cost_weight,
        geometry_weight=cli.geometry_weight,
        geometry_scale=cli.geometry_scale,
        policy_temperature=cli.policy_temperature,
        target_tau=cli.target_tau,
        grad_clip=cli.grad_clip,
        seed=seed,
        device=cli.device,
    )


def _episode_rows(
    run_id: str,
    environment: str,
    initial_state: str,
    backend: str,
    seed: int,
    results: list[EpisodeResult],
) -> list[dict[str, object]]:
    return [
        {
            "run_id": run_id,
            "environment": environment,
            "initial_state": initial_state,
            "backend": backend,
            "seed": seed,
            "episode": episode,
            **asdict(result),
        }
        for episode, result in enumerate(results, start=1)
    ]


def _summary_rows(
    run_id: str,
    environment: str,
    initial_state: str,
    backend: str,
    seed: int,
    phase: str,
    goals,
    results: list[EpisodeResult],
) -> list[dict[str, object]]:
    by_goal = summarize_by_goal(results, goals)
    rows = []
    for goal_id, goal in enumerate(goals):
        rows.append(
            {
                "run_id": run_id,
                "environment": environment,
                "initial_state": initial_state,
                "backend": backend,
                "seed": seed,
                "phase": phase,
                "goal_id": goal_id,
                "goal_name": goal.name,
                "episodes": sum(r.goal_id == goal_id for r in results),
                **by_goal[goal.name],
            }
        )
    rows.append(
        {
            "run_id": run_id,
            "environment": environment,
            "initial_state": initial_state,
            "backend": backend,
            "seed": seed,
            "phase": phase,
            "goal_id": -1,
            "goal_name": "OVERALL",
            "episodes": len(results),
            **summarize_results(results),
        }
    )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--backends", default="tabular,gru,multi-gru")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--geometry-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--output", default="results/standard")
    parser.add_argument("--weak-q", type=float, default=0.80)
    parser.add_argument("--max-goals", type=int, default=0, help="0 keeps every catalog goal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--interaction-embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--goal-dim", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prediction-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--geometry-scale", type=float, default=2.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--target-tau", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = _workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    geometry_root = output / "geometry"
    models_root = output / "models"
    geometry_root.mkdir(exist_ok=True)
    models_root.mkdir(exist_ok=True)

    scenarios = parse_scenarios(args.scenarios)
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    invalid_backends = set(backends) - {"tabular", "gru", "multi-gru"}
    if invalid_backends:
        raise ValueError(f"unknown backends: {sorted(invalid_backends)}")
    seeds = [int(value) for value in args.seeds.split(",")]
    training_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    run_manifest = []

    total_runs = len(scenarios) * len(backends) * len(seeds)
    run_number = 0
    for environment_name, initial_state in scenarios:
        goals = parse_goals(DEFAULT_GOALS_BY_ENVIRONMENT[environment_name])
        if args.max_goals > 0:
            goals = goals[: args.max_goals]
        for backend in backends:
            for seed in seeds:
                run_number += 1
                train_env = QuantumEnvironment(
                    environment=environment_name,
                    initial_state=initial_state,
                    weak_q=args.weak_q,
                    seed=seed,
                )
                eval_env = QuantumEnvironment(
                    environment=environment_name,
                    initial_state=initial_state,
                    weak_q=args.weak_q,
                    seed=seed + 1_000_000,
                )
                validate_goals(goals, train_env)
                selected_state = train_env.initial_state
                scenario_id = f"{environment_name}__{selected_state}"
                run_id = f"{scenario_id}__{backend}__seed{seed}"
                print(f"[{run_number}/{total_runs}] {run_id}", flush=True)
                agent = make_agent(_agent_args(backend, seed, args), train_env, goals)
                train_results = train_agent(
                    train_env,
                    goals,
                    agent,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=seed + 9_999,
                    log_every=args.log_every,
                )
                eval_results = evaluate_agent(
                    eval_env,
                    goals,
                    agent,
                    episodes_per_goal=args.eval_episodes,
                    max_steps=args.max_steps,
                )
                training_rows.extend(
                    _episode_rows(
                        run_id, environment_name, selected_state, backend, seed, train_results
                    )
                )
                evaluation_rows.extend(
                    _episode_rows(
                        run_id, environment_name, selected_state, backend, seed, eval_results
                    )
                )
                summary_rows.extend(
                    _summary_rows(
                        run_id,
                        environment_name,
                        selected_state,
                        backend,
                        seed,
                        "training",
                        goals,
                        train_results,
                    )
                )
                summary_rows.extend(
                    _summary_rows(
                        run_id,
                        environment_name,
                        selected_state,
                        backend,
                        seed,
                        "evaluation",
                        goals,
                        eval_results,
                    )
                )
                run_record: dict[str, object] = {
                    "run_id": run_id,
                    "environment": environment_name,
                    "initial_state": selected_state,
                    "backend": backend,
                    "seed": seed,
                    "goals": [goal.name for goal in goals],
                }
                if backend in {"gru", "multi-gru"}:
                    import torch

                    torch.save(agent.network.state_dict(), models_root / f"{run_id}.pt")
                if backend == "multi-gru":
                    geometry_env = QuantumEnvironment(
                        environment=environment_name,
                        initial_state=selected_state,
                        weak_q=args.weak_q,
                        seed=seed + 2_000_000,
                    )
                    diagnostics = collect_geometry_diagnostics(
                        agent,
                        geometry_env,
                        goals,
                        episodes_per_goal=args.geometry_episodes,
                        max_steps=args.max_steps,
                    )
                    geometry_directory = geometry_root / run_id
                    save_geometry_diagnostics(geometry_directory, diagnostics)
                    run_record["geometry_directory"] = str(geometry_directory.relative_to(output))
                    run_record["embedding_strategy_spearman"] = diagnostics[
                        "embedding_strategy_spearman"
                    ]
                    run_record["embedding_trajectory_spearman"] = diagnostics[
                        "embedding_trajectory_spearman"
                    ]
                run_manifest.append(run_record)
                overall = summarize_results(eval_results)
                print(
                    f"    success={overall['success_rate']:.3f} "
                    f"steps={overall['mean_steps_success']:.2f}",
                    flush=True,
                )
                # Checkpoint after every run so an interrupted long suite still
                # leaves valid, inspectable data.
                _write_rows(output / "training_episodes.csv", training_rows)
                _write_rows(output / "evaluation_episodes.csv", evaluation_rows)
                _write_rows(output / "summary.csv", summary_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "output": str(output),
        "runs": run_manifest,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved experiment bundle to {output}")


if __name__ == "__main__":
    main()
