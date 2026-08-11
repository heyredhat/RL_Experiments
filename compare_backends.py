"""
compare_backends.py
===================

Convenience entry point for running the SAME multi-goal experiment with a
swappable backend.

Examples
--------
Tabular baseline:
    python compare_backends.py --backend tabular

Multi-goal predictive GRU:
    python compare_backends.py --backend multi-gru

Use identical task arguments:
    python compare_backends.py --backend tabular \
        --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0" \
        --episodes 30000 --seed 7

    python compare_backends.py --backend multi-gru \
        --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0" \
        --episodes 30000 --seed 7

A future backend can be added by:
    1. implementing the AgentBackend protocol in quantum_rl_common.py
    2. importing it here
    3. adding one branch in make_agent()
"""

from __future__ import annotations

from quantum_rl_common import (
    common_arg_parser,
    evaluate_agent,
    make_environment,
    make_goals,
    print_evaluation_summary,
    save_results_csv,
    train_agent,
)

from q_learning import TabularQAgent
from gru_q_learning import GRUQAgent
from multi_goal_gru import MultiGoalGRUAgent, print_goal_geometry


def make_agent(args, env, goals):
    if args.backend == "tabular":
        return TabularQAgent(
            n_actions=env.n_actions,
            history_length=args.history_length,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed + 12345,
        )

    if args.backend == "gru":
        return GRUQAgent(
            n_actions=env.n_actions,
            n_outcomes=env.n_outcomes,
            goals=goals,
            interaction_embedding_dim=args.interaction_embedding_dim,
            hidden_dim=args.hidden_dim,
            goal_dim=args.goal_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_tau=args.target_tau,
            grad_clip=args.grad_clip,
            seed=args.seed + 12345,
            device=args.device,
        )

    if args.backend == "multi-gru":
        return MultiGoalGRUAgent(
            n_actions=env.n_actions,
            n_outcomes=env.n_outcomes,
            action_outcome_counts=env.action_outcome_counts,
            goals=goals,
            interaction_embedding_dim=args.interaction_embedding_dim,
            hidden_dim=args.hidden_dim,
            goal_dim=args.goal_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            prediction_weight=args.prediction_weight,
            cost_weight=args.cost_weight,
            geometry_weight=args.geometry_weight,
            geometry_scale=args.geometry_scale,
            policy_temperature=args.policy_temperature,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_tau=args.target_tau,
            grad_clip=args.grad_clip,
            seed=args.seed + 12345,
            device=args.device,
        )

    raise ValueError(f"unknown backend {args.backend}")


def main():
    parser = common_arg_parser("Compare swappable quantum-RL backends")

    parser.add_argument(
        "--backend",
        choices=("tabular", "gru", "multi-gru"),
        default="tabular",
    )

    # Shared RL options.
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)

    # Tabular-only options.
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)

    # GRU-only options.
    parser.add_argument(
        "--interaction-embedding-dim",
        type=int,
        default=8,
    )
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--goal-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prediction-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--geometry-scale", type=float, default=2.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--target-tau", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    goals = make_goals(args)

    train_env = make_environment(args, seed_offset=0)
    eval_env = make_environment(args, seed_offset=1_000_000)

    agent = make_agent(args, train_env, goals)

    print(f"backend: {agent.name}")
    if hasattr(agent, "device"):
        print(f"device: {agent.device}")

    train_agent(
        train_env,
        goals,
        agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed + 9999,
        log_every=args.log_every,
        step_penalty=args.step_penalty,
        checkpoint_reward=args.checkpoint_reward,
        completion_reward=args.completion_reward,
    )

    results = evaluate_agent(
        eval_env,
        goals,
        agent,
        episodes_per_goal=args.eval_episodes,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        checkpoint_reward=args.checkpoint_reward,
        completion_reward=args.completion_reward,
    )

    print_evaluation_summary(agent, goals, results, eval_env.action_names)

    if args.backend == "multi-gru":
        print_goal_geometry(agent, goals)

    if args.csv:
        save_results_csv(args.csv, agent.name, "evaluation", results)
        print(f"saved evaluation results to {args.csv}")


if __name__ == "__main__":
    main()
