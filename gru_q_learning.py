"""
gru_q_learning.py
=================

Multi-goal recurrent Q-learning using a GRU, compatible with
`quantum_rl_common.py`.

This is the direct recurrent analogue of `q_learning.py`.

Unlike `predictive_gru_q_learning.py`, this file does NOT include an auxiliary
outcome-prediction loss.  Its purpose is to isolate the benefit of replacing a
literal finite history with a learned recurrent state.

Architecture
------------

Observed history:
    (a_0,o_0), (a_1,o_1), ...

is compressed into a learned recurrent memory

    z_{t+1} = GRU(z_t, embed(a_t,o_t)).

The current goal is represented by a learned embedding e_g.

The Q-network then computes

    Q(z_t, e_g, progress, action).

Thus one GRU history representation is shared across all goals, while the
goal-conditioned Q head can choose different strategies for different goals.

The agent NEVER receives rho, Kraus operators, or Born probabilities.

Run
---
    python gru_q_learning.py

Custom goals:
    python gru_q_learning.py \
        --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0" \
        --episodes 30000

Requires:
    pip install torch numpy
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_rl_common import (
    GoalSpec,
    common_arg_parser,
    evaluate_agent,
    make_environment,
    make_goals,
    print_evaluation_summary,
    save_results_csv,
    train_agent,
)


class RecurrentQNetwork(nn.Module):
    """
    Shared GRU history encoder plus goal-conditioned Q head.

    Inputs to Q:
        z_t          learned summary of intervention/outcome history
        e_g          learned embedding of current goal
        progress     known progress through the current goal
    """

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        n_goals: int,
        max_goal_length: int,
        interaction_embedding_dim: int = 8,
        hidden_dim: int = 32,
        goal_dim: int = 8,
    ):
        super().__init__()

        self.n_actions = int(n_actions)
        self.n_outcomes = int(n_outcomes)
        self.n_goals = int(n_goals)
        self.max_goal_length = int(max_goal_length)
        self.hidden_dim = int(hidden_dim)

        # Embeddings turn discrete action/outcome symbols into learned vectors.
        self.action_embedding = nn.Embedding(
            n_actions,
            interaction_embedding_dim,
        )
        self.outcome_embedding = nn.Embedding(
            n_outcomes,
            interaction_embedding_dim,
        )

        # Recurrent memory update.
        self.gru = nn.GRUCell(
            input_size=2 * interaction_embedding_dim,
            hidden_size=hidden_dim,
        )

        # Each known goal gets a learned vector.
        self.goal_embedding = nn.Embedding(
            n_goals,
            goal_dim,
        )

        q_input_dim = (
            hidden_dim
            + goal_dim
            + max_goal_length
            + 1
        )

        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def initial_memory(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(
            self.hidden_dim,
            device=device,
        )

    def progress_vector(
        self,
        progress: int,
        device: torch.device,
    ) -> torch.Tensor:
        p = torch.zeros(
            self.max_goal_length + 1,
            device=device,
        )
        p[int(progress)] = 1.0
        return p

    def goal_vector(
        self,
        goal_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        g = torch.tensor(
            goal_id,
            dtype=torch.long,
            device=device,
        )
        return self.goal_embedding(g)

    def q_values(
        self,
        z: torch.Tensor,
        goal_id: int,
        progress: int,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                z,
                self.goal_vector(goal_id, z.device),
                self.progress_vector(progress, z.device),
            ],
            dim=-1,
        )

        return self.q_head(features)

    def update_memory(
        self,
        z: torch.Tensor,
        action: int,
        outcome: int,
    ) -> torch.Tensor:
        a = torch.tensor(
            action,
            dtype=torch.long,
            device=z.device,
        )
        o = torch.tensor(
            outcome,
            dtype=torch.long,
            device=z.device,
        )

        x = torch.cat(
            [
                self.action_embedding(a),
                self.outcome_embedding(o),
            ],
            dim=-1,
        )

        return self.gru(x, z)


class GRUQAgent:
    """
    Multi-goal recurrent Q-learning / DRQN-style backend.

    Experience is collected one episode at a time. At episode end, the complete
    observed sequence is replayed through the GRU with gradients enabled.
    PyTorch then performs backpropagation through time.

    A slowly updated target network stabilizes the Bellman targets.
    """

    name = "gru-q"

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        goals: tuple[GoalSpec, ...],
        interaction_embedding_dim: int = 8,
        hidden_dim: int = 32,
        goal_dim: int = 8,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_tau: float = 0.02,
        grad_clip: float = 5.0,
        seed: int = 0,
        device: str = "auto",
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.goals = goals
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)

        self.rng = np.random.default_rng(seed + 999)

        self.network = RecurrentQNetwork(
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            n_goals=len(goals),
            max_goal_length=max(g.length for g in goals),
            interaction_embedding_dim=interaction_embedding_dim,
            hidden_dim=hidden_dim,
            goal_dim=goal_dim,
        ).to(self.device)

        self.target_network = copy.deepcopy(
            self.network
        ).to(self.device)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        self.z = None
        self.trajectory = []

        self.last_loss = float("nan")

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------

    def reset_episode(
        self,
        goal_id: int,
        goal_length: int,
        training: bool,
    ) -> None:
        self.z = self.network.initial_memory(
            self.device
        )
        self.trajectory = []

    def act(
        self,
        goal_id: int,
        progress: int,
        training: bool,
    ) -> int:
        if self.z is None:
            raise RuntimeError(
                "reset_episode() must be called before act()"
            )

        if training and self.rng.random() < self.epsilon:
            return int(
                self.rng.integers(self.n_actions)
            )

        with torch.no_grad():
            q = self.network.q_values(
                self.z,
                goal_id,
                progress,
            )

        best = torch.nonzero(
            torch.isclose(q, q.max()),
            as_tuple=False,
        ).flatten()

        index = int(
            self.rng.integers(len(best))
        )

        return int(best[index].item())

    def observe(
        self,
        goal_id: int,
        progress: int,
        action: int,
        outcome: int,
        reward: float,
        done: bool,
        next_progress: int,
        training: bool,
    ) -> None:
        if self.z is None:
            raise RuntimeError(
                "observe() called before reset_episode()"
            )

        if training:
            # Store only first-person information available to the agent.
            self.trajectory.append(
                (
                    int(goal_id),
                    int(progress),
                    int(action),
                    int(outcome),
                    float(reward),
                    bool(done),
                    int(next_progress),
                )
            )

        # Update online recurrent memory for the next action choice.
        # We do not retain a gradient graph during data collection; training
        # reconstructs the graph by replaying the trajectory.
        with torch.no_grad():
            self.z = self.network.update_memory(
                self.z,
                action,
                outcome,
            )

    def end_episode(
        self,
        training: bool,
    ) -> None:
        if not training:
            return

        self.last_loss = self._train_on_trajectory()
        self._soft_update_target()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _target_hidden_states(self):
        """
        Replay the observed history through the target GRU.

        states[t] is the target-network memory BEFORE transition t.
        """
        z = self.target_network.initial_memory(
            self.device
        )

        states = [z]

        with torch.no_grad():
            for (
                _goal_id,
                _progress,
                action,
                outcome,
                _reward,
                _done,
                _next_progress,
            ) in self.trajectory:
                z = self.target_network.update_memory(
                    z,
                    action,
                    outcome,
                )
                states.append(z)

        return states

    def _train_on_trajectory(
        self,
    ) -> float:
        if not self.trajectory:
            return float("nan")

        target_states = self._target_hidden_states()

        # Replay through the online network with gradients enabled.
        z = self.network.initial_memory(
            self.device
        )

        losses = []

        for t, transition in enumerate(
            self.trajectory
        ):
            (
                goal_id,
                progress,
                action,
                outcome,
                reward,
                done,
                next_progress,
            ) = transition

            # Current estimate Q(z_t,g,a_t).
            q = self.network.q_values(
                z,
                goal_id,
                progress,
            )
            q_chosen = q[action]

            # Construct z_{t+1}.  Do NOT detach: future losses should be able
            # to backpropagate through this recurrent transition.
            z_next = self.network.update_memory(
                z,
                action,
                outcome,
            )

            # Bellman target from the slowly moving target network.
            with torch.no_grad():
                if done:
                    target = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    q_next = self.target_network.q_values(
                        target_states[t + 1],
                        goal_id,
                        next_progress,
                    )

                    target = (
                        reward
                        + self.gamma * torch.max(q_next)
                    )

            losses.append(
                F.mse_loss(
                    q_chosen,
                    target,
                )
            )

            z = z_next

        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.grad_clip,
        )

        self.optimizer.step()

        return float(
            loss.detach().cpu()
        )

    def _soft_update_target(
        self,
    ) -> None:
        tau = self.target_tau

        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.network.parameters(),
            ):
                target_param.mul_(1.0 - tau)
                target_param.add_(
                    tau * online_param
                )

    # ------------------------------------------------------------------
    # Useful diagnostics
    # ------------------------------------------------------------------

    def goal_embeddings(
        self,
    ) -> np.ndarray:
        """
        Return the raw learned goal vectors.

        In this basic GRU baseline there is NO geometry regularizer, so these
        embeddings should not automatically be interpreted as a meaningful
        goal-distance geometry.  multi_goal_gru.py adds that extra structure.
        """
        with torch.no_grad():
            return (
                self.network.goal_embedding.weight
                .detach()
                .cpu()
                .numpy()
                .copy()
            )


def main() -> None:
    parser = common_arg_parser(
        "Multi-goal recurrent Q-learning with a GRU"
    )

    parser.add_argument(
        "--interaction-embedding-dim",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--goal-dim",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9995,
    )

    parser.add_argument(
        "--target-tau",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    goals = make_goals(args)

    train_env = make_environment(
        args,
        seed_offset=0,
    )
    eval_env = make_environment(
        args,
        seed_offset=1_000_000,
    )

    agent = GRUQAgent(
        n_actions=train_env.n_actions,
        n_outcomes=train_env.n_outcomes,
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

    print_evaluation_summary(
        agent,
        goals,
        results,
    )

    print(
        f"last training loss: "
        f"{agent.last_loss:.6f}"
    )

    if args.csv:
        save_results_csv(
            args.csv,
            agent.name,
            "evaluation",
            results,
        )
        print(
            f"saved evaluation results to "
            f"{args.csv}"
        )


if __name__ == "__main__":
    main()
