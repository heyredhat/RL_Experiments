"""
predictive_gru_q_learning.py
============================

Single- or multi-goal GRU Q-learning with an auxiliary outcome-prediction model.

NEW IDEA RELATIVE TO gru_q_learning.py
--------------------------------------
The recurrent state z_t is trained for two jobs:

1. CONTROL:
       Q(z_t, progress, action)
   should identify useful measurements.

2. PREDICTION:
       P(outcome | z_t, action)
   should predict what classical result will occur if the agent performs a
   contemplated measurement.

The prediction head never sees rho or Kraus operators.  It learns only from
tuples actually experienced by the agent:

    (history summary z_t, chosen action a_t, observed outcome o_t)

This auxiliary task pressures the GRU memory to encode experimentally predictive
information about the hidden quantum system.

Run:
    python predictive_gru_q_learning.py

Example with a custom single goal:
    python predictive_gru_q_learning.py \
        --goals "Z0X0=0:0,1:0" \
        --episodes 30000

For several goals, use multi_goal_gru.py.

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
    common_arg_parser,
    evaluate_agent,
    make_environment,
    make_goals,
    print_evaluation_summary,
    save_results_csv,
    train_agent,
)


class PredictiveRecurrentQNetwork(nn.Module):
    """
    Shared GRU history encoder with:
        * a Q-value head
        * an outcome-prediction head

    The predictor answers a counterfactual-looking but purely empirical question:

        "Given my current memory z, if I choose measurement a, what outcome
         distribution should I expect?"

    For every action a it returns logits over outcomes.
    """

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        action_outcome_counts: tuple[int, ...] | None = None,
        max_goal_length: int,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        counts = action_outcome_counts or (n_outcomes,) * n_actions
        if len(counts) != n_actions:
            raise ValueError("action_outcome_counts must have one entry per action")
        mask = torch.arange(n_outcomes)[None, :] < torch.tensor(counts)[:, None]
        self.register_buffer("valid_outcome_mask", mask)
        self.max_goal_length = max_goal_length
        self.hidden_dim = hidden_dim

        self.action_embedding = nn.Embedding(n_actions, embedding_dim)
        self.outcome_embedding = nn.Embedding(n_outcomes, embedding_dim)

        self.gru = nn.GRUCell(
            input_size=2 * embedding_dim,
            hidden_size=hidden_dim,
        )

        # Goal progress is directly known from the agent's own record.
        q_input_dim = hidden_dim + max_goal_length + 1

        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

        # Prediction is goal-independent: physics should not change merely
        # because the agent is pursuing a different goal.
        #
        # Given z and a contemplated action embedding, return logits over
        # possible outcomes.
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_outcomes),
        )

    def initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

    def progress_vector(
        self,
        progress: int,
        device: torch.device,
    ) -> torch.Tensor:
        p = torch.zeros(self.max_goal_length + 1, device=device)
        p[int(progress)] = 1.0
        return p

    def q_values(
        self,
        z: torch.Tensor,
        progress: int,
    ) -> torch.Tensor:
        p = self.progress_vector(progress, z.device)
        return self.q_head(torch.cat([z, p], dim=-1))

    def outcome_logits(
        self,
        z: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        a = torch.tensor(action, dtype=torch.long, device=z.device)
        a_emb = self.action_embedding(a)
        logits = self.predictor(torch.cat([z, a_emb], dim=-1))
        return logits.masked_fill(~self.valid_outcome_mask[action], -1e9)

    def outcome_probabilities(
        self,
        z: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        return torch.softmax(self.outcome_logits(z, action), dim=-1)

    def update_memory(
        self,
        z: torch.Tensor,
        action: int,
        outcome: int,
    ) -> torch.Tensor:
        a = torch.tensor(action, dtype=torch.long, device=z.device)
        o = torch.tensor(outcome, dtype=torch.long, device=z.device)

        x = torch.cat(
            [
                self.action_embedding(a),
                self.outcome_embedding(o),
            ],
            dim=-1,
        )

        return self.gru(x, z)


class PredictiveGRUQAgent:
    name = "predictive-gru-q"

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        action_outcome_counts: tuple[int, ...] | None = None,
        max_goal_length: int,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        prediction_weight: float = 0.5,
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

        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.gamma = float(gamma)
        self.prediction_weight = float(prediction_weight)
        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.rng = np.random.default_rng(seed + 777)

        self.network = PredictiveRecurrentQNetwork(
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            action_outcome_counts=action_outcome_counts,
            max_goal_length=max_goal_length,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        self.z = None
        self.trajectory = []
        self.last_total_loss = float("nan")
        self.last_q_loss = float("nan")
        self.last_prediction_loss = float("nan")

    def reset_episode(
        self,
        goal_id: int,
        goal_length: int,
        training: bool,
    ) -> None:
        self.z = self.network.initial_memory(self.device)
        self.trajectory = []

    def act(
        self,
        goal_id: int,
        progress: int,
        training: bool,
    ) -> int:
        if self.z is None:
            raise RuntimeError("reset_episode() must be called before act()")

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        with torch.no_grad():
            q = self.network.q_values(self.z, progress)

        best = torch.nonzero(
            torch.isclose(q, q.max()),
            as_tuple=False,
        ).flatten()

        return int(best[int(self.rng.integers(len(best)))].item())

    def predict_outcomes(self, action: int) -> np.ndarray:
        """
        Public diagnostic: return the agent's currently learned probability
        distribution P_hat(o | current history, action).

        This is learned entirely from experience.
        """
        if self.z is None:
            raise RuntimeError("no active episode")

        with torch.no_grad():
            p = self.network.outcome_probabilities(self.z, action)

        return p.detach().cpu().numpy()

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
            raise RuntimeError("observe() called before reset_episode()")

        if training:
            self.trajectory.append(
                (
                    int(progress),
                    int(action),
                    int(outcome),
                    float(reward),
                    bool(done),
                    int(next_progress),
                )
            )

        with torch.no_grad():
            self.z = self.network.update_memory(
                self.z,
                action,
                outcome,
            )

    def _target_states(self):
        z = self.target_network.initial_memory(self.device)
        states = [z]

        with torch.no_grad():
            for _, action, outcome, _, _, _ in self.trajectory:
                z = self.target_network.update_memory(z, action, outcome)
                states.append(z)

        return states

    def _train_on_trajectory(self):
        if not self.trajectory:
            return float("nan"), float("nan"), float("nan")

        target_states = self._target_states()

        z = self.network.initial_memory(self.device)

        q_losses = []
        prediction_losses = []

        for t, transition in enumerate(self.trajectory):
            (
                progress,
                action,
                outcome,
                reward,
                done,
                next_progress,
            ) = transition

            # ---------------------------------------------------------------
            # 1. CONTROL LOSS
            # ---------------------------------------------------------------
            q = self.network.q_values(z, progress)
            q_chosen = q[action]

            with torch.no_grad():
                if done:
                    q_target = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    q_next = self.target_network.q_values(
                        target_states[t + 1],
                        next_progress,
                    )
                    q_target = reward + self.gamma * torch.max(q_next)

            q_losses.append(F.mse_loss(q_chosen, q_target))

            # ---------------------------------------------------------------
            # 2. OUTCOME-PREDICTION LOSS
            # ---------------------------------------------------------------
            #
            # Before the outcome has been incorporated into memory, ask the
            # network to predict it from (z_t, action_t).
            logits = self.network.outcome_logits(z, action)

            target_outcome = torch.tensor(
                outcome,
                dtype=torch.long,
                device=self.device,
            )

            prediction_losses.append(
                F.cross_entropy(
                    logits.unsqueeze(0),
                    target_outcome.unsqueeze(0),
                )
            )

            # Now, and only now, incorporate the observed outcome into memory.
            z = self.network.update_memory(z, action, outcome)

        q_loss = torch.stack(q_losses).mean()
        prediction_loss = torch.stack(prediction_losses).mean()

        total_loss = q_loss + self.prediction_weight * prediction_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.grad_clip,
        )

        self.optimizer.step()

        return (
            float(total_loss.detach().cpu()),
            float(q_loss.detach().cpu()),
            float(prediction_loss.detach().cpu()),
        )

    def _soft_update_target(self):
        tau = self.target_tau

        with torch.no_grad():
            for tp, op in zip(
                self.target_network.parameters(),
                self.network.parameters(),
            ):
                tp.mul_(1.0 - tau)
                tp.add_(tau * op)

    def end_episode(self, training: bool) -> None:
        if not training:
            return

        (
            self.last_total_loss,
            self.last_q_loss,
            self.last_prediction_loss,
        ) = self._train_on_trajectory()

        self._soft_update_target()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )


def main():
    parser = common_arg_parser(
        "GRU Q-learning plus empirical outcome prediction"
    )

    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--prediction-weight", type=float, default=0.5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--target-tau", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    # This file intentionally isolates the "predictive GRU" idea from the
    # multi-goal idea.  Use multi_goal_gru.py for several goals.
    parser.set_defaults(goals="Z0=0:0")

    args = parser.parse_args()

    goals = make_goals(args)
    if len(goals) != 1:
        raise ValueError(
            "predictive_gru_q_learning.py is the single-goal predictive "
            "baseline. Pass exactly one goal, or use multi_goal_gru.py."
        )
    train_env = make_environment(args, seed_offset=0)
    eval_env = make_environment(args, seed_offset=1_000_000)

    agent = PredictiveGRUQAgent(
        n_actions=train_env.n_actions,
        n_outcomes=train_env.n_outcomes,
        action_outcome_counts=train_env.action_outcome_counts,
        max_goal_length=max(g.length for g in goals),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        prediction_weight=args.prediction_weight,
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

    print_evaluation_summary(agent, goals, results, eval_env.action_names)
    print(
        f"last training losses: total={agent.last_total_loss:.4f}, "
        f"Q={agent.last_q_loss:.4f}, "
        f"prediction={agent.last_prediction_loss:.4f}"
    )

    if args.csv:
        save_results_csv(args.csv, agent.name, "evaluation", results)
        print(f"saved evaluation results to {args.csv}")


if __name__ == "__main__":
    main()
