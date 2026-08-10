"""
multi_goal_gru.py
=================

Goal-conditioned predictive GRU with two learned geometries:

1. GOAL--GOAL SIMILARITY
   Each goal g has a learned embedding e_g.  A geometry regularizer encourages

       ||e_g - e_h||

   to be small when the goal-conditioned policies for g and h are similar over
   histories encountered during training.

2. AGENT--GOAL REACHABILITY
   A cost head learns

       C(z, g, progress, a)

   = expected number of remaining interventions if action a is taken now and
     the agent behaves optimally afterward.

   The agent's learned distance to a goal is

       d(z,g) = min_a C(z,g,progress,a).

   This is a directional reachability quantity, not assumed to be a symmetric
   Euclidean metric.

The same GRU also learns:
    * Q(z,g,progress,a) for reward-based control
    * P(outcome | z,a) as an empirical model of measurement consequences

The hidden quantum state and Kraus operators are NEVER exposed to the agent.

Run:
    python multi_goal_gru.py

Save ordinary evaluation results:
    python multi_goal_gru.py --csv multi_gru_eval.csv

Save goal geometry diagnostics:
    python multi_goal_gru.py --geometry-prefix learned_geometry

This writes:
    learned_geometry_embeddings.csv
    learned_geometry_goal_distances.csv
    learned_geometry_initial_reachability.csv

Requires:
    pip install torch numpy
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

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


# ============================================================================
# Network
# ============================================================================

class MultiGoalNetwork(nn.Module):
    """
    One shared latent history state z, one learned embedding per goal, and four
    heads:

        Q head          -> reward value for each action
        cost head       -> intervention-count cost for each action
        outcome head    -> P(outcome | z, action)
        GRU update      -> z_{t+1} after observing (action,outcome)

    Physics prediction is intentionally goal-independent.
    """

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        n_goals: int,
        max_goal_length: int,
        interaction_embedding_dim: int = 8,
        hidden_dim: int = 48,
        goal_dim: int = 8,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.n_goals = n_goals
        self.max_goal_length = max_goal_length
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim

        self.action_embedding = nn.Embedding(
            n_actions,
            interaction_embedding_dim,
        )
        self.outcome_embedding = nn.Embedding(
            n_outcomes,
            interaction_embedding_dim,
        )

        self.history_gru = nn.GRUCell(
            input_size=2 * interaction_embedding_dim,
            hidden_size=hidden_dim,
        )

        # This is the geometric object whose pairwise Euclidean distances we
        # inspect after training.
        self.goal_embedding = nn.Embedding(n_goals, goal_dim)

        conditioned_dim = (
            hidden_dim
            + goal_dim
            + max_goal_length
            + 1
        )

        self.q_head = nn.Sequential(
            nn.Linear(conditioned_dim, 96),
            nn.ReLU(),
            nn.Linear(96, n_actions),
        )

        # The raw cost head is transformed in action_costs() as
        #
        #     C = 1 + softplus(raw)
        #
        # so an unfinished goal always requires at least one more intervention.
        self.cost_head = nn.Sequential(
            nn.Linear(conditioned_dim, 96),
            nn.ReLU(),
            nn.Linear(96, n_actions),
        )

        # Goal-independent empirical model of intervention outcomes.
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim + interaction_embedding_dim, 64),
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

    def goal_vector(
        self,
        goal_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        g = torch.tensor(goal_id, dtype=torch.long, device=device)
        return self.goal_embedding(g)

    def conditioned_features(
        self,
        z: torch.Tensor,
        goal_id: int,
        progress: int,
    ) -> torch.Tensor:
        return torch.cat(
            [
                z,
                self.goal_vector(goal_id, z.device),
                self.progress_vector(progress, z.device),
            ],
            dim=-1,
        )

    def q_values(
        self,
        z: torch.Tensor,
        goal_id: int,
        progress: int,
    ) -> torch.Tensor:
        x = self.conditioned_features(z, goal_id, progress)
        return self.q_head(x)

    def action_costs(
        self,
        z: torch.Tensor,
        goal_id: int,
        progress: int,
    ) -> torch.Tensor:
        """
        C(z,g,p,a): estimated remaining intervention count if a is taken now.
        """
        x = self.conditioned_features(z, goal_id, progress)
        return 1.0 + F.softplus(self.cost_head(x))

    def distance_to_goal(
        self,
        z: torch.Tensor,
        goal_id: int,
        progress: int,
    ) -> torch.Tensor:
        """
        d(z,g,p) = min_a C(z,g,p,a)
        """
        return torch.min(
            self.action_costs(z, goal_id, progress)
        )

    def outcome_logits(
        self,
        z: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        a = torch.tensor(action, dtype=torch.long, device=z.device)
        a_emb = self.action_embedding(a)
        return self.outcome_head(torch.cat([z, a_emb], dim=-1))

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

        return self.history_gru(x, z)


# ============================================================================
# Agent
# ============================================================================

class MultiGoalGRUAgent:
    name = "multi-goal-predictive-gru"

    def __init__(
        self,
        *,
        n_actions: int,
        n_outcomes: int,
        goals: tuple[GoalSpec, ...],
        interaction_embedding_dim: int = 8,
        hidden_dim: int = 48,
        goal_dim: int = 8,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        prediction_weight: float = 0.5,
        cost_weight: float = 0.5,
        geometry_weight: float = 0.05,
        geometry_scale: float = 2.0,
        policy_temperature: float = 1.0,
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
        self.n_goals = len(goals)
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes

        self.gamma = float(gamma)
        self.prediction_weight = float(prediction_weight)
        self.cost_weight = float(cost_weight)
        self.geometry_weight = float(geometry_weight)
        self.geometry_scale = float(geometry_scale)
        self.policy_temperature = float(policy_temperature)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)

        self.rng = np.random.default_rng(seed + 2026)

        self.network = MultiGoalNetwork(
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            n_goals=len(goals),
            max_goal_length=max(g.length for g in goals),
            interaction_embedding_dim=interaction_embedding_dim,
            hidden_dim=hidden_dim,
            goal_dim=goal_dim,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        self.z = None
        self.trajectory = []
        self.all_goal_progress = [0 for _ in goals]

        self.last_total_loss = float("nan")
        self.last_q_loss = float("nan")
        self.last_prediction_loss = float("nan")
        self.last_cost_loss = float("nan")
        self.last_geometry_loss = float("nan")

    # ---------------------------------------------------------------------
    # Interaction-time methods
    # ---------------------------------------------------------------------

    def reset_episode(
        self,
        goal_id: int,
        goal_length: int,
        training: bool,
    ) -> None:
        self.z = self.network.initial_memory(self.device)
        self.trajectory = []
        self.all_goal_progress = [0 for _ in self.goals]

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
            q = self.network.q_values(
                self.z,
                goal_id,
                progress,
            )

        best = torch.nonzero(
            torch.isclose(q, q.max()),
            as_tuple=False,
        ).flatten()

        return int(best[int(self.rng.integers(len(best)))].item())

    def _update_all_goal_progress(
        self,
        action: int,
        outcome: int,
    ) -> None:
        """
        Given the actual action/outcome history, compute how far that same
        history has progressed toward every known goal.

        This uses no hidden quantum information.
        """
        pair = (int(action), int(outcome))

        for i, spec in enumerate(self.goals):
            p = self.all_goal_progress[i]

            if p >= spec.length:
                continue

            if pair == spec.target[p]:
                self.all_goal_progress[i] += 1

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

        # Store the complete first-person transition.
        if training:
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

        self._update_all_goal_progress(action, outcome)

        with torch.no_grad():
            self.z = self.network.update_memory(
                self.z,
                action,
                outcome,
            )

    # ---------------------------------------------------------------------
    # Diagnostics available after / during training
    # ---------------------------------------------------------------------

    def predict_outcomes(self, action: int) -> np.ndarray:
        if self.z is None:
            raise RuntimeError("no active episode")

        with torch.no_grad():
            logits = self.network.outcome_logits(self.z, action)
            probs = torch.softmax(logits, dim=-1)

        return probs.detach().cpu().numpy()

    def distances_to_all_goals(self) -> np.ndarray:
        """
        Return the learned reachability vector

            [d(z,g_0), ..., d(z,g_{N-1})]

        using the progress toward each goal implied by the current history.
        """
        if self.z is None:
            raise RuntimeError("no active episode")

        values = []

        with torch.no_grad():
            for goal_id, progress in enumerate(self.all_goal_progress):
                if progress >= self.goals[goal_id].length:
                    values.append(0.0)
                else:
                    d = self.network.distance_to_goal(
                        self.z,
                        goal_id,
                        progress,
                    )
                    values.append(float(d.cpu()))

        return np.array(values, dtype=float)

    def goal_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            return (
                self.network.goal_embedding.weight
                .detach()
                .cpu()
                .numpy()
                .copy()
            )

    def goal_embedding_distance_matrix(self) -> np.ndarray:
        E = self.goal_embeddings()
        delta = E[:, None, :] - E[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=-1))

    # ---------------------------------------------------------------------
    # Training helpers
    # ---------------------------------------------------------------------

    def _target_states(self):
        z = self.target_network.initial_memory(self.device)
        states = [z]

        with torch.no_grad():
            for _, _, action, outcome, _, _, _ in self.trajectory:
                z = self.target_network.update_memory(
                    z,
                    action,
                    outcome,
                )
                states.append(z)

        return states

    @staticmethod
    def _js_divergence(
        p: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Jensen-Shannon divergence between two categorical distributions.
        """
        eps = 1e-8
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        m = 0.5 * (p + q)

        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))

        return 0.5 * (kl_pm + kl_qm)

    def _geometry_loss_at_state(
        self,
        z: torch.Tensor,
        progress_by_goal: list[int],
    ) -> torch.Tensor:
        """
        Encourage Euclidean goal-embedding distance to reflect strategy
        dissimilarity at this encountered latent state.

        For each pair of goals g,h:
            policy_g = softmax(Q(z,g)/temperature)
            policy_h = softmax(Q(z,h)/temperature)

        target behavioral distance:
            geometry_scale * sqrt(JS(policy_g, policy_h))

        embedding distance:
            ||e_g - e_h||

        The policy-derived target is detached.  Thus this auxiliary loss shapes
        the goal embedding geometry rather than trying to make policies imitate
        an arbitrary embedding.
        """
        if self.n_goals < 2 or self.geometry_weight == 0.0:
            return torch.tensor(0.0, device=self.device)

        policies = []

        for goal_id in range(self.n_goals):
            p = min(
                progress_by_goal[goal_id],
                self.goals[goal_id].length,
            )

            # A completed goal has no meaningful next-action policy.  For the
            # regularizer, clamp to the final valid progress representation.
            if p >= self.goals[goal_id].length:
                p = self.goals[goal_id].length

            q = self.network.q_values(z, goal_id, p)
            policy = torch.softmax(
                q / self.policy_temperature,
                dim=-1,
            )
            policies.append(policy)

        pair_losses = []

        for i in range(self.n_goals):
            for j in range(i + 1, self.n_goals):
                js = self._js_divergence(
                    policies[i],
                    policies[j],
                ).detach()

                behavioral_distance = (
                    self.geometry_scale * torch.sqrt(js + 1e-8)
                )

                ei = self.network.goal_vector(i, self.device)
                ej = self.network.goal_vector(j, self.device)

                embedding_distance = torch.linalg.vector_norm(ei - ej)

                pair_losses.append(
                    (embedding_distance - behavioral_distance) ** 2
                )

        return torch.stack(pair_losses).mean()

    def _advance_progress_vector(
        self,
        progress_by_goal: list[int],
        action: int,
        outcome: int,
    ) -> list[int]:
        new = list(progress_by_goal)
        pair = (int(action), int(outcome))

        for i, spec in enumerate(self.goals):
            p = new[i]

            if p < spec.length and pair == spec.target[p]:
                new[i] += 1

        return new

    def _train_on_trajectory(self):
        if not self.trajectory:
            nan = float("nan")
            return nan, nan, nan, nan, nan

        target_states = self._target_states()

        z = self.network.initial_memory(self.device)
        progress_by_goal = [0 for _ in self.goals]

        q_losses = []
        prediction_losses = []
        cost_losses = []
        geometry_losses = []

        for t, transition in enumerate(self.trajectory):
            (
                goal_id,
                progress,
                action,
                outcome,
                reward,
                done,
                next_progress,
            ) = transition

            # ---------------------------------------------------------------
            # Reward-value Q loss
            # ---------------------------------------------------------------
            q = self.network.q_values(z, goal_id, progress)
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
                        goal_id,
                        next_progress,
                    )
                    q_target = reward + self.gamma * torch.max(q_next)

            q_losses.append(F.mse_loss(q_chosen, q_target))

            # ---------------------------------------------------------------
            # Empirical outcome-model loss
            # ---------------------------------------------------------------
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

            # ---------------------------------------------------------------
            # Directed cost-to-go / reachability loss
            # ---------------------------------------------------------------
            #
            # C(z,g,p,a) = expected remaining number of interventions if a is
            # chosen now.  Therefore an action that immediately succeeds has
            # target cost 1.  Otherwise:
            #
            #     target = 1 + min_a' C_target(z',g,p',a')
            #
            action_costs = self.network.action_costs(
                z,
                goal_id,
                progress,
            )
            chosen_cost = action_costs[action]

            with torch.no_grad():
                if done:
                    cost_target = torch.tensor(
                        1.0,
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    next_costs = self.target_network.action_costs(
                        target_states[t + 1],
                        goal_id,
                        next_progress,
                    )
                    cost_target = 1.0 + torch.min(next_costs)

            cost_losses.append(
                F.smooth_l1_loss(chosen_cost, cost_target)
            )

            # ---------------------------------------------------------------
            # Goal-embedding geometry loss
            # ---------------------------------------------------------------
            geometry_losses.append(
                self._geometry_loss_at_state(
                    z,
                    progress_by_goal,
                )
            )

            # Update all-goal progress using this observed pair.
            progress_by_goal = self._advance_progress_vector(
                progress_by_goal,
                action,
                outcome,
            )

            # Finally incorporate the outcome into recurrent memory.
            z = self.network.update_memory(z, action, outcome)

        q_loss = torch.stack(q_losses).mean()
        prediction_loss = torch.stack(prediction_losses).mean()
        cost_loss = torch.stack(cost_losses).mean()
        geometry_loss = torch.stack(geometry_losses).mean()

        total_loss = (
            q_loss
            + self.prediction_weight * prediction_loss
            + self.cost_weight * cost_loss
            + self.geometry_weight * geometry_loss
        )

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
            float(cost_loss.detach().cpu()),
            float(geometry_loss.detach().cpu()),
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
            self.last_cost_loss,
            self.last_geometry_loss,
        ) = self._train_on_trajectory()

        self._soft_update_target()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )


# ============================================================================
# Geometry diagnostics
# ============================================================================

def print_goal_geometry(
    agent: MultiGoalGRUAgent,
    goals: tuple[GoalSpec, ...],
) -> None:
    """
    Print:
        * pairwise Euclidean distances between learned goal embeddings
        * learned distance from blank initial history to each goal
    """
    agent.reset_episode(
        goal_id=0,
        goal_length=goals[0].length,
        training=False,
    )

    D = agent.goal_embedding_distance_matrix()
    reach = agent.distances_to_all_goals()

    print()
    print("LEARNED GOAL-EMBEDDING DISTANCES")
    print("(smaller means learned strategies are intended to be more similar)")
    print()

    names = [g.name for g in goals]
    header = " " * 14 + "".join(f"{n[:10]:>12}" for n in names)
    print(header)

    for i, name in enumerate(names):
        row = f"{name[:12]:<14}" + "".join(
            f"{D[i, j]:12.3f}"
            for j in range(len(names))
        )
        print(row)

    print()
    print("LEARNED REACHABILITY FROM BLANK HISTORY")
    print("(approximate expected number of interventions remaining)")
    for spec, d in zip(goals, reach):
        print(f"  {spec.name:<18} d ~= {d:.3f}")


def save_geometry(
    prefix: str,
    agent: MultiGoalGRUAgent,
    goals: tuple[GoalSpec, ...],
) -> None:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = agent.goal_embeddings()
    distances = agent.goal_embedding_distance_matrix()

    # Evaluate reachability from a blank history.
    agent.reset_episode(
        goal_id=0,
        goal_length=goals[0].length,
        training=False,
    )
    reach = agent.distances_to_all_goals()

    emb_path = Path(str(prefix_path) + "_embeddings.csv")
    dist_path = Path(str(prefix_path) + "_goal_distances.csv")
    reach_path = Path(str(prefix_path) + "_initial_reachability.csv")

    with emb_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["goal_id", "goal_name"]
            + [f"e{i}" for i in range(embeddings.shape[1])]
        )
        for i, spec in enumerate(goals):
            writer.writerow(
                [i, spec.name] + embeddings[i].tolist()
            )

    with dist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["goal"] + [g.name for g in goals])
        for i, spec in enumerate(goals):
            writer.writerow(
                [spec.name] + distances[i].tolist()
            )

    with reach_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["goal_id", "goal_name", "distance_from_blank_history"]
        )
        for i, spec in enumerate(goals):
            writer.writerow([i, spec.name, reach[i]])

    print(f"saved {emb_path}")
    print(f"saved {dist_path}")
    print(f"saved {reach_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = common_arg_parser(
        "Multi-goal predictive GRU with learned goal geometry"
    )

    parser.add_argument(
        "--interaction-embedding-dim",
        type=int,
        default=8,
    )
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--goal-dim", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--prediction-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--geometry-scale", type=float, default=2.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)

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

    parser.add_argument(
        "--geometry-prefix",
        type=str,
        default=None,
        help="Optional prefix for learned-geometry CSV files.",
    )

    args = parser.parse_args()

    goals = make_goals(args)

    train_env = make_environment(args, seed_offset=0)
    eval_env = make_environment(args, seed_offset=1_000_000)

    agent = MultiGoalGRUAgent(
        n_actions=train_env.n_actions,
        n_outcomes=train_env.n_outcomes,
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

    print_evaluation_summary(agent, goals, results)

    print(
        "last losses: "
        f"total={agent.last_total_loss:.4f}, "
        f"Q={agent.last_q_loss:.4f}, "
        f"prediction={agent.last_prediction_loss:.4f}, "
        f"cost={agent.last_cost_loss:.4f}, "
        f"geometry={agent.last_geometry_loss:.4f}"
    )

    print_goal_geometry(agent, goals)

    if args.csv:
        save_results_csv(args.csv, agent.name, "evaluation", results)
        print(f"saved evaluation results to {args.csv}")

    if args.geometry_prefix:
        save_geometry(
            args.geometry_prefix,
            agent,
            goals,
        )


if __name__ == "__main__":
    main()
