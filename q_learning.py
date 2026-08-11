"""
q_learning.py
=============

Multi-goal finite-history tabular Q-learning compatible with
`quantum_rl_common.py`.

This is the simplest baseline in the current architecture.

Observable state
----------------
The agent is allowed to know only:

    * which goal it is currently pursuing;
    * its progress through that goal;
    * its own recent action/outcome history.

So the tabular state is

    s_t = (goal_id, progress, last L (action,outcome) pairs).

The agent NEVER receives the hidden quantum state, Kraus operators, or Born
probabilities.

Run
---
    python q_learning.py

Custom goals:
    python q_learning.py \
        --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0" \
        --episodes 30000

Save evaluation data:
    python q_learning.py --csv q_eval.csv

This backend implements the same protocol as the GRU and future backends:

    reset_episode(...)
    act(...)
    observe(...)
    end_episode(...)
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from quantum_rl_common import (
    common_arg_parser,
    evaluate_agent,
    make_environment,
    make_goals,
    print_evaluation_summary,
    save_results_csv,
    train_agent,
)


class TabularQAgent:
    """
    Multi-goal finite-history tabular Q-learning.

    Q(s,a) estimates expected discounted future reward after choosing
    measurement a in observable state s.
    """

    name = "tabular-q"

    def __init__(
        self,
        *,
        n_actions: int,
        history_length: int = 4,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int = 0,
    ):
        self.n_actions = int(n_actions)
        self.history_length = int(history_length)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.rng = np.random.default_rng(seed)

        # Each entry stores [Q(s,0), ..., Q(s,n_actions-1)].
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self.history = deque(maxlen=self.history_length)
        self._last_state = None

    def _state(
        self,
        goal_id: int,
        progress: int,
    ):
        """
        Construct the agent's observable state.

        No hidden quantum information appears here.
        """
        return (
            int(goal_id),
            int(progress),
            tuple(self.history),
        )

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------

    def reset_episode(
        self,
        goal_id: int,
        goal_length: int,
        training: bool,
    ) -> None:
        self.history.clear()
        self._last_state = None

    def act(
        self,
        goal_id: int,
        progress: int,
        training: bool,
    ) -> int:
        state = self._state(goal_id, progress)
        self._last_state = state

        # epsilon-greedy exploration
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q = self.Q[state]

        # Random tie-breaking.
        best = np.flatnonzero(np.isclose(q, q.max()))
        return int(self.rng.choice(best))

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
        if self._last_state is None:
            raise RuntimeError("observe() called before act()")

        # The newly observed action/outcome pair becomes part of history.
        self.history.append((int(action), int(outcome)))

        next_state = self._state(goal_id, next_progress)

        if not training:
            return

        old_q = self.Q[self._last_state][action]

        if done:
            target = float(reward)
        else:
            target = (
                float(reward)
                + self.gamma * np.max(self.Q[next_state])
            )

        # Standard tabular Q-learning update:
        #
        # Q <- Q + alpha [ target - Q ]
        self.Q[self._last_state][action] += (
            self.alpha * (target - old_q)
        )

    def end_episode(self, training: bool) -> None:
        if training:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay,
            )


def main() -> None:
    parser = common_arg_parser(
        "Multi-goal finite-history tabular Q-learning"
    )

    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)

    args = parser.parse_args()

    goals = make_goals(args)

    train_env = make_environment(args, seed_offset=0)
    eval_env = make_environment(args, seed_offset=1_000_000)

    agent = TabularQAgent(
        n_actions=train_env.n_actions,
        history_length=args.history_length,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed + 12345,
    )

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

    if args.csv:
        save_results_csv(
            args.csv,
            agent.name,
            "evaluation",
            results,
        )
        print(f"saved evaluation results to {args.csv}")


if __name__ == "__main__":
    main()
