# Quantum Measurement Reinforcement Learning: Architecture Guide

This directory contains a small research scaffold for studying artificial agents
that learn to control a quantum system **without being given the quantum state,
Kraus operators, Born probabilities, Hamiltonian, or any other privileged
description of the system**.

The agent's interface is intentionally first-person and operational:

1. choose a measurement/intervention \(a_t\);
2. observe a classical outcome \(o_t\);
3. record the resulting reward / goal progress;
4. choose what to do next.

The quantum model exists only inside the simulated environment.

---

## 1. Files

### Current shared infrastructure

- `quantum_rl_common.py`
  - hidden quantum environment;
  - sequence-goal definitions;
  - multi-goal task parser;
  - backend protocol;
  - common training and evaluation loop.

### Baselines / agents

- `multi_goal_q_learning.py`
  - naive finite-history tabular Q-learning;
  - handles several goals;
  - baseline for comparing learned recurrent representations against literal
    finite memory.

- `predictive_gru_q_learning.py`
  - single-goal recurrent Q-learning;
  - GRU summarizes action/outcome history;
  - adds an auxiliary model
    \[
    \hat P(o_{t+1}\mid z_t,a_{t+1})
    \]
    that learns the empirical effect of contemplated interventions.

- `multi_goal_gru.py`
  - goal-conditioned GRU;
  - predicts outcomes;
  - learns Q-values for several goals;
  - learns a cost-to-go / reachability distance to each goal;
  - learns goal embeddings whose Euclidean geometry is regularized to reflect
    policy/strategy similarity.

### Convenience runner

- `compare_backends.py`
  - runs the same environment and goal set with either:
    - `--backend tabular`
    - `--backend multi-gru`
  - intended as the main place to add future backends.

### Legacy files from the first iteration

- `quantum_environment.py`
- `q_learning.py`
- `gru_q_learning.py`

Those remain useful as the simplest one-goal versions, but the newer files use
`quantum_rl_common.py`.

---

# 2. Physical environment

The default hidden system is one qubit.

Available interventions:

| action index | measurement |
|---|---|
| `0` | projective \(Z\) |
| `1` | projective \(X\) |
| `2` | weak \(Z\) |

Every measurement has classical outcomes `0` and `1`.

The environment secretly uses Kraus operators:

\[
p(o\mid \rho,a)
=
\operatorname{Tr}
K_o^{(a)}\rho K_o^{(a)\dagger},
\]

followed by

\[
\rho'
=
\frac{
K_o^{(a)}\rho K_o^{(a)\dagger}
}{
p(o\mid\rho,a)
}.
\]

But the public agent/environment interaction is only

```python
outcome = env.step(action)
```

An agent should **never** inspect:

```python
env._rho
env._kraus
```

Those variables are deliberately private.

---

# 3. Goals

A goal is an ordered sequence of desired action/outcome checkpoints.

For example

```text
Z0_X0 = 0:0,1:0
```

means:

1. at some point perform \(Z\) and obtain outcome \(0\);
2. later perform \(X\) and obtain outcome \(0\).

Other exploratory measurements may occur between checkpoints.

The command-line syntax for several goals is:

```bash
--goals "Z0=0:0;X0=1:0;Z0_X0=0:0,1:0"
```

A `GoalTracker` maintains the currently completed prefix.

This progress variable is allowed information: it is computable entirely from
the agent's own action/outcome record and the goal it is trying to achieve.

---

# 4. Common backend interface

Every agent backend implements:

```python
reset_episode(goal_id, goal_length, training)
act(goal_id, progress, training)
observe(
    goal_id,
    progress,
    action,
    outcome,
    reward,
    done,
    next_progress,
    training,
)
end_episode(training)
```

This is the important abstraction boundary.

The common runner knows nothing about whether the backend is:

- a Q-table;
- a GRU;
- a transformer;
- a learned world model;
- successor features;
- Monte-Carlo tree search;
- something else.

To add a new backend, implement these methods and add one constructor branch in
`compare_backends.py`.

---

# 5. Naive multi-goal Q-learning

File:

```text
multi_goal_q_learning.py
```

The observable state is

\[
s_t =
(
g,\,
j_t,\,
(a_{t-L},o_{t-L}),\ldots,(a_{t-1},o_{t-1})
),
\]

where:

- \(g\) is the current goal ID;
- \(j_t\) is current progress through that goal;
- \(L\) is a fixed history length.

The Q-table stores

\[
Q(s,a).
\]

After observing reward \(r_t\) and next observable state \(s_{t+1}\),

\[
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+
\gamma\max_a Q(s_{t+1},a)
-
Q(s_t,a_t)
\right].
\]

This is model-free. The agent never tries to predict quantum outcomes.

## Strength

It is transparent and easy to debug.

## Weakness

The choice of history length \(L\) is arbitrary. Two almost identical histories
are unrelated table entries. The state space grows rapidly.

---

# 6. GRU history representation

The recurrent agents replace literal finite history with a learned latent memory

\[
z_t\in\mathbb R^d.
\]

After the agent chooses \(a_t\) and sees \(o_t\),

\[
z_{t+1}
=
\operatorname{GRU}
\left(
z_t,\,
\operatorname{embed}(a_t,o_t)
\right).
\]

The GRU is a gated recurrent neural network. Its gates learn what old
information should be retained, modified, or forgotten.

Conceptually:

```text
history so far
      |
      v
     z_t
      |
 choose a_t
      |
 observe o_t
      |
      v
GRU(z_t, a_t, o_t)
      |
      v
    z_{t+1}
```

The desired interpretation is not that \(z_t\) *is* a density matrix.

Instead,

\[
z_t
\]

is whatever compressed representation of past experience is useful for
prediction and control.

One scientific question is whether the learned latent states become equivalent,
up to a nonlinear coordinate transformation, to an operational representation
of the hidden quantum state.

---

# 7. Predictive GRU: learning the effects of interventions

File:

```text
predictive_gru_q_learning.py
```

The GRU is trained simultaneously for control and prediction.

## Control head

\[
Q_\theta(z_t,j_t,a)
\]

estimates future reward.

## Outcome head

Before outcome \(o_t\) is incorporated into memory, the predictor computes

\[
\hat P_\theta(o\mid z_t,a_t).
\]

The observed outcome supplies a supervised training signal:

\[
L_{\rm pred}
=
-\log
\hat P_\theta(o_t\mid z_t,a_t).
\]

This is ordinary categorical cross-entropy.

The complete loss is approximately

\[
L
=
L_Q
+
\lambda_{\rm pred} L_{\rm pred}.
\]

This matters because reward alone may be a sparse signal. Outcome prediction
forces the recurrent state to remember facts about history that are useful for
predicting future experimental consequences.

Importantly, the predictor is **goal independent**. The physical consequences of
a measurement do not depend on which goal the agent currently wants.

The method

```python
agent.predict_outcomes(action)
```

returns the agent's currently learned probability distribution for a
contemplated action.

---

# 8. Multi-goal GRU

File:

```text
multi_goal_gru.py
```

The full network contains:

```text
                    action/outcome history
                            |
                            v
                           GRU
                            |
                            v
                           z_t
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
        outcome model    goal-conditioned  cost-to-go
        P(o | z,a)          Q values         C values
                              ^
                              |
                       learned goal e_g
```

Every goal \(g\) receives a learned vector

\[
e_g\in\mathbb R^k.
\]

The reward-value network uses

\[
Q(z_t,e_g,j_t,a).
\]

Therefore one recurrent history state can be compared against several possible
goals.

---

# 9. Two different geometries

The code intentionally distinguishes two notions that should not be conflated.

## 9.1 Goal--goal similarity

The learned goal vectors define Euclidean distances

\[
D_G(g,h)
=
\|e_g-e_h\|.
\]

The intended meaning is:

> goals are close when the strategies used to achieve them are similar.

At encountered history state \(z\), define soft goal-conditioned policies

\[
\pi_g(a\mid z)
=
\operatorname{softmax}
\left(
\frac{Q(z,g,a)}{T}
\right).
\]

For two goals, policy difference is measured with Jensen--Shannon divergence:

\[
D_{\rm behavior}(g,h;z)
=
D_{\rm JS}
\left(
\pi_g(\cdot\mid z),
\pi_h(\cdot\mid z)
\right).
\]

The geometry regularizer trains

\[
\|e_g-e_h\|
\]

toward a scaled version of

\[
\sqrt{D_{\rm JS}}.
\]

Averaged over histories encountered during training, this encourages the
embedding geometry to reflect similarity of strategies rather than superficial
similarity of goal labels.

This is an experimental modeling choice, not a theorem. Alternative behavioral
distances should be explored.

---

## 9.2 Agent--goal reachability

The second notion is not Euclidean distance between two goal vectors.

The network learns an action cost

\[
C(z,g,j,a),
\]

interpreted as approximately:

> expected number of interventions remaining until goal \(g\) is completed, if
> action \(a\) is taken now and good actions are chosen afterward.

For an unfinished goal, the implementation enforces

\[
C\ge 1
\]

using

\[
C = 1+\operatorname{softplus}(\text{raw network output}).
\]

The learned distance from the current experiential state to the goal is

\[
\boxed{
d(z,g,j)=\min_a C(z,g,j,a)
}
\]

and the temporal-difference target is

\[
C(z_t,g,j_t,a_t)
\approx
\begin{cases}
1, & \text{if }a_t\text{ completes the goal},\\
1+\min_a C(z_{t+1},g,j_{t+1},a), & \text{otherwise}.
\end{cases}
\]

This resembles a stochastic shortest-path Bellman equation.

The method

```python
agent.distances_to_all_goals()
```

returns

```text
[d(z,g_0), d(z,g_1), ..., d(z,g_N)]
```

for the current history.

The code also tracks how the *same observed history* has progressed toward every
known goal, even though only one goal generated reward during the episode.

---

# 10. Why reachability is not forced to be an ordinary metric

Quantum measurement disturbance naturally makes control directional.

It may be easy to move from one experimentally relevant condition toward a goal
but impossible or expensive to reverse that process.

Therefore the model does **not** impose

\[
d(x,y)=d(y,x).
\]

The learned reachability should be thought of as a directed cost-to-go or
quasimetric-like object.

The Euclidean goal embedding and the directional reachability cost answer
different questions:

\[
\|e_g-e_h\|
\quad\text{asks}\quad
\text{“Are these goals achieved similarly?”}
\]

whereas

\[
d(z,g)
\quad\text{asks}\quad
\text{“How difficult is this goal from my present history?”}
\]

---

# 11. Complete multi-goal loss

The full GRU agent optimizes approximately

\[
L =
L_Q
+
\lambda_{\rm pred}L_{\rm pred}
+
\lambda_{\rm cost}L_{\rm cost}
+
\lambda_{\rm geom}L_{\rm geom}.
\]

### Reward control

\[
L_Q
=
\left(
Q(z_t,g,a_t)
-
[
r_t+\gamma\max_aQ_{\rm target}(z_{t+1},g,a)
]
\right)^2.
\]

### Outcome prediction

\[
L_{\rm pred}
=
-\log \hat P(o_t\mid z_t,a_t).
\]

### Reachability cost

\[
L_{\rm cost}
=
\operatorname{Huber}
\left(
C(z_t,g,a_t),
1+\min_a C_{\rm target}(z_{t+1},g,a)
\right),
\]

with target \(1\) if the current action completes the goal.

### Goal geometry

\[
L_{\rm geom}
=
\left(
\|e_g-e_h\|
-
c\sqrt{
D_{\rm JS}(\pi_g,\pi_h)
}
\right)^2.
\]

---

# 12. Backpropagation through time

Training trajectories are collected without retaining a gradient graph.

At episode end, the observed sequence is replayed through the GRU:

\[
z_0
\to
z_1
\to
z_2
\to
\cdots
\to
z_T.
\]

PyTorch then differentiates losses backward through this recurrence.

Thus a reward or prediction error at a late time can alter how the GRU encodes
an event many steps earlier.

Both recurrent files also use a slowly updated target network for TD targets:

\[
\theta_{\rm target}
\leftarrow
(1-\tau)\theta_{\rm target}
+
\tau\theta_{\rm online}.
\]

This is a standard stabilization device.

---

# 13. Running the experiments

Install dependencies:

```bash
pip install numpy torch
```

## Multi-goal tabular baseline

```bash
python multi_goal_q_learning.py
```

## Single-goal predictive GRU

```bash
python predictive_gru_q_learning.py
```

Custom single goal:

```bash
python predictive_gru_q_learning.py \
  --goals "Z0X0=0:0,1:0" \
  --episodes 30000
```

## Full multi-goal predictive GRU

```bash
python multi_goal_gru.py
```

Save geometry:

```bash
python multi_goal_gru.py \
  --episodes 50000 \
  --geometry-prefix results/geometry
```

## Direct backend comparison

Run identical task parameters with the same random seed:

```bash
python compare_backends.py \
  --backend tabular \
  --seed 7 \
  --episodes 30000 \
  --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0"
```

then

```bash
python compare_backends.py \
  --backend multi-gru \
  --seed 7 \
  --episodes 30000 \
  --goals "Z0=0:0;X0=1:0;Z0X0=0:0,1:0"
```

For serious comparisons, repeat over several random seeds.

---

# 14. Evaluation outputs

The common evaluator reports, separately for every goal:

- success rate;
- mean number of interventions conditional on success;
- mean reward.

CSV output can be enabled with:

```bash
--csv results.csv
```

The full multi-goal GRU can additionally save:

```text
<prefix>_embeddings.csv
<prefix>_goal_distances.csv
<prefix>_initial_reachability.csv
```

These expose:

1. the raw learned goal coordinates;
2. pairwise Euclidean goal distances;
3. the learned distance from blank history to each goal.

---

# 15. Important experimental caveats

## Reachability estimates are learned approximations

The cost head is not guaranteed to equal the exact optimal hitting time.
It is trained off-policy using a TD/min backup. Its quality must be checked
empirically.

A good validation is to compare

\[
d(z_0,g)
\]

against the actual mean number of steps achieved by the trained policy.

## Embedding geometry is partly imposed

The statement

> “nearby goals use similar strategies”

does not emerge from nowhere: `multi_goal_gru.py` includes an explicit geometry
regularizer based on policy similarity.

An important ablation is:

```bash
--geometry-weight 0
```

Then ask whether useful goal geometry emerges without direct pressure.

## Goal IDs have no semantics by themselves

The current full model uses a learned lookup embedding for a finite goal set.
This is ideal for studying geometry among a known set of goals, but it cannot
immediately generalize to a never-before-seen goal description.

A natural next version should encode the actual goal sequence

\[
((a_1,o_1),\ldots,(a_k,o_k))
\]

with a separate GRU or transformer, producing \(e_g\) compositionally.

## Prediction is one-step prediction

The current outcome head learns

\[
P(o_{t+1}\mid z_t,a_{t+1}).
\]

A stronger predictive-state objective would also train multi-step imagined
intervention sequences:

\[
P(
o_{t+1:t+k}
\mid
z_t,
a_{t+1:t+k}
).
\]

That would put much stronger pressure on \(z_t\) to become a sufficient
predictive state.

---

# 16. Recommended next experiments

### A. Prediction ablation

Compare:

```text
prediction_weight = 0
prediction_weight > 0
```

Does explicit outcome prediction improve sample efficiency or latent-state
quality?

### B. History-length comparison

Compare the GRU against tabular agents with:

```text
history_length = 1, 2, 4, 8, ...
```

This tests whether learned recurrent memory outperforms arbitrarily fixed
finite memory.

### C. Geometry ablation

Compare:

```text
geometry_weight = 0
geometry_weight > 0
```

Then visualize the goal embeddings and compare them against independent
behavioral similarity measurements.

### D. Validate distance

For each goal \(g\), compare:

```text
predicted d(z0,g)
```

against:

```text
empirical mean hitting time under the learned policy
```

Calibration matters if the distance is to be interpreted literally.

### E. Compare agent positions after different histories

Take several deliberately generated histories \(h_i\), encode them into \(z_i\),
and compute

\[
\mathbf d(z_i)
=
(
d(z_i,g_1),\ldots,d(z_i,g_N)
).
\]

This gives a goal-relative coordinate representation of the agent's present
experimental situation.

### F. Goal-sequence encoder

Replace `nn.Embedding(n_goals, goal_dim)` with a recurrent/transformer encoder of
the goal specification itself.

This enables transfer to new goals.

### G. Transformer history backend

Replace the history GRU with a causal transformer while leaving the same heads:

```text
history -> transformer -> z_t
                         |-> outcome predictor
                         |-> Q
                         |-> cost-to-go
```

The shared backend protocol means the environment and evaluation code need not
change.

---

# 17. Core conceptual picture

The project is trying to learn, from intervention/outcome experience alone,

\[
h_t
\longmapsto
z_t,
\]

a compressed representation of the agent's experimental past;

\[
(z_t,a)
\longmapsto
\hat P(o\mid z_t,a),
\]

an empirical model of contemplated interventions;

\[
(z_t,g)
\longmapsto
d(z_t,g),
\]

a learned directed difficulty/reachability relation;

and

\[
g
\longmapsto
e_g,
\]

a geometry of goals based on similarity of the strategies by which they are
achieved.

The hidden density operator remains available only to the simulator for
generating experience. It is never part of the agent's input.

That separation should be preserved in every future backend.
