# Vision: Learning an Agent-Centered Geometry of Quantum Possibility

## 1. Basic premise

The central idea of this project is to study an artificial agent that interacts with a quantum system under an unusually strict informational constraint:

> The agent may only choose which measurement to perform and record which classical outcome occurs.

The agent is **not** given:

- the quantum state \(\rho\);
- the Kraus operators associated with its possible measurements;
- the Hamiltonian;
- Born-rule probabilities;
- expectation values;
- tomography;
- any privileged external description of the physical system.

Its raw experience has the form

\[
(a_1,o_1),(a_2,o_2),\ldots,
\]

where \(a_t\) is an intervention chosen by the agent and \(o_t\) is the outcome that the agent experiences.

The hidden quantum system evolves according to ordinary quantum mechanics. If the agent chooses measurement \(a\), whose outcome-\(o\) Kraus operator is \(K_o^{(a)}\), then the environment secretly generates

\[
p(o\mid \rho,a)
=
\operatorname{Tr}
\left(
K_o^{(a)}\rho K_o^{(a)\dagger}
\right),
\]

and updates the physical state by

\[
\rho'
=
\frac{
K_o^{(a)}\rho K_o^{(a)\dagger}
}{
p(o\mid\rho,a)
}.
\]

But none of these quantities are exposed to the agent.

The scientific question is therefore not merely:

> Can reinforcement learning control a quantum system?

It is instead:

> What internal representation of an unknown quantum world can an agent construct from its own interventions and consequences, and how can that representation support prediction, planning, and goal-directed control?

This is deliberately an operational and agent-centered problem.

### Space as a special hodological geometry

The ultimate hypothesis is stronger than representation learning. Physical
space may be unnecessary as a primitive in the agent's description. What the
agent directly learns is a hodological structure: goals that can be achieved
with few reliable interventions are near, while goals requiring long, risky,
or highly constrained strategies are far.

In general this goal-relative space will be high-dimensional, directed,
history-dependent, and non-Euclidean. Familiar two- or three-dimensional space
would be a remarkable special regime in which:

1. a large family of place-like goals has an approximately symmetric
   reachability cost;
2. those costs admit a low-stress common embedding in two or three dimensions;
3. interventions act as approximately local displacements in that embedding;
4. observed action histories become stochastic trajectories through the
   emergent coordinates.

The inverse scientific question is therefore:

> Which initial quantum states, Kraus instruments, observation coarse-grainings,
> and goal repertoires make a low-dimensional spatial interpretation possible?

This is not a claim that every goal geometry is space. It is a program for
identifying the operational conditions under which one hodological factor
behaves like ordinary space, and then progressively removing the simplifying
conditions used to produce it.

The longer-term target is a total hodological space containing spatial and
internal directions. Mathematically, the desired decomposition resembles a
fiber bundle

\[
F_b\hookrightarrow E\xrightarrow{\pi}B,
\]

where the base \(B\) is an emergent 2D or 3D spatial geometry and the fiber
\(F_b\) contains internal, predictive, or task-specific possibilities at place
\(b\). Horizontal interventions move in the base, vertical interventions
change internal state, and coupled actions do both. Only after such a
decomposition, its connection, and its curvature are operationally measurable
does it become meaningful to ask whether anything analogous to general
relativity can emerge.

---

# 2. The hidden-state problem

From an external physicist's perspective, the quantum state \(\rho_t\) is the natural state variable.

For the artificial agent, however, \(\rho_t\) is inaccessible.

The only thing the agent actually possesses is its experiential history

\[
h_t
=
(a_1,o_1,\ldots,a_{t-1},o_{t-1}).
\]

Therefore the control problem is naturally partially observed.

The fundamental policy is not

\[
\pi(a_t\mid \rho_t),
\]

but

\[
\pi(a_t\mid h_t).
\]

The challenge is that the history grows indefinitely. A useful agent should compress the past into some internal state

\[
z_t = F(h_t)
\]

that preserves the information relevant to future prediction and action.

This produces the first core research object of the project:

\[
\boxed{
h_t \longmapsto z_t
}
\]

where \(z_t\) is the agent's learned internal representation of its current experimental situation.

The hope is not necessarily that \(z_t\) becomes a literal density matrix.

Instead, the desired interpretation is:

> \(z_t\) should contain whatever distinctions among past histories matter for the agent's future experimental expectations and possible actions.

---

# 3. From finite history to learned recurrent state

The simplest baseline uses a literal finite history:

\[
s_t
=
\big(
(a_{t-L},o_{t-L}),
\ldots,
(a_{t-1},o_{t-1})
\big).
\]

A tabular Q-learning algorithm can then estimate

\[
Q(s_t,a).
\]

This gives an important baseline because it demonstrates that useful control may be learned without access to the hidden quantum description.

However, finite history is arbitrary.

Why should the previous four observations matter rather than the previous five or fifty?

A recurrent neural network avoids making that choice by learning a memory state

\[
z_t.
\]

In the current implementation, a gated recurrent unit (GRU) updates this memory according to

\[
z_{t+1}
=
F_\theta(z_t,a_t,o_t).
\]

The GRU is trained so that the information retained in \(z_t\) is useful for future decisions.

The conceptual shift is important:

\[
\text{raw history}
\quad\longrightarrow\quad
\text{learned operational state}.
\]

The artificial agent is no longer simply memorizing observations. It is learning what features of its own experience deserve to count as part of its present state.

---

# 4. Learning an empirical model of interventions

A pure model-free reinforcement learner need only learn which action is useful.

But the project becomes more interesting when the agent also learns what its possible interventions tend to do.

The proposed predictive model is

\[
\boxed{
\hat P_\theta(o\mid z,a)
}
\]

which means:

> Given my present internal state \(z\), if I were to perform intervention \(a\), what outcomes should I expect?

This model is trained directly from experience.

At time \(t\), before the newly observed outcome is incorporated into memory, the agent predicts

\[
\hat P_\theta(o_t\mid z_t,a_t).
\]

Once the outcome \(o_t\) is experienced, the prediction loss is

\[
L_{\rm pred}
=
-\log
\hat P_\theta(o_t\mid z_t,a_t).
\]

No quantum state reconstruction is required.

No Kraus operator is inferred explicitly.

The agent simply develops a predictive model of the consequences of its contemplated actions.

This gives the second core research object:

\[
\boxed{
(z,a)
\longmapsto
\hat P(o\mid z,a).
}
\]

This is an operational world model.

It says what the agent expects to experience if it performs a particular intervention.

---

# 5. Predictive-state interpretation

The predictive model suggests a deeper interpretation of \(z_t\).

Consider two histories \(h\) and \(h'\).

Suppose they lead to exactly the same statistics for every future sequence of contemplated experiments:

\[
P(o_{1:k}\mid a_{1:k},h)
=
P(o_{1:k}\mid a_{1:k},h')
\]

for every future action sequence \(a_{1:k}\) and outcome sequence \(o_{1:k}\).

Then, from the agent's operational perspective, those histories are indistinguishable.

They support exactly the same future expectations.

This defines an equivalence relation:

\[
h\sim h'
\]

whenever all future intervention-outcome predictions agree.

The equivalence classes

\[
[h]
\]

constitute an operational notion of state.

Quantum theory tells an external physicist that such predictive information may often be represented by a density operator.

But the artificial agent does not need to know that representation in advance.

A sufficiently capable recurrent or transformer model may instead discover its own coordinates

\[
h_t\mapsto z_t
\]

on the space of operationally distinct predictive states.

One ambitious goal of the project is therefore to ask:

> If an agent is trained only to predict and control, does a low-dimensional state space analogous to the quantum state space emerge inside its learned representation?

---

# 6. Adding goals

The next step is to give the agent many possible objectives.

A goal may be a desired sequence of future intervention-outcome events, for example

\[
g
=
\big(
(Z,0),
(X,1),
(Z,0)
\big).
\]

The agent may be allowed to perform exploratory or preparatory measurements between the required checkpoints.

For several goals

\[
g_1,\ldots,g_N,
\]

the natural goal-conditioned value function is

\[
Q(z,a,g).
\]

This asks:

> Given my current experiential state \(z\), and given that I now care about goal \(g\), how useful would intervention \(a\) be?

The same history can therefore support different policies depending on which future the agent is trying to realize:

\[
\pi(a\mid z,g_1),
\qquad
\pi(a\mid z,g_2),
\qquad
\ldots
\]

This gives a unified multi-goal agent rather than one unrelated policy per task.

---

# 7. Goals as points in a learned space

The central geometric proposal is to represent each goal by an embedding

\[
\boxed{
g\longmapsto e_g\in\mathbb R^k.
}
\]

But the purpose of the embedding is not merely compression.

We want the geometry of the embedding space to carry operational meaning.

The most important proposed meaning is:

> Two goals should be close when the agent tends to achieve them in similar ways.

Thus goal similarity should be defined behaviorally rather than linguistically or syntactically.

Two goals may look very different as symbolic descriptions while nevertheless requiring almost the same control strategy.

Conversely, two superficially similar goals may require very different sequences of interventions because quantum measurement backaction changes what is reachable.

The relevant geometry should therefore be learned from the agent's behavior.

---

# 8. Strategy similarity as the basis of goal geometry

Suppose the agent has learned a goal-conditioned policy

\[
\pi(a\mid z,g).
\]

At a given experiential state \(z\), compare the policies for two goals \(g\) and \(h\):

\[
\pi(\cdot\mid z,g)
\qquad\text{and}\qquad
\pi(\cdot\mid z,h).
\]

If they are nearly identical for many relevant \(z\), then the two goals call for similar behavior.

A natural local behavioral distance is therefore

\[
D_{\rm local}(g,h;z)
=
D_{\rm JS}
\left[
\pi(\cdot\mid z,g),
\pi(\cdot\mid z,h)
\right],
\]

where \(D_{\rm JS}\) is the Jensen-Shannon divergence.

Average over experiential states encountered by the agent:

\[
D_{\rm strategy}(g,h)
=
\mathbb E_{z\sim\mu}
\left[
D_{\rm local}(g,h;z)
\right].
\]

This quantity asks:

> Across the situations I actually encounter, how differently would I act if I were pursuing \(g\) rather than \(h\)?

The goal embeddings can then be trained so that

\[
\boxed{
\|e_g-e_h\|
\approx
c\,D_{\rm strategy}(g,h)
}
\]

or some monotone function thereof.

The current implementation uses a scaled square root of Jensen-Shannon divergence, but this is only one possible choice.

The deeper principle is more general:

\[
\boxed{
\text{distance between goals}
\;\sim\;
\text{difference between strategies for realizing them}.
}
\]

---

# 9. Goal similarity can also be trajectory-based

Policy similarity at individual states is not the only possible notion.

One may instead compare the actual trajectories induced by pursuing different goals.

For each goal, define a feature representation of the future intervention-outcome stream:

\[
\phi(a_t,o_t).
\]

Then define the expected discounted trajectory signature

\[
S(g)
=
\mathbb E_{\pi_g}
\left[
\sum_{t=0}^{\infty}
\gamma^t
\phi(a_t,o_t)
\right].
\]

The vector \(S(g)\) might encode:

- how frequently each measurement is used;
- how often each outcome occurs;
- common measurement transitions;
- recurring action-outcome motifs;
- expected episode duration;
- entropy of the policy;
- expected disturbance patterns.

Then one may define

\[
D_{\rm trajectory}(g,h)
=
\|S(g)-S(h)\|.
\]

This is closely related in spirit to successor-feature representations.

It may eventually be preferable to the local policy-divergence definition because it compares complete strategies rather than single-step action distributions.

The project should treat the definition of goal similarity as an empirical question.

Several candidate geometries should be compared.

---

# 10. Distance from the agent to a goal

Goal-goal similarity is not the same as the distance from the agent's present situation to a goal.

We therefore introduce a second geometry.

Let \(\tau_g\) be the first future time at which goal \(g\) is achieved.

An ideal definition is

\[
\boxed{
d(z,g)
=
\min_\pi
\mathbb E_\pi
\left[
\tau_g
\mid z
\right].
}
\]

This says:

> How many further interventions should I expect to need, starting from my present experiential state, if I pursue this goal optimally?

This is a direct operational notion of difficulty.

A goal that can be achieved immediately has small distance.

A goal requiring a delicate preparation sequence has large distance.

A goal that is effectively unreachable should have very large or infinite distance.

---

# 11. Bellman equation for reachability

The distance has a stochastic shortest-path structure.

If the goal is not yet achieved, the agent must first choose an intervention \(a\).

That intervention produces an outcome \(o\) with learned probability

\[
\hat P(o\mid z,a).
\]

The internal state then updates to

\[
z'=F(z,a,o).
\]

So the ideal Bellman equation is

\[
\boxed{
d(z,g)
=
1+
\min_a
\sum_o
P(o\mid z,a)
d(F(z,a,o),g).
}
\]

This equation has a striking interpretation.

The agent can ask:

> Which possible intervention is expected to move me closest to my goal?

This turns the learned geometry itself into a control principle.

Instead of treating the policy as an unrelated black-box output, one can derive action choice from expected distance reduction.

---

# 12. Action-conditioned cost-to-go

The implementation learns an action-conditioned cost

\[
C(z,g,a),
\]

where

\[
C(z,g,a)
\]

is interpreted as:

> the expected number of remaining interventions if I take \(a\) now and then behave optimally.

The state-goal distance is

\[
\boxed{
d(z,g)=\min_a C(z,g,a).
}
\]

A sampled temporal-difference target is

\[
C(z_t,g,a_t)
\approx
1+\min_a C(z_{t+1},g,a),
\]

unless the current intervention completes the goal, in which case the target cost is \(1\).

This is the basis of the current learned reachability metric.

---

# 13. Why the agent-goal geometry is directional

An ordinary metric satisfies

\[
d(x,y)=d(y,x).
\]

There is no reason to impose that here.

Quantum interventions may destroy information, prepare states irreversibly from the agent's perspective, or make some goals easier while making others harder.

For example, an outcome-producing measurement may take the agent from a situation in which goal \(g_1\) is easy to one in which \(g_1\) is difficult, while simultaneously making \(g_2\) easy.

The natural geometry of control is therefore directed.

One should think of

\[
d(z,g)
\]

as a reachability cost or quasimetric-like object rather than ordinary Euclidean distance.

This distinction is essential:

\[
\boxed{
D_G(g,h)
=
\text{symmetric similarity between goals}
}
\]

versus

\[
\boxed{
d(z,g)
=
\text{directed difficulty of achieving }g\text{ from here}.
}
\]

They answer different questions.

---

# 14. The agent's position relative to all goals

Given \(N\) goals, define the distance vector

\[
\boxed{
\mathbf D(z)
=
\left(
d(z,g_1),
d(z,g_2),
\ldots,
d(z,g_N)
\right).
}
\]

This vector provides another representation of the agent's present situation.

Two experiential states \(z\) and \(z'\) are similar with respect to the current repertoire of goals if

\[
\mathbf D(z)\approx \mathbf D(z').
\]

In words:

> From either history, all of my currently meaningful goals are about equally easy or difficult.

This suggests an explicitly goal-relative notion of state equivalence.

Instead of identifying situations because they predict exactly the same future measurement statistics, one may identify them because they afford the same future possibilities.

Thus the project naturally contains two complementary notions of state:

### Predictive state

Two histories are equivalent if they imply the same expectations for all future interventions.

### Pragmatic or goal-relative state

Two histories are equivalent if they place the agent at approximately the same reachability relation to all goals it currently cares about.

These need not coincide.

---

# 15. Interventions as displacements in goal space

The distance vector makes every observed intervention/outcome pair into a movement through a landscape of possibilities.

Before an intervention:

\[
\mathbf D(z_t)
=
\big(
d(z_t,g_1),\ldots,d(z_t,g_N)
\big).
\]

After choosing \(a_t\) and observing \(o_t\),

\[
z_{t+1}=F(z_t,a_t,o_t),
\]

so

\[
\mathbf D(z_{t+1})
\]

changes.

One action may move the agent closer to some goals and farther from others.

Thus an intervention may be characterized by a displacement

\[
\Delta\mathbf D
=
\mathbf D(z_{t+1})-\mathbf D(z_t).
\]

This is an especially useful way to think about quantum measurement backaction.

A measurement does not merely reveal information.

It changes the agent's future possibilities.

The geometry of distances to goals makes that change explicit.

---

# 16. Contemplated interventions and expected displacement

Once the agent has learned

\[
\hat P(o\mid z,a),
\]

it can contemplate an intervention before performing it.

For every possible outcome \(o\), it can construct the hypothetical next state

\[
z'_o=F(z,a,o).
\]

It can then calculate

\[
\mathbf D(z'_o)
\]

for each branch.

Therefore the contemplated action \(a\) generates a probability distribution over future goal-distance vectors.

The expected distance to a particular goal becomes

\[
\mathbb E[d'_{g}\mid z,a]
=
\sum_o
\hat P(o\mid z,a)
d(F(z,a,o),g).
\]

Similarly, the expected entire displacement vector is

\[
\mathbb E[\Delta\mathbf D\mid z,a].
\]

This gives the agent an operational form of planning:

> If I do this, what possible outcomes do I expect, and how would each outcome alter the space of goals available to me?

---

# 17. A possible intrinsic notion of useful information

The predictive and goal-geometric pictures can be combined.

An intervention may be valuable for two different reasons:

1. it moves the agent closer to a chosen goal;
2. it teaches the agent something that improves future predictions or decisions.

This suggests defining an information-seeking value for contemplated experiments.

For example, the agent might prefer actions expected to reduce uncertainty in

\[
\hat P(o\mid z,a')
\]

for future actions \(a'\), or actions expected to reduce uncertainty in its distances

\[
d(z,g).
\]

This would turn exploration itself into a goal-directed experimental activity.

The agent would not merely act randomly for \(\epsilon\)-greedy exploration.

It could learn to perform informative interventions because they improve its model of the world and its map of future possibilities.

This is a natural future extension.

---

# 18. Why goal embeddings are interesting

The embeddings

\[
e_g
\]

should not be interpreted merely as parameters needed by a neural network.

They may reveal structure in the set of tasks available to the agent.

Suppose two goals repeatedly receive nearby embeddings.

That may indicate:

- they are achieved by almost the same sequence of interventions;
- one is a small variation of the other;
- they share a common preparatory subroutine;
- the same regions of experiential state space make them easy;
- the same measurement outcomes act as bottlenecks.

Clusters in goal space may therefore reveal latent families of experimentally related objectives.

One can imagine discovering:

- measurement-preparation families;
- incompatible goal clusters;
- bridge goals that connect otherwise distinct strategies;
- hierarchical task structure;
- approximately compositional directions in goal space.

The learned geometry becomes a scientific object to inspect, not merely a means to improve performance.

---

# 19. Goal geometry should be validated, not assumed

A crucial methodological principle is that any learned geometric interpretation must be checked empirically.

For goal-goal distance, compare

\[
\|e_g-e_h\|
\]

against independent measures such as:

- average policy divergence;
- trajectory-feature distance;
- overlap between optimal action sequences;
- transfer performance from \(g\) to \(h\);
- similarity of successor representations;
- number of parameter updates required to adapt from one goal to another.

For agent-goal distance, compare

\[
d(z,g)
\]

against:

- empirical mean hitting time;
- success probability within a fixed horizon;
- expected intervention cost;
- robustness to stochastic outcomes.

The geometry becomes meaningful only to the extent that these operational relationships are calibrated.

---

# 20. A richer reachability object: success probability vs time

Expected hitting time is not the only notion of difficulty.

Consider two goals:

- \(g_1\): 90% chance of success in 3 steps, 10% chance of catastrophic delay;
- \(g_2\): always succeeds in exactly 5 steps.

Their mean hitting times may be similar, yet their risk profiles differ.

A richer object is therefore the entire reachability curve

\[
R_g(z,T)
=
P(
\tau_g\le T
\mid z,\pi^\star_g
).
\]

This says:

> What is the probability that I can achieve this goal within \(T\) more interventions?

Then a scalar distance can be derived in many ways, but the full function

\[
T\mapsto R_g(z,T)
\]

contains more information.

Long-term, the project's "geometry" may therefore be better thought of as a geometry induced by distributions over reachable futures rather than a single scalar metric.

---

# 21. Goal composition

The current goals are explicit sequences of action/outcome checkpoints.

A future architecture should encode their internal structure rather than assigning each one an arbitrary lookup embedding.

Suppose

\[
g
=
((a_1,o_1),\ldots,(a_k,o_k)).
\]

A goal encoder could learn

\[
e_g
=
G_\phi(
(a_1,o_1),\ldots,(a_k,o_k)
).
\]

This encoder might itself be:

- a GRU;
- a transformer;
- a graph network;
- a compositional symbolic encoder.

Then previously unseen goals could be embedded immediately.

The agent could exploit geometric proximity to known goals to transfer strategies.

This would transform goal space from a visualization of a finite task list into a genuinely generative representation of possible aims.

---

# 22. Hierarchical structure in goal space

Many complicated goals may share subgoals.

For example,

\[
g_1=(A,B,C),
\qquad
g_2=(A,B,D).
\]

Both require reaching the intermediate condition \((A,B)\).

A sufficiently rich goal geometry should detect this shared structure.

One could therefore learn:

- embeddings for whole goals;
- embeddings for subgoals;
- reusable options or skills associated with regions of goal space.

This suggests a hierarchy:

\[
\text{primitive interventions}
\rightarrow
\text{learned subroutines}
\rightarrow
\text{goal families}
\rightarrow
\text{long-horizon tasks}.
\]

The geometry may eventually encode not only similarity but compositional structure.

---

# 23. Relationship between predictive state and goal geometry

There are two latent spaces in the project:

\[
z
\]

for the agent's current experiential state, and

\[
e_g
\]

for goals.

The interaction between them is central.

The distance function

\[
d(z,g)
\]

relates these spaces.

One may think of \(d\) as defining a bipartite geometry between:

- where the agent currently is;
- where it wants to get.

This opens several questions.

### Can \(z\) and \(e_g\) be embedded into a common space?

Perhaps there exists a representation in which

\[
d(z,g)
\approx
\|f(z)-e_g\|.
\]

This would yield a very intuitive geometry.

But it may be too restrictive because reachability is directional.

### Should the common space be non-Euclidean?

Possibilities include:

- quasimetric embeddings;
- hyperbolic geometry;
- Finsler-like geometry;
- directed graph geometry;
- learned energy functions;
- optimal-transport-style distances.

The appropriate geometry should be discovered from the control problem rather than imposed for aesthetic reasons.

---

# 24. Quantum structure as an emergent constraint

The environment is quantum, but the agent's learning rules do not explicitly contain quantum formalism.

This creates an interesting experimental question:

> Which specifically quantum structures become visible in the geometry learned by an otherwise generic prediction-and-control agent?

Possible signatures include:

- incompatibility between goals associated with noncommuting measurements;
- irreversible-looking reachability relations caused by measurement disturbance;
- state-space dimensions reflecting informational completeness;
- convex structure in predictive representations;
- equivalence classes corresponding to operationally indistinguishable density operators;
- characteristic tradeoffs between simultaneously accessible goal families.

Rather than assuming the Hilbert-space formalism, the project asks which parts of that structure are recoverable as useful regularities in the agent's own experience.

---

# 25. Informational completeness and agent-relative state space

Suppose the agent has access to an informationally complete set of interventions.

Then, in principle, its predictive state may need to distinguish every physically distinct density operator relevant to those interventions.

If its available measurements are not informationally complete, many physically different density operators may be operationally indistinguishable to the agent.

Therefore the effective learned state space should depend on the agent's intervention repertoire.

This is a major conceptual point.

The learned state is not simply:

\[
\text{“the true state of the system.”}
\]

It is:

\[
\boxed{
\text{the distinctions among histories that matter relative to the agent's available actions and goals}.
}
\]

Changing the action repertoire may change the state space itself.

Changing the goal repertoire may change which distinctions are pragmatically relevant.

This makes both predictive state and goal geometry explicitly agent-relative.

---

# 26. Transformer vs recurrent architectures

The current implementation uses a GRU because it provides a simple persistent state

\[
z_t.
\]

A transformer could instead process the full history:

\[
(a_1,o_1),\ldots,(a_t,o_t)
\]

and use attention to retrieve arbitrarily old relevant events.

A transformer may be advantageous when:

- long-range dependencies matter;
- particular isolated events in the distant past remain relevant;
- histories have complex relational structure.

A GRU may be advantageous when:

- the environment admits a compact sufficient state;
- online interaction requires constant memory;
- interpretability of a persistent latent state is important.

The correct architecture is therefore itself experimentally testable.

The scientific object of interest is not the GRU as such.

It is the learned operational representation.

---

# 27. The current computational ladder

The project now has a deliberately staged sequence of increasingly rich agents.

## Stage 1: tabular finite-history Q-learning

\[
h_t^{(L)}
\rightarrow
Q(s,a,g).
\]

Question:

> Can useful measurement strategies be learned at all from action/outcome experience?

## Stage 2: recurrent Q-learning

\[
h_t
\rightarrow
z_t
\rightarrow
Q(z,a,g).
\]

Question:

> Can the agent learn its own compressed memory representation?

## Stage 3: predictive recurrent agent

\[
h_t
\rightarrow
z_t
\]

with

\[
Q(z,a,g)
\]

and

\[
\hat P(o\mid z,a).
\]

Question:

> Does explicitly modeling experimental consequences produce a more meaningful latent state?

## Stage 4: multi-goal predictive agent

Add

\[
e_g,
\qquad
C(z,g,a),
\qquad
d(z,g).
\]

Question:

> Can the agent organize its possible aims into a geometry based on strategy and reachability?

## Stage 5: model-based planning

Use

\[
\hat P(o\mid z,a)
\]

and

\[
F(z,a,o)
\]

to imagine future branches.

Question:

> Can the agent choose measurements by explicitly simulating how they alter its distances to possible goals?

## Stage 6: compositional goal encoding

Replace finite goal IDs with a goal encoder.

Question:

> Can geometry support generalization to genuinely new objectives?

---

# 28. A possible long-term picture

The long-term architecture may look like

\[
\boxed{
\text{experience}
\longrightarrow
\text{predictive state}
\longrightarrow
\text{map of possible goals}
\longrightarrow
\text{contemplated interventions}
\longrightarrow
\text{new experience}.
}
\]

More explicitly:

\[
h_t
\longmapsto
z_t,
\]

\[
(z_t,a)
\longmapsto
\hat P(o\mid z_t,a),
\]

\[
(z_t,a,o)
\longmapsto
z_{t+1},
\]

\[
g
\longmapsto
e_g,
\]

\[
(z_t,g)
\longmapsto
d(z_t,g),
\]

and therefore

\[
(z_t,g)
\longmapsto
\text{which contemplated intervention best improves reachability}.
\]

This is an agent that does not begin with a physical state space.

It begins with actions and consequences.

It learns:

- what distinctions in its past matter;
- what consequences its interventions tend to have;
- which goals resemble one another;
- where it currently stands relative to those goals;
- which interventions transform its future possibilities.

---

# 29. The most important conceptual distinction

There are three different notions that should remain separate throughout the project.

## Predictive similarity of histories

\[
h\sim_{\rm pred}h'
\]

when they imply similar future experimental statistics.

## Strategic similarity of goals

\[
g\sim_{\rm strat}g'
\]

when they are achieved by similar policies or trajectories.

## Reachability from the present

\[
d(z,g)
\]

measures how difficult a particular goal is from the current experiential state.

These relations interact, but none should simply be identified with another.

Keeping them distinct makes it possible to ask genuinely interesting questions about when they coincide.

---

# 30. Core research questions

The project can be organized around a small set of questions.

### Representation

What minimal latent information must an agent retain from its intervention-outcome history to support accurate prediction and effective control?

### Emergence of state

Does the learned predictive representation reproduce, approximate, or reorganize the operational state space predicted by quantum theory?

### Intervention modeling

Can an agent learn a useful empirical model of measurement backaction without reconstructing Kraus operators explicitly?

### Goal geometry

Do goals cluster according to the strategies required to achieve them?

What behavioral definition of goal distance is most useful?

### Reachability

Can the learned quantity

\[
d(z,g)
\]

be calibrated as an actual expected intervention cost or hitting time?

### Directionality

What asymmetric structure does quantum measurement disturbance induce in the reachability geometry?

### Transfer

Does geometric proximity between goals predict how easily knowledge transfers from one goal to another?

### Compositionality

Can complex goals be represented as combinations or trajectories in a learned goal space?

### Agent relativity

How do the learned predictive and goal spaces change when the set of available interventions or goals changes?

---

# 31. Immediate experiments worth running

A concrete next research program could proceed as follows.

## Experiment 1: validate prediction

Train the predictive GRU and compare

\[
\hat P(o\mid z,a)
\]

against empirical outcome frequencies.

Do this across deliberately chosen histories.

## Experiment 2: inspect latent dimensionality

Collect pairs

\[
(z_t,\rho_t)
\]

using \(\rho_t\) only for offline analysis.

Ask whether the latent \(z_t\) lies near a low-dimensional manifold and whether the hidden Bloch vector can be decoded from it.

## Experiment 3: vary informational completeness

Remove some measurements.

Ask which dimensions of the learned predictive state disappear.

## Experiment 4: validate reachability

Compare predicted

\[
d(z,g)
\]

with empirical hitting times under the learned goal-conditioned policy.

## Experiment 5: compare goal metrics

Compare:

- goal-embedding Euclidean distance;
- policy Jensen-Shannon divergence;
- trajectory-feature distance;
- transfer learning speed;
- overlap of discovered action motifs.

Ask which quantities agree.

## Experiment 6: intervention displacement maps

For a collection of latent states \(z\), estimate how each possible action changes

\[
\mathbf D(z).
\]

Visualize the resulting goal-distance flow.

## Experiment 7: new-goal transfer

Train a compositional goal encoder, hold out some goals entirely, and test whether nearby known goals provide useful zero-shot or few-shot strategies.

---

# 32. Vision statement

The project begins with an intentionally austere agent:

> It can act, and it can experience consequences.

Everything else must be learned.

From those experiences, the agent gradually constructs:

\[
\text{a state space of what it currently expects},
\]

\[
\text{a model of what its interventions tend to do},
\]

\[
\text{a geometry of possible aims},
\]

and

\[
\text{a notion of how far those aims are from its present situation}.
\]

The most ambitious version of the idea is that physical state, control, and goal-directed possibility are not supplied as separate primitives.

They emerge together from the structure of intervention and consequence.

In that sense, the project is not merely about applying reinforcement learning to quantum control.

It is an attempt to computationally study how an agent can build an **operational world and a geometry of its own possibilities** from first-person experimental experience alone.

---

# 33. Implemented comparative program

The software now varies the intervention repertoire as well as the agent. Its
catalog includes sharp and unsharp qubit measurements, an informationally rich
four-outcome qubit SIC alongside Pauli measurements, and qutrit mutually
unbiased bases. This realizes the proposed experiment on agent-relative state
spaces: the same learning architecture can now encounter worlds with different
dimensions, disturbance profiles, and outcome alphabets.

The goal geometry is inspected through several deliberately separate views:

1. Euclidean distance between learned goal embeddings;
2. held-out Jensen–Shannon strategy distance;
3. distance between complete empirical trajectory signatures;
4. directed distance from the current experiential history to each goal;
5. intervention/outcome displacement of the entire goal-distance vector;
6. finite-horizon reachability curves rather than mean hitting time alone.

These views prevent a visually attractive embedding from being treated as
self-validating. A useful learned geometry should predict independent behavior,
calibrate against actual reachability, and make measurement-induced changes in
future possibility legible.

---

# 34. Implemented first spatial objective

The project now contains an explicit first construction, fully documented in
`SPATIAL_HODOLOGY.md`.

Its Hilbert space is the nine-dimensional span of localized symbols
\(\lvert x,y\rangle\) on a concealed \(3\times3\) arrangement. Eight movement
instruments implement axial and stochastic diagonal displacements, and a
single projective place probe supplies nine coordinate-free place outcomes.
The nine goals are simply to obtain each outcome from that common probe.

An outer inverse-design loop searches the diagonal success probability so that
expected diagonal cost matches Euclidean diagonal length. It selects
\(p_d=0.715\), close to \(1/\sqrt2\), and yields exact 2D stress 0.0365. Three
Q-learning seeds then learn all-pairs navigation with 100% held-out success.
The policy-derived geometry has mean 1D/2D/3D stresses
0.372/0.071/0.064 and recovers the concealed lattice with Procrustes
\(R^2=0.975\).

Two controls are essential to the interpretation. Cardinal-only movement
recovers the coordinate arrangement but retains a less Euclidean Manhattan
metric (2D stress 0.142). Coarse success/failure observations preserve the same
hidden Kraus connectivity but reduce all-pairs success to 0.482 and raise 2D
stress to 0.233. Thus neither a grid-looking plot nor hidden spatial dynamics
alone is sufficient. Spatial emergence requires calibrated all-pairs behavior,
adequate operational localization or memory, and a movement repertoire with
approximately isotropic costs.

This result is intentionally modest. The construction is an
entanglement-breaking, measure-and-prepare quantum walk on localized inputs.
It establishes existence and supplies diagnostics; it does not yet show that
space emerges from generic coherent quantum dynamics. The next ladder is:

1. weaken and alias place observations while learning predictive memory;
2. optimize complete quantum channels and landmark goals rather than one
   movement probability;
3. scale the lattice and vary topology and curvature;
4. introduce internal states and test a learned base/fiber decomposition;
5. study holonomy and stochastic curvature;
6. replace Euclidean time with causal reachability before making any spacetime
   or gravitational claim.

---

# 35. Implemented predictive atlas: place as anticipated experience

The first item on that ladder has now been implemented. The successful spatial
agent no longer receives an exact current-place outcome before choosing a
movement. Instead, four weak binary QND instruments provide overlapping,
place-dependent evidence. A recurrent model integrates twelve scans and
predicts the outcome of a sharp landmark probe that the agent could perform
next. During navigation that probe is terminal: its outcome can validate a
commitment but cannot serve as a place report for a later decision.

This changes the interpretation of operational place. A place is no longer
merely the latest observed symbol. It is a distribution over possible future
landmark experiences,

\[
b_t(s)=P(\text{terminal landmark }s\mid h_t),
\]

maintained from the agent's history. Blind movement outcomes update this
distribution through a learned joint model

\[
\widehat P(o,s'\mid s,a).
\]

Goal-directed behavior then takes place in belief space. For a goal (g), the
agent chooses the movement minimizing its belief-weighted stochastic
shortest-path cost. The resulting all-pairs empirical costs, not the beliefs or
concealed coordinates, define the hodological distance matrix.

The production result across three seeds is:

- delayed landmark accuracy (0.964\pm0.008), near the Bayes ceiling
  (0.977\pm0.004);
- held-out all-pairs success (0.973\pm0.004), versus
  (1.000\pm0.001) for an online-localization oracle;
- 1D/2D/3D MDS stress (0.407/0.075/0.043);
- exact stochastic-cost correlation (0.948\pm0.013);
- concealed-coordinate Procrustes (R^2=0.987\pm0.005).

A control allowed the same 48 observations but retained only the last four. It
localized at (0.463) accuracy and navigated at (0.614) success; its learned
costs were uncorrelated with the exact map. A second control made every beacon
place-independent. It localized at (0.114), indistinguishable from (1/9)
chance, and navigated at (0.484). These comparisons show that temporal
integration of informative evidence is the operative resource.

The conceptual gain is important. The agent's “where” is now a predictive
hypothesis that earns its role by supporting future goal achievement. This is
closer to the desired view of space as an organization of possible action than
a state-estimation problem with supplied coordinates.

The remaining assumptions must stay visible. The nine landmark alternatives,
their delayed labels, and landmark-anchored transition surveys are designed;
movement is entanglement-breaking; beacons commute with the place basis; and a
fixed scan costs 48 interventions. The experiment therefore demonstrates
learned operational localization, not spontaneous discovery of a place basis.
Its complete derivation, protocol, controls, results, and artifacts are in
`PREDICTIVE_ATLAS.md`.

## The revised research frontier

The next step should make localization itself goal-directed. Sensing, moving,
and committing should compete under a common intervention cost. The agent
should request a beacon observation only when its expected reduction in
goal-relevant uncertainty exceeds its cost. This creates an active predictive
atlas in which “where am I?” is answered only to the resolution demanded by
present aims.

After active sensing, the most consequential sequence is:

1. learn sensing and movement models jointly from uninterrupted experience;
2. replace labeled landmarks with operationally discovered predictive tests;
3. inverse-design noncommuting quantum instruments and goal repertoires under
   locality, sensing-cost, and low-dimensionality constraints;
4. scale to multiple topologies and test dimensional stability out of sample;
5. introduce local internal quantum degrees of freedom and ask whether the
   predictive state factors into spatial base and internal fiber;
6. study path-dependent transport and holonomy before introducing curvature;
7. derive causal rather than Euclidean temporal organization before pursuing
   an emergent spacetime interpretation.

The key methodological standard remains unchanged: geometry has emerged only
to the extent that it predicts and improves independent behavior. A map-like
picture, a decodable hidden coordinate, or a low stress number without
competence and controls is insufficient.

---

# 36. Implemented active atlas: information has a hodological price

The revised research frontier has now been implemented and is documented in
`ACTIVE_PREDICTIVE_ATLAS.md`. Sensing, moving, and committing each cost one
intervention. The agent is no longer given a fixed scan; it chooses whether
additional place information is valuable for the present decision.

## A failure that sharpened the vision

The first active pilot exposed a new requirement. In the open grid, repeated
translations could push every unknown starting state against a boundary. A
blind policy could “home” to corner goals without identifying its source. This
was successful control but not motion through a source-sensitive atlas.

The correction replaces open translations with local unitary layer swaps.
Every movement kernel is doubly stochastic, so a uniform prior remains uniform
under blind action. The null-beacon controller then returns to (1/9) success.
This yields a stronger design principle:

> A world supports an actively learned spatial atlas only when cheap control
> cannot erase the distinctions that the atlas is meant to represent.

Reversibility is one sufficient constraint. It also moves the construction
beyond entanglement-breaking movement: successful actions apply coherent
permutation unitaries to superpositions.

## Pragmatic state versus common atlas

Two active learners reveal a distinction anticipated earlier in this document.
The first groups landmarks whenever they require the same next action for the
current goal. A three-step value-of-information calculation purchases a beacon
only when its expected reduction in decision error exceeds its unit cost. This
goal-relative state uses (16.12\pm0.11) total interventions and reaches
(0.938\pm0.002) success, but its movement geometry has 2D stress
(0.193\pm0.015).

The second learner preserves the full landmark posterior, actively selecting
the beacon with maximum expected entropy reduction until 0.95 confidence. It
uses (19.12\pm0.53) interventions and reaches (0.965\pm0.004) success.
Its movement costs have 2D stress (0.111\pm0.006), exact-cost correlation
(0.929\pm0.012), and Procrustes (R^2=0.987\pm0.004).

The fixed 48-probe policy obtains (0.976\pm0.002) success at
(50.49\pm0.02) total interventions. Active atlas construction therefore
removes 62.1% of total cost for a 1.1 percentage-point success loss.

This result makes the predictive/pragmatic distinction empirical:

\[
\text{goal-sufficient state}
\quad\ne\quad
\text{state sufficient for a shared geometry}.
\]

An agent can safely forget distinctions irrelevant to one immediate action.
But if those equivalences change with the goal, the resulting sequence of
pragmatic states does not preserve one consistent space. A common spatial atlas
is an informational achievement above and beyond local competence.

## The first base–fiber separation

Total difficulty now decomposes operationally:

\[
C^{\rm total}=C^{\rm sensing}+C^{\rm movement}+C^{\rm commitment}.
\]

Movement cost retains the 2D base. Sensing cost records internal epistemic work
and has only a weak and seed-variable correlation with movement distance.
Adding the two into one scalar dissimilarity warps the 2D map. A belief
trajectory can instead be drawn over the spatial base with normalized entropy as a vertical coordinate:
the agent descends through an epistemic fiber while localizing, then moves
horizontally at low uncertainty.

This is not yet a fiber bundle. It is the first implemented example in which
multiple internal predictive conditions sit over the same spatial base and
have distinct action costs. It suggests that the correct hodological object is
not a single metric space of all goals, but a structured projection

\[
\text{full predictive/goal state}
\longrightarrow
\text{spatial affordance base}.
\]

## Revised frontier

The next decisive step is to make both base and fiber less supervised:

1. learn observation and movement models jointly from uninterrupted history;
2. replace table-based beliefs with an action-conditional recurrent predictive
   state and plan directly in that latent space;
3. introduce internal quantum goals at each landmark and distinguish physical
   fiber coordinates from epistemic uncertainty;
4. use noncommuting sensors so acquiring place information changes internal
   quantum possibilities;
5. test local chart consistency, transition functions, closed-loop internal
   transport, and holonomy;
6. scale reversible locality to larger topologies without hidden
   synchronization mechanisms;
7. ask whether different agents, sensor costs, and goal repertoires recover the
   same spatial base but different fibers.

The active result adds a new validation criterion: an emergent geometry should
survive rational selective attention. If space appears only when the agent is
forced to observe everything, its operational status is fragile.

---

# 37. Low-dimensional quantum systems can support larger goal spaces

The nine-level studies made one place goal correspond to one orthogonal basis
state. That was an excellent first control experiment, but it leaves open the
central ontological question: did space emerge, or was a classical position
register merely redescribed through policy cost?

The low-dimensional miniproject answers the mathematical part decisively.
Hilbert dimension does not bound the number or dimension of an agent's goals.
It bounds perfect simultaneous distinguishability. A qubit has infinitely
many density operators, an action history has arbitrarily many equivalence
classes, and a sequence-goal recognizer adds its own state. The relevant
operational state is therefore

\[
(\rho_t,q_t),
\]

where \(q_t\) is the history-derived goal or predictive state. Geometry may
live in \(\rho\), in \(q\), in their coupling, or merely in a bookkeeping
convention. These possibilities must be separated experimentally.

## An exact qubit result—and why it is not yet enough

Two rationally independent qubit phase rotations \(U,V\) act faithfully as
\(\mathbb Z^2\) on one equatorial ray. Nine nonorthogonal orbit goals
\(U^iV^j|+\rangle\), with \(i,j\in\{0,1,2\}\), inherit the exact open-grid
Manhattan word metric. Binary random-unitary retry instruments with
success probability

\[
p_{rs}=\frac{1}{\sqrt{r^2+s^2}}
\]

give exact Euclidean expected intervention cost. The proof is analytic:
faithfulness removes word shortcuts, a geometric waiting time supplies the
direct distance, and the triangle inequality excludes cheaper composites.

This is a genuine existence theorem, but the full orbit is dense on one Bloch
circle. The inverse coordinate map is discontinuous; the nine projectors are
not exclusive tests; worst one-shot false acceptance is 0.9025; and at
infidelity tolerance \(10^{-3}\), 30.6% of ordered pairs acquire a shorter
alias. The qubit construction proves that exact hodological dimension can
exceed robust physical dimension. It does not yet explain stable perceived
space.

## A robust two-phase qutrit chart

A qutrit has two independent relative phases. With

\[
|\psi_0\rangle=
\sqrt{3/8}|0\rangle+\tfrac12|1\rangle+\sqrt{3/8}|2\rangle,
\]

and diagonal generators

\[
A=\operatorname{diag}(0,1,0),\qquad
B=\operatorname{diag}(0,\tfrac12,1),
\]

their Fubini--Study covariance metric is exactly isotropic:

\[
g=\frac{3}{16}I_2.
\]

Choosing an order-11 phase torus gives 121 distinct nonorthogonal physical
states in a three-dimensional Hilbert space. The nine goals indexed by
\((x,y)\in\{0,1,2\}^2\) form a local chart without wraparound. The same
unit-cost retry construction yields an exact Euclidean Bellman matrix:
its maximum analytic residual is \(8.9\times10^{-16}\), its Schoenberg Gram
matrix has rank two, and MDS reconstruction error is
\(1.3\times10^{-15}\). This is the strongest current exact benchmark.

The success probabilities still encode the Euclidean norm by design. What has
been derived is the compatibility of a low-dimensional quantum phase manifold,
sequence-labelled control, and exact planar hodological cost. What has not been
derived is why generic physics should choose the inverse-distance retry law.

## Fully operational and skeptical controls

The qutrit Hesse SIC supplies nine nonorthogonal outcome goals with no external
place probe. Weyl controls give the exact Bellman solution

\[
V_g(s)=6+d_{\mathbb Z_3^2}(s,g).
\]

This is a homogeneous two-generator quantum control space, but it is a torus,
not an open Euclidean plane; its planar classical-MDS stress is 0.383. A second
qutrit phase POVM with goals requiring one, two, or three consecutive target
outcomes retains 0.996 correlation with control displacement while baseline
difficulty grows from 5.59 to 72.65 interventions.

A qubit tangent-plane search finds an excellent approximation
(correlation 0.990, 2D stress 0.0074), but flatness competes with distinguishability
and translation covariance. A null-qubit control retains a perfect nine-state
goal automaton while every quantum trace distance is zero. Finally, projective
\(X/Z\) sequence goals have a nonadditive cost residual of one, whereas matched
independent coins are exactly additive. Quantum backaction can therefore warp
a history goal geometry even when it does not supply the base.

## Revised necessary-and-sufficient program

For a finite candidate, exact ordinary-space hodology requires three jointly
necessary and sufficient mathematical conditions:

1. the desired costs solve the proper stochastic-shortest-path Bellman
   equations of the augmented predictive process;
2. the symmetric cost matrix is a metric;
3. its Schoenberg matrix
   \(-\tfrac12J(D\circ D)J\) is positive semidefinite of rank at most two or
   three.

Those conditions classify a distance matrix, not an emergent space. A physical
identification additionally requires local and approximately homogeneous
action fields, goal covariance, operational localization, trajectory
coherence, low external-memory provenance, and robustness to unknown starts
and hidden disturbances. Perfect one-shot recognition of nine mutually
exclusive physical places still requires \(d\ge9\); histories evade that bound
only by changing the premise.

The next frontier is consequently sharper: learn the qutrit phase chart from
weak outcomes without access to the displacement counter, perturb the exact
model away from covariance, and measure when Bellman, Schoenberg, localization,
and memory-provenance conditions fail. Only after those tests should the phase
base be coupled to internal goal fibers and used for curvature or connection
experiments.
