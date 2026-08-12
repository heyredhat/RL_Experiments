# Informative qutrit actions: geometry from opaque controlled predictions

## Executive result

This miniproject constructs a qutrit model in which the same opaque actions
both move and report. The learner receives no preparation coordinate and no
displacement label. From observed triples

\[
 (\hbox{previous outcome token},\ \hbox{opaque action ID},\
 \hbox{next outcome token}),
\]

it recovers five permutations: one identity and four commuting order-three
generators. They generate a transitive group of order nine, operationally
identifying a \(\mathbb Z_3\times\mathbb Z_3\) translation space up to relabeling
and choice of axes.

The construction is exactly solvable, but its strongest metric conclusion is
negative. Each integrated action has Kraus
branches

\[
 K^{(a)}_o=\frac1{\sqrt3}\Pi_oU_a,
\]

where \(\{\Pi_o\}\) is the qutrit Hesse SIC and \(U_a\) is the identity or one
of four local Weyl translations. Its outcome is immediately informative about
the pre-action predictive state, and its branch state is \(\Pi_o\), so future
statistics retain an operational meaning. The controlled kernel is learned to
mean absolute error about 0.005 per entry from 2,000 samples per source token
and action; all five permutations are recovered exactly despite independent
shuffling of action and outcome names.

For the goal “report token \(g\),” Bellman values are exactly 4 from the goal
or a nearest neighbor and 5 from a toroidal diagonal. A non-goal neighbor thus
has zero baseline-subtracted separation from the goal: this is **not a metric
and not an exact torus hodology**. The collapse is a
real consequence of integrating movement with stochastic measurement. In the
separated move-then-report comparison, the exact solution is

\[
 V_g(s)=6+d_T(s,g).
\]

Thus opaque action statistics robustly recover two-dimensional translation
topology, while fully integrating movement and informative reporting destroys
identity of indiscernibles in this minimal symmetric family.

## Reproduction

Run from this directory:

```bash
python -m unittest -v test_informative_qutrit.py
MPLBACKEND=Agg python informative_qutrit.py --episodes 2000 --seed 20260812
```

The production bundle contains 34,000 navigation episodes, 20,000 training
and 8,000 held-out five-step prediction sequences, exact matrix calculations,
two figures, and nine unit tests. Every stochastic artifact is determined by
the recorded seed.

## 1. The operational criterion

Assigning matrices the names “north,” “east,” and so on does not make those
meanings emergent. A stronger criterion is:

> Two actions have translation-like meaning only when their meaning can be
> reconstructed from how they transform observable future probability laws.

Let histories \(h,h'\) be predictively equivalent when every future action
sequence induces the same distribution over future outcomes. An operational
state is an equivalence class \([h]\). An opaque action \(a\) is a translation
when it induces a closed permutation \(T_a\) of these classes. Two independent
translation meanings require, at minimum, a group or local semigroup generated
by two nonredundant transformations.

This definition separates three quantities:

1. **Immediate information:** \(I(S;O_t\mid A_t=a)\), information in the
   action's present outcome about the pre-action predictive state.
2. **Future predictive effect:** the difference between
   \(p(O_{t+1}\mid O_t,a,A_{t+1})\) and the corresponding no-motion law.
3. **Geometric closure:** whether these future transformations close into a
   low-dimensional translation group rather than merely being distinguishable.

A deterministic unitary has zero immediate outcome information but can have a
large future predictive effect. A measure-and-prepare action can have both.

## 2. Exact integrated Hesse construction

Let \(\omega=e^{2\pi i/3}\), and define the qutrit Weyl pair

\[
 X|j\rangle=|j+1\bmod3\rangle,
 \qquad Z|j\rangle=\omega^j|j\rangle.
\]

Starting from \(|\psi_{00}\rangle=(0,1,-1)^T/\sqrt2\), the nine rays

\[
 |\psi_{mn}\rangle=X^mZ^n|\psi_{00}\rangle
\]

form the Hesse SIC. Their projectors satisfy

\[
 \frac13\sum_o\Pi_o=I,
 \qquad \operatorname{tr}(\Pi_o\Pi_s)=
 \begin{cases}1&o=s,\\1/4&o\ne s.\end{cases}
\]

The action catalog uses

\[
 U_a\in\{I,X,X^\dagger,Z,Z^\dagger\}.
\]

Action \(a\) is not a unitary followed by a separately labeled sensor. It is
one nine-outcome instrument with branches

\[
 K^{(a)}_o=\frac1{\sqrt3}\Pi_oU_a.
\]

Completeness follows immediately:

\[
 \sum_oK_o^{(a)\dagger}K_o^{(a)}
 =U_a^\dagger\left(\frac13\sum_o\Pi_o\right)U_a=I.
\]

If the pre-action predictive state is \(\Pi_s\), the probability of outcome
\(o\) is

\[
 P_a(o\mid s)=\frac13
 \operatorname{tr}(\Pi_oU_a\Pi_sU_a^\dagger)
 =\begin{cases}
 1/3,&o=T_a(s),\\
 1/12,&o\ne T_a(s).
 \end{cases}
\]

Conditional on any possible outcome, the normalized branch state is exactly
\(\Pi_o\). Consequently the last observed token is a sufficient predictive
state, and the directly observed action kernel is exactly \(P_a\). This is a
genuine quantum instrument: it is informative, stochastic, disturbing, and
state preparing.

### Immediate information

With a uniform prior over the nine predictive classes, every outcome is
uniform. The information in one integrated action is

\[
 I(S;O\mid a)=\log_2 9-
 H\left(\frac13,\underbrace{\frac1{12},\ldots,
 \frac1{12}}_{8}\right)
 =0.251629\ \text{bits}.
\]

Every translation has the same information because it only permutes the
likelihood rows.

## 3. Learning the two meanings without coordinate labels

Outcome tokens and action IDs are independently shuffled before training. For
each opaque action, the learner estimates

\[
 \widehat P_a(o'\mid o)
\]

from sequences. It assigns each token \(o\) to the unique maximum-probability
successor

\[
 \widehat T_a(o)=\arg\max_{o'}\widehat P_a(o'\mid o).
\]

No hidden preparation state appears in this rule. With 2,000 samples per
source token/action, production recovered all 45 token images correctly. The
per-entry kernel MAE ranged from 0.00505 to 0.00579.

The inferred permutations themselves reveal the structure:

- exactly one action has order one;
- the other four have order three;
- all generators commute as permutations;
- their generated group has order nine;
- the orbit of any token has size nine.

An abstract group of order nine with the observed noncyclic generator pattern
is \(\mathbb Z_3^2\). Selecting which inverse pair is called \(x\), which is
called \(y\), their signs, and the origin token is gauge. This residual
ambiguity is desirable: ordinary coordinate names should not be physically
privileged.

The Cayley graph learned from those permutations has four neighbors per node
and 3x3 toroidal topology. Its graph metric is not a literal two-dimensional
Euclidean distance matrix: classical 2D MDS stress is 0.383. “Two-dimensional”
here means two independent local translations and group topology, not a flat
Euclidean embedding.

## 4. Exact Bellman solutions

### 4.1 Integrated report-and-move actions

Let goal \(g\) terminate the episode whenever an integrated action reports
outcome \(g\). Every action costs one. Since a failed action prepares the
reported class, the stochastic-shortest-path equation is

\[
 V_g(s)=\min_a\left[
 1+\sum_{o\ne g}P_a(o\mid s)V_g(o)
 \right].
\]

Translation symmetry reduces this to three potential classes: current token at
the goal, at a graph neighbor, or at distance two. Direct substitution gives

\[
 V_0=V_1=4,\qquad V_2=5.
\]

For the goal or a neighbor, choose an action aligning the translated
pre-measurement ray with \(g\). The goal is reported with probability \(1/3\),
while each other token has probability \(1/12\):

\[
 1+\frac{4V_1+4V_2}{12}
 =1+\frac{16+20}{12}=4.
\]

From a diagonal, one local translation reaches a neighbor before reporting;
the peaked outcome is then a neighbor. Substitution yields

\[
 1+\frac13V_1+\frac3{12}V_1+\frac4{12}V_2=5.
\]

The numerical Bellman solver converged independently to these values with
errors below \(10^{-12}\). If diagonal entries are discarded, the two
remaining off-diagonal shells correlate perfectly with word distance. That
statistic is misleading: nearest-neighbor value is compressed to the same
value as the goal. Identity of indiscernibles fails, so the integrated value
matrix is not a metric and does not reproduce \(c+d_T\).

This corrects a tempting but false shortcut: treating an expected three trials
at an aligned SIC outcome as an additive “report baseline” ignores that failed
reports prepare other states. Backaction changes the Bellman equations.

### 4.2 Weak integrated instruments: a small no-go result

The sharp SIC action was weakened while retaining covariance, common unit cost,
and outcome-state preparation:

\[
 P_a^{(\eta)}(o\mid s)=\eta P_a(o\mid s)+(1-\eta)/9.
\]

This is implemented by a measure-and-prepare refinement of effects
\(\eta\Pi_o/3+(1-\eta)I/9\). It interpolates from a state-independent random
reset to the sharp Hesse action.

| \(\eta\) | self value | edge value | diagonal value |
|---:|---:|---:|---:|
| 0.0 | 9.000 | 9.000 | 9.000 |
| 0.2 | 7.438 | 7.438 | 7.810 |
| 0.4 | 6.250 | 6.250 | 6.875 |
| 0.6 | 5.325 | 5.325 | 6.124 |
| 0.8 | 4.592 | 4.592 | 5.510 |
| 1.0 | 4.000 | 4.000 | 5.000 |

No strength restores a self--edge gap. From a neighbor, a local integrated
action aligns the premeasurement state with the goal and then uses exactly the
same report kernel as the identity integrated action at the goal. Equal action
cost therefore forces equal continuation value. Weakening changes their common
value and separates the diagonal shell, but cannot restore identity of
indiscernibles.

Within this symmetric grammar, fully integrated informative actions recover
topology but cannot define a metric. One must change at least one assumption:
separate arrival from reporting, distinguish movement and report costs, retain
nondestructive goal memory, or change the terminal semantics.

### 4.3 Separated control and report comparison

If the four translations are unit-cost single-Kraus actions and SIC reporting
is a separate unit-cost action, the exact solution is instead

\[
 V_g(s)=6+d_T(s,g).
\]

The diagonal baseline is six and excess costs are exactly 0, 1, and 2. This
model has a cleaner scalar word metric but weaker action integration: moves
have no immediate observed information. The contrast proves that topology is
more robust than the scalar hodological cost under changes to action grammar.

## 5. Axis-instrument comparison

The phase-grid orbit

\[
 |\phi_{xy}\rangle=
 (|0\rangle+\omega^x|1\rangle+\omega^y|2\rangle)/\sqrt3
\]

supports a lower-outcome alternative. Two three-outcome instruments probe the
\(0\!-!1\) and \(0\!-!2\) relative phases. For the first axis, define

\[
 E^x_a(\eta)=\eta\left[
 \frac23|\chi_a^{01}\rangle\langle\chi_a^{01}|+
 \frac13|2\rangle\langle2|
 \right]+(1-\eta)\frac I3,
\]

where \(|\chi_a^{01}\rangle=(|0\rangle+\omega^a|1\rangle)/\sqrt2\). The second
axis is analogous on levels 0 and 2. Positive-square-root Kraus operators give
Lüders instruments.

Their exact outcome laws are

\[
 p(a\mid x,y,X)=
 \begin{cases}
 (3+2\eta)/9,&a=x,\\
 (3-\eta)/9,&a\ne x,
 \end{cases}
\]

and similarly \(p(b\mid x,y,Y)=p(b\mid y)\). Thus every \(\eta>0\) yields nine
distinct joint signatures, while \(\eta=0\) yields one.

At \(\eta=0.6\), the combined predictive Jensen--Shannon distance is exactly
proportional to square-root Hamming distance on the two coordinates
(correlation 1 to numerical precision). Its centered Euclidean rank is four,
and 2D MDS stress is 0.408. Once again, two independently factorized meanings
do not imply that the finite global distance matrix embeds in a plane.

The strength scan displays the information--disturbance tradeoff. Joint mutual
information rises from zero to 0.14944 bits between \(\eta=0\) and 1, while
mean survival fidelity falls from 1 to 0.75871. Neither maximizing information
nor minimizing disturbance alone selects the best control policy.

## 6. Held-out prediction and navigation

### Outcome-sequence prediction

For each strength, 5,000 random five-step sequences train empirical marginal
and last-action/outcome models; 2,000 independent sequences are held out. An
exact quantum filter carrying all nine conditional branch states provides the
oracle predictive state.

| \(\eta\) | marginal NLL | learned last-event NLL | quantum-filter NLL |
|---:|---:|---:|---:|
| 0.0 | 1.5854 | 1.5862 | 1.5850 |
| 0.3 | 1.5848 | 1.5852 | 1.5834 |
| 0.6 | 1.5852 | 1.5759 | 1.5647 |
| 1.0 | 1.5856 | 1.5207 | 1.5019 |

The theoretical null entropy is \(\log_2 3=1.58496\) bits. At sharp strength,
history-aware quantum prediction saves 0.0837 bits/outcome relative to the
marginal and 0.0187 relative to the learned last-event table. The latter gap
shows that the Lüders axis observation alone is not a sufficient predictive
state.

### Hidden-start goal navigation

As a comparison rather than the headline construction, a Bayesian filter uses
the axis instruments to localize a uniformly hidden phase-grid preparation.
The four unitary action IDs are shuffled, and their meanings are reconstructed
from covariance of future outcome statistics before navigation.

| \(\eta\) | senses | label success ± SE | target fidelity | mean cost |
|---:|---:|---:|---:|---:|
| 0.0 | 0 | 0.111 ± 0.007 | 0.334 | 2.34 |
| 0.0 | 6 | 0.109 ± 0.007 | 0.331 | 8.33 |
| 0.3 | 2 | 0.152 ± 0.008 | 0.479 | 4.35 |
| 0.3 | 6 | 0.167 ± 0.008 | 0.518 | 8.35 |
| 0.6 | 2 | 0.219 ± 0.009 | 0.640 | 4.33 |
| 0.6 | 6 | 0.247 ± 0.010 | 0.680 | 8.32 |
| 1.0 | 2 | 0.298 ± 0.010 | 0.878 | 4.31 |
| 1.0 | 6 | 0.299 ± 0.010 | 0.414 | 8.33 |
| 0.6 | known start, 0 senses | 1.000 | 1.000 | 2.34 |

Null sensing stays at chance \(1/9\). Known-start control is exact. Strong
oversensing preserves only modest information about the original preparation
while damaging present-state navigation, reproducing the information/backaction
distinction from the earlier localization study.

## 7. Dialectical controls

### Random-unitary future-probe benchmark

A one-outcome unitary action has immediate information exactly zero. Both Weyl
translations and four Haar-random unitaries nevertheless have distinct future
SIC kernels. The Weyl future effect has mean total-variation displacement 0.25
from identity; the sampled Haar controls range from 0.386 to 0.404.

Thus distinguishability of action effects is not enough. Haar kernels are
identifiable, but their best fits to the phase-axis translation covariance have
residual at least 0.305 and do not close on the nine phase states. Only the Weyl
actions give zero residual and the required translation group.

### Null sensing

At \(\eta=0\), all axis effects equal \(I/3\). Immediate information is zero,
all quantum predictive signatures coincide, and centered signature rank is
zero. Navigation remains at chance regardless of added observations.

### Permuted labels

Reordering opaque action IDs merely reorders the recovered transformations.
Independently permuting all SIC outcome tokens conjugates every learned
permutation but leaves order, commutation, group size, orbit size, and Cayley
distance invariant. Geometry is therefore recovered only up to coordinate
gauge, as it should be.

### External automaton false positive

A classical nine-node \(3\times3\) counter attached to the null qutrit retains
an exact hand-coded graph while quantum predictive rank remains zero. It fails
the criterion that action meanings act nontrivially on observed quantum future
laws. This control prevents goal-progress memory from being misreported as
low-dimensional physical geometry.

## 8. What has and has not been established

Established:

1. A qutrit can carry nine operational predictive classes without nine
   orthogonal basis states.
2. A single family of integrated, informative quantum actions can reveal a
   two-generator translation group using only opaque action/outcome sequences.
3. The action group and its Cayley topology are invariant under label gauge.
4. Immediate information, future predictive effect, and group closure are
   distinct diagnostics.
5. Exact Bellman analysis shows that integrated backaction destroys identity
   of indiscernibles while retaining two-dimensional action topology.

Not established:

1. The toroidal word metric is not an exact 2D Euclidean distance matrix.
2. The Hesse predictive signatures have centered rank eight, not two. Their
   geometry comes from controlled automorphisms, not static ambient distance.
3. The model assumes a finite homogeneous SIC orbit and exact covariance.
4. A learned neural agent has not yet discovered the predictive quotient from
   raw unsegmented histories.
5. The choice of a Hesse SIC and Weyl controls is engineered; a general
   necessity theorem remains open.

## 9. Toward necessary and sufficient conditions

This example suggests an operational finite-space theorem schema. Let an
instrument family induce a finite set \(Q\) of predictive equivalence classes.
A spatial interpretation with translation group \(G\) is supported if:

1. **observability:** classes in \(Q\) have distinct future test statistics;
2. **closure:** each local action induces a well-defined stochastic map or
   permutation on \(Q\);
3. **homogeneity:** these maps act transitively on the candidate base;
4. **independent generators:** the action semigroup contains the desired number
   of locally independent generators;
5. **goal covariance:** goals are translates of one operational template;
6. **cost compatibility:** Bellman values depend only on relational group
   displacement, perhaps after an explicitly justified baseline or monotone
   transformation;
7. **memory ablation:** the structure disappears when quantum future laws are
   replaced by null laws while external automata remain fixed.

The first five and seventh hold exactly here. Cost compatibility fails as a
metric: integrated actions collapse distance zero and one throughout the weak
interpolation. A stronger sufficient condition for an exact metric must prevent actions
from combining a spatial step and terminal report in a way that changes local
cost ordering.

## 10. Recommended next experiments

1. Learn predictive equivalence classes from long raw strings using spectral
   predictive-state reconstruction or a recurrent model, without treating the
   last SIC outcome as a supplied state variable.
2. Replace exact SIC preparation by weak full-rank instruments. Determine when
   finite causal states become a continuous belief fiber and measure sample
   complexity for recovering the action group.
3. Optimize integrated Kraus instruments subject to immediate-information,
   disturbance, group-closure, and Bellman-metric objectives.
4. Search for an instrument grammar whose integrated Bellman values preserve
   strict \(0<1<2\) torus ordering, or prove that rank-one covariant
   measure-and-prepare actions necessarily flatten the first shell.
5. Break translation symmetry slightly and infer local generators chart by
   chart. The resulting cocycle mismatch is a natural precursor of discrete
   curvature or holonomy.
6. Attach translated internal sequence goals and test whether their predictive
   state forms a nontrivial fiber over the recovered opaque action group.

## Artifacts

- `informative_qutrit.py`: all exact constructions, simulation, diagnostics,
  Bellman solvers, CSV generation, and plotting.
- `test_informative_qutrit.py`: nine tests of instruments, likelihoods,
  predictive classes, opaque group recovery, label gauge, and both exact
  Bellman solutions.
- `results/hesse_measure_prepare_kernel.csv`: exact integrated controlled base
  kernel for identity action.
- `results/opaque_hesse_action_learning.csv`: learned opaque action kernels and
  recovered permutation diagnostics.
- `results/hesse_integrated_action_cost.csv` and
  `results/hesse_bellman_cost.csv`: integrated and separated exact values.
- `results/integrated_weak_scan.csv`: the strength interpolation and persistent
  self--edge collapse.
- `results/candidate_instruments.csv` and
  `results/information_disturbance.csv`: broad sensor comparison and strength
  scan.
- `results/heldout_prediction.csv`: held-out sequence NLL.
- `results/navigation.csv`: hidden-start controls with Monte Carlo errors.
- `results/action_discovery.csv` and `results/future_effect_controls.csv`:
  covariance and immediate/future-information controls.
- `figures/candidate_search_and_geometry.png` and
  `figures/action_meanings_navigation_prediction.png`: main visual summaries.
