# Informative group-covariant qutrit actions with retained memory

## Executive result

A small change in terminal semantics corrects the earlier apparent integrated
Hesse no-go. If a goal means **the predictive state is at token \(g\)**, then
\(V_g(g)=0\). The unchanged rank-one Hesse instrument has exact Bellman shells

\[
 V_{\rm self}=0,\qquad V_{\rm edge}=4,
 \qquad V_{\rm diagonal}=5.
\]

This is a valid nondegenerate \(\mathbb Z_3^2\) torus metric. The earlier
\(4,4,5\) result answered a different question—how long until the instrument
reports \(g\) again—even when the system was already at \(g\). The correction
changes the goal semantics, not the physics.

The new retained-memory family improves the metric while keeping outcomes
informative. Every action has an observed full-operator-rank unitary branch that retains
and translates the current Hesse state, plus observed reset branches that
measure and prepare Hesse states. A deterministic 380-point search over memory
probability and report sharpness finds:

- a selected high-memory torus candidate \((\mu,\xi)=(0.8,1)\) with immediate
  information 0.050326 bits, exact strict shells \(0,1.234568,2.283951\), and
  only 3.33% relative distortion after scaling the torus word metric;
- an analytic candidate
  \(\mu=(4\sqrt2-5)/3=0.218951\), \(\xi=1\), whose edge and diagonal costs have
  the exact ratio \(\sqrt2\), so every elementary four-state cell is an exact
  Euclidean square while the global nine-state topology remains toroidal;
- the memory-only limit \(\mu=1\), which is the exact \(0,1,2\) torus word
  metric but has zero information;
- the report-only limit \(\mu=0\), which has 0.251629 bits and the exact
  \(0,4,5\) state-hitting metric.

Thus the state-hitting problem admits a continuous, exactly soluble
information--metric tradeoff. A qutrit can have one integrated action family
with positive immediate information, opaque action identifiability, exact
group covariance, exact finite predictive state, and a valid nondegenerate
Bellman metric.

The remaining limitation is important: after outcomes are discarded,
equal-length words with the same group displacement have identical
nonselective endpoints, but cancelling detours leave different amounts of
ensemble decoherence. At \(\mu=0.8\), the all-length nonselective residual is
about 0.319. This is a length-dependent coarse-grained ensemble effect, not an
additional predictive coordinate for the fully outcome-conditioned agent.

## 1. Exact Hesse predictive states

Let \(X,Z\) be the qutrit Weyl pair and let

\[
 |\psi_{mn}\rangle=X^mZ^n(0,1,-1)^T/\sqrt2,
 \qquad m,n\in\mathbb Z_3.
\]

Their projectors \(\Pi_s\) form the Hesse SIC:

\[
 {1\over3}\sum_o\Pi_o=I,
 \qquad
 \operatorname{tr}(\Pi_o\Pi_s)=
 \begin{cases}1&o=s,\\1/4&o\ne s.\end{cases}
\]

Five opaque integrated actions correspond, only in the privileged audit, to
\(a\in\{0,\pm e_x,\pm e_y\}\). Their unitary parts \(U_a\) permute the nine
Hesse rays as the translation \(T_a\). Token names and coordinate orientation
are gauge.

## 2. Full-operator-rank retained-memory instrument

The model has one observed outcome `memory` and nine observed outcomes
`reset:o`. The memory branch is

\[
 K_{a,r}=\sqrt\mu\,U_a.
\]

For \(\mu>0\), it has full operator rank but branch Choi rank one, is not
measure-and-prepare, and maps
\(\Pi_s\mapsto\Pi_{T_a(s)}\) exactly. The reset branch is the CP map

\[
 \mathcal E^a_o(\rho)
 =(1-\mu)\operatorname{tr}
 \left[F_o^{(\xi)}U_a\rho U_a^\dagger\right]\Pi_o,
\]

where

\[
 F_o^{(\xi)}=\xi{\Pi_o\over3}+(1-\xi){I\over9}.
\]

For \(\mu<1\) and \(\xi<1\), a reset branch has Kraus/Choi rank three. At the
selected sharp point \(\xi=1\), each reset branch has Choi rank one. The implementation
constructs it with

\[
 K_{a,o,j}=\sqrt{1-\mu}\,
 |\psi_o\rangle\langle j|\sqrt{F_o^{(\xi)}}U_a,
 \qquad j=0,1,2.
\]

Summing every observed and unobserved branch gives identity. The production
maximum Kraus completeness residual is recorded in `results/summary.json`.

For any Hesse input, every conditional branch again produces a Hesse state:
the memory outcome produces its translate, and reset outcome \(o\) produces
\(\Pi_o\). The predictive state space is therefore exactly finite and the
observed history determines the current predictive class, starting from any
known/anchored class. This is retained memory rather than a hidden external
counter: both kinds of state transition are physical CP branches.

## 3. Immediate information

The memory/reset mode is selected independently of state. Conditioned on a
sharp reset, the Hesse report contains

\[
 I_{\rm Hesse}=0.2516291674\ \text{bits}.
\]

Therefore, at \(\xi=1\),

\[
 I(S;O\mid A)=(1-\mu)I_{\rm Hesse}.
\]

More generally the noisy reset kernel is

\[
 P^{(\xi)}_a(o\mid s)
 =\xi P^{\rm Hesse}_a(o\mid s)+{1-\xi\over9},
\]

and its mutual information is calculated exactly from the finite table.
Information vanishes at \(\xi=0\) or \(\mu=1\), distinguishing null-report and
memory-only controls from the selected integrated candidate.

## 4. Exact state-hitting Bellman solution

The task terminates whenever the current predictive class equals goal \(g\):

\[
 V_g(g)=0.
\]

Every action costs one. Away from the goal,

\[
 V_g(s)=1+\min_a\sum_{o,s'}
 P(o,s'\mid s,a)V_g(s').
\]

Translation covariance reduces the nine values to an edge value \(E\) and a
toroidal diagonal value \(D\). Let

\[
 c={(1-\mu)(4-\xi)\over9},\quad
 h=\mu+{(1-\mu)(16+5\xi)\over36},\quad
 A=1-c.
\]

Moving an edge toward the goal and a diagonal toward an edge gives

\[
 E=1+c(E+D),
 \qquad
 D=1+hE+cD.
\]

Solving,

\[
 \boxed{E={1\over A^2-ch}},
 \qquad
 \boxed{D={1+hE\over A}}.
\]

Full nine-state Bellman iteration independently matches these expressions to
below numerical tolerance for every reported candidate. The resulting matrix
is symmetric, has a zero diagonal and strictly positive off-diagonal entries,
and satisfies all triangle inequalities throughout the searched parameter
grid.

For the sharp family \(\xi=1\), these simplify remarkably:

\[
 E={4\over(1+\mu)^2},
 \qquad
 D={3\mu+5\over(1+\mu)^2}.
\]

Hence

\[
 D-E={3\mu+1\over(1+\mu)^2}>0,
 \qquad
 D\le2E\quad(0\le\mu\le1).
\]

Identity of indiscernibles and the only nontrivial torus triangle inequality
hold analytically. At \(\mu=0\), \((E,D)=(4,5)\). At \(\mu=1\),
\((E,D)=(1,2)\).

## 5. Two selected solutions

### 5.1 High-memory torus compromise

The deterministic stated selection rule requires:

- \(\mu>0\);
- immediate mutual information at least 0.05 bits;
- a valid metric;
- minimum scaled distortion to the \(0,1,2\) torus word metric.

On the 20-by-19 grid, this selects \((\mu,\xi)=(0.8,1)\). It has

\[
 I=0.0503258\ \text{bits},\quad
 E=1.2345679,\quad D=2.2839506.
\]

The shell margin is 1.049383. After the best common scale, relative torus RMSE
is 0.033287. Training only on the local edge cost and predicting the diagonal
as \(2E\) gives a held-out radial-shell error of 8.11%; this reports the
remaining nonadditivity rather than hiding it in a perfect two-shell
correlation.

The primary action-learning audit is operational. Each trial begins with an
opaque Hesse anchor token that physically prepares its ray. An opaque action is
applied; trials with the observed `memory` outcome are retained; then the same
sharp Hesse reporter is applied as a common future probe. The learner receives
only `(opaque anchor, opaque action, opaque future report)` triples. It never
receives the translated state or a successor-state label. With 500 attempts per
anchor/action, 17,982 observed memory events recover five bijections, one
identity and four order-three maps. They commute, generate a transitive group of
order nine, and agree 100% with the hidden physical maps after offline scoring
in the persistent token gauge.

A **separate oracle-labelled benchmark** retains the previous joint
`(outcome,next_state)` table. It is trained on 400 samples per latent
state/action and evaluated on 2,000 independent length-eight strings. Exact,
learned, and latent-state-marginal NLL are respectively 1.3132, 1.3989, and
3.8964 bits per joint event. These numbers validate finite-state estimation
when latent Hesse labels are supplied; they are not evidence of observable-only
string learning.

### 5.2 Exact local Euclidean square

An elementary square is Euclidean exactly when \(D/E=\sqrt2\). In the sharp
family,

\[
 {D\over E}={3\mu+5\over4}.
\]

Therefore

\[
 \boxed{\mu_\square={4\sqrt2-5\over3}=0.2189514\ldots}
\]

makes every four-state cell generated by two independent translations an exact
Euclidean square. It retains a full-rank memory branch and has positive
immediate information

\[
 I=(1-\mu_\square)I_{\rm Hesse}\approx0.1965\ \text{bits}.
\]

This is an exact local result, not a numerical fit. It does not make all nine
toroidal distances an open planar \(3\times3\) grid. Periodic wraparound makes
opposite boundaries adjacent, so the full finite torus has a different global
topology.

## 6. Nonselective closure and coarse-grained noise age

The nonselective channel has the covariant form

\[
 \Phi_a=\mathcal U_a\circ C,
\]

where the same translation-covariant noise channel \(C\) is used for every
action. Thus all action orders with the same displacement and the same length
have identical nonselective future states. For the selected candidate the
length-matched closure residual through length four is at numerical precision.

Words of different lengths acquire different powers \(C^L\). A path with a
cancelling loop therefore need not agree with a shorter word having the same
net displacement. The selected candidate's all-length trace-distance closure
residual is approximately 0.319.

This diagnostic deliberately discards the observed branch record. For the
fully observing agent, every conditional branch ends on a pure Hesse ray and
the history determines that class; noise age is therefore not an extra
predictive coordinate in this exact model. Under explicit outcome
coarse-graining, displacement may be treated as a toroidal base and
nonselective noise age as an ensemble fiber. A genuinely predictive candidate
fiber is instead detected by the separate Lüders experiment.

## 7. Geometry diagnostics and open-flat limitation

Every candidate has exact translation-invariant shells and correlation one
with torus graph distance whenever \(E<D\). Correlation is therefore
insufficient. The report additionally records optimal scaling distortion,
diagonal additivity error, triangle inequalities, and shell margin.

Schoenberg diagnostics ask whether the full nine-state metric embeds in an
ordinary Euclidean space. The \(3\times3\) torus does not embed isometrically
in a plane. The selected candidate consequently has nonzero negative
eigenmass and 2D MDS stress; exact values are in `candidate_diagnostics.csv`.
Its correlation with an open \(3\times3\) Euclidean grid is only about 0.528
because toroidal boundary neighbors are far apart in the open chart.

For the selected torus candidate, Schoenberg negative eigenmass is 0.1957,
the centered Gram matrix has four positive dimensions, and a deterministic
classical two-dimensional MDS realization has stress 0.3731. The exact local
square has zero negative eigenmass on the four-point cell and rank exactly two,
but the full nine-token matrix has 2D stress 0.4082. These diagnostics prevent
the exact cell result from being overextended into a global planar claim.

The exact \(2\times2\) square at \(\mu_\square\) is the largest honest
open-flat claim from this \(m=3\) construction. A genuine open \(3\times3\)
patch requires a larger group \(\mathbb Z_m^2\), \(m\ge5\), followed by a patch
whose operative radius does not see periodic wraparound.

## 8. Ablations

- **Corrected Hesse baseline, \(\mu=0,\xi=1\):** maximum information,
  measure-and-prepare, exact valid shells \(0,4,5\). No retained branch.
- **Memory-only, \(\mu=1\):** exact torus word metric \(0,1,2\), exact closure,
  but zero immediate information.
- **Null report, \(\mu=0.8,\xi=0\):** retained movement and a valid metric,
  but reset outcomes carry zero state information.
- **Weak report, \(\mu=0.8,\xi=0.5\):** positive but reduced information and a
  higher metric distortion than the selected sharp reporter.
- **External coordinates:** used only to score the recovered opaque
  permutations and plot the torus; the learner uses observed predictive tokens
  and transition counts.

These controls show that metric validity alone does not establish informative
semantics, while information alone does not optimize the metric.

## 9. What is established

Established exactly:

1. Correct state-hitting semantics makes integrated Hesse costs \(0,4,5\), a
   valid nondegenerate torus metric.
2. A covariant instrument with a full-rank memory branch and informative reset
   branches remains closed on nine finite Hesse predictive states.
3. The sharp family's Bellman shells have closed forms and satisfy strict
   identity of indiscernibles and all triangle inequalities for every
   \(\mu\in[0,1]\).
4. Positive immediate information and retained quantum memory coexist in one
   integrated action family.
5. One analytic memory probability gives exact Euclidean elementary squares.

Established numerically:

1. The constrained grid search selects a high-memory, positive-information
   candidate with 3.33% scaled torus distortion.
2. Opaque action permutations are recovered from observable
   anchor/action/future-report triples without successor-state labels.
3. Oracle-labelled transition tables generalize to held-out length-eight
   strings; this is a separate benchmark.
4. Kraus completeness, covariance, Bellman, metric, and restricted
   nonselective equal-length closure residuals are at floating-point scale.

Not established:

1. A predictive internal fiber has not been learned; the exact family's
   detour residual is for its outcome-discarded ensemble channel.
2. The full nine-state torus is not a flat open \(3\times3\) Euclidean patch.
3. The observed memory/reset mode is engineered rather than discovered by an
   unconstrained optimizer.
4. Hidden-start localization and learning a base/fiber quotient from raw
   unsegmented histories remain future work.

## 10. Reproduction and artifacts

Run:

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_search.py
```

Seven focused tests cover the corrected Hesse baseline, analytic Bellman
solution, strict shell ordering, exact local square, Kraus completeness,
covariance, equal-length versus detour closure, and the memory-only control.

Artifacts:

- `results/search_grid.csv`: all 380 parameter pairs and constrained metrics;
- `results/candidate_diagnostics.csv`: six baseline, selected, analytic, and
  ablation candidates;
- `results/path_closure.csv`: equal-length and all-length audits;
- `results/bellman_*.csv`: exact all-pairs state-hitting matrices;
- `results/summary.json`: configuration and headline results;
- `results/figures/covariant_memory_search.png`: Bellman shells,
  information--distortion frontier, embedding diagnostics, and selected metric.

The stochastic opaque-action and held-out-string audits use seed 20260812.
