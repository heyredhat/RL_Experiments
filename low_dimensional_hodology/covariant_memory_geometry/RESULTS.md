# Covariant quantum memory and operational torus geometry

## Executive conclusion

This miniproject resolves the central defect of the earlier informative-action
study in a qualified but substantive way.  A qutrit instrument can have all of
the following at once:

- outcomes that reveal information about the present predictive state;
- opaque actions whose meanings are inferred from observable transition laws;
- exact Weyl covariance and a learned \(\mathbb Z_3^2\) translation topology;
- a nondegenerate state-hitting Bellman metric;
- retained quantum memory in a full-rank physical branch; and
- an exactly Euclidean elementary square at an analytic parameter value.

The global nine-state space is a torus, not an open planar \(3\times3\) patch.
For the exact memory/reset family, a fully observed conditional history closes
on the nine Hesse states. Only the outcome-discarded nonselective channel has
length-dependent ``noise age.'' A separate Lüders family does have latent
future-predictive memory beyond its last token, but the present suffix learner
does not recover it. Thus the evidence supports a toroidal base and a
candidate internal fiber; it does not yet support a learned fiber bundle.

The most important correction is semantic.  The earlier integrated Hesse
calculation assigned a further reporting cost even when the agent was already
at its goal.  That event goal has shells \((4,4,5)\).  The standard
state-hitting goal instead imposes the absorbing boundary \(V_g(g)=0\) and has
the exact shells

\[
  \boxed{(V_{\rm self},V_{\rm edge},V_{\rm diagonal})=(0,4,5).}
\]

Thus the original physical instrument already induces a valid torus metric
under state-hitting semantics.  This changes the interpretation, not the
underlying transition probabilities.

## 1. What must be learned

The agent observes only a controlled history

\[
 h=(a_1,o_1),\ldots,(a_t,o_t).
\]

It does not receive coordinates, density matrices, Kraus operators, or names
such as ``east.''  Histories are operationally equivalent only if every
allowed future action string has the same outcome law after either history.
An emergent spatial claim therefore has several logically independent parts:

1. **predictive separation:** different proposed places have different future
   statistics;
2. **action identification:** opaque controls induce distinguishable,
   compositional transformations of those statistics;
3. **Bellman realizability:** the proposed costs solve the actual stochastic
   shortest-path equations;
4. **metricity:** self-distance is zero, distinct points are separated,
   symmetry holds, and triangle inequalities pass;
5. **geometry/topology:** the recovered object is tested against a plane,
   torus, or other model rather than declared Euclidean from a plot; and
6. **path closure:** histories assigned the same base displacement agree
   predictively, or the residual variable is retained as an explicit fiber.

This project evaluates all six.  A torus is counted as a positive
two-dimensional result because its two periodic generators and local
neighborhood structure can be learned operationally.  It is not counted as a
globally planar Euclidean result.

## 2. Qutrit Hesse--Weyl substrate

Let \(G=\mathbb Z_3^2\), let \(X,Z\) be the qutrit Weyl pair, and define the
nine Hesse SIC rays

\[
 |\psi_{mn}\rangle=X^mZ^n(0,1,-1)^T/\sqrt2,
 \qquad \Pi_{mn}=|\psi_{mn}\rangle\langle\psi_{mn}|.
\]

They satisfy

\[
 {1\over3}\sum_s\Pi_s=I,
 \qquad
 \operatorname{tr}(\Pi_s\Pi_t)=
 \begin{cases}1,&s=t,\\1/4,&s\ne t.\end{cases}
\]

The five controls are identity and the four cardinal Weyl translations.  The
coordinate names are used only by the external audit.  In the learning study,
both action labels and outcome labels are shuffled, so only the relational
permutation algebra is identifiable.

## 3. Exact retained-memory instrument

For memory probability \(\mu\in[0,1]\) and report sharpness
\(\xi\in[0,1]\), action \(a\) has one observed `memory` branch and nine
observed `reset:o` branches.  The memory Kraus operator is

\[
 K_{a,r}=\sqrt\mu\,U_a.
\]

It has full operator rank, physically retains the incoming qutrit state, and
translates every Hesse state.  The reset branch is

\[
 \mathcal E^a_o(\rho)
 =(1-\mu)\operatorname{tr}
   \!\left[F_o^{(\xi)}U_a\rho U_a^\dagger\right]\Pi_o,
 \qquad
 F_o^{(\xi)}=\xi{\Pi_o\over3}+(1-\xi){I\over9}.
\]

For \(\xi<1\), this reset map has Choi rank three.  One explicit realization is

\[
 K_{a,o,j}=\sqrt{1-\mu}\,
 |\psi_o\rangle\langle j|\sqrt{F_o^{(\xi)}}U_a,
 \qquad j=0,1,2.
\]

The sum of all branches is trace preserving.  The maximum numerical
completeness and covariance residuals are respectively
\(1.55\times10^{-15}\) and \(3.19\times10^{-16}\).

This exact finite-state construction should be distinguished from the
rank-two branch family derived in the theory report,

\[
 \mathcal J_o^\lambda(\rho)=
 {\lambda\over3}\Pi_o\rho\Pi_o+{1-\lambda\over9}\rho.
\]

For \(0<\lambda<1\), each observed branch has Choi rank two, has positive
state information, and leaves a nonzero copy of the incoming state in its
posterior.  It is the cleaner model of within-outcome quantum memory, but its
reachable predictive state is no longer the nine-token Hesse set.  It is
therefore a target for a learned predictive-state representation rather than
the source of the closed-form finite Bellman solution below.

## 4. Information is genuine but tunable

For a uniform Hesse input and a sharp reset, the outcome carries

\[
 I_{\rm Hesse}=0.2516291674\ \text{bits}.
\]

The memory/reset mode itself is independent of the input, so at \(\xi=1\),

\[
 I(S;O\mid A)=(1-\mu)I_{\rm Hesse}.
\]

Consequently, the memory-only limit has exact translation but no immediate
information, the report-only limit maximizes immediate information, and every
intermediate point gives a controlled information--memory tradeoff.  This is
not an external coordinate counter: the observable branches are physical CP
maps and their meanings can be recovered from later statistics.

## 5. Exact Bellman derivation

Every intervention costs one.  Reaching predictive state \(g\) terminates the
task:

\[
 V_g(g)=0,
 \qquad
 V_g(s)=1+\min_a\sum_{o,s'}P(o,s'\mid s,a)V_g(s')
 \quad(s\ne g).
\]

Translation and dihedral symmetry reduce the nine values to an edge shell
\(E\) and a toroidal diagonal shell \(D\).  Define

\[
 c={(1-\mu)(4-\xi)\over9},\qquad
 h=\mu+{(1-\mu)(16+5\xi)\over36},\qquad A=1-c.
\]

The optimal edge action aims at the goal; the optimal diagonal action aims at
an adjacent shell.  Sorting their possible successor classes gives

\[
 E=1+c(E+D),\qquad D=1+hE+cD.
\]

Solving this two-equation system yields

\[
 \boxed{E={1\over A^2-ch}},\qquad
 \boxed{D={1+hE\over A}}.
\]

For a sharp report, \(\xi=1\), these reduce to

\[
 \boxed{E={4\over(1+\mu)^2}},\qquad
 \boxed{D={3\mu+5\over(1+\mu)^2}}.
\]

Metricity is now analytic:

\[
 D-E={3\mu+1\over(1+\mu)^2}>0,
 \qquad D\le 2E.
\]

The first inequality gives strict self--edge--diagonal ordering; the second is
the only nontrivial torus triangle inequality.  Full nine-state Bellman
iteration agrees with the formulas to floating-point tolerance.

## 6. Two useful operating points

### High-memory torus

A deterministic stated rule required positive memory, at least 0.05 bits of immediate
information, a valid metric, and minimum scaled error relative to the torus
word metric.  Among 380 candidates it selected

\[
 (\mu,\xi)=(0.8,1),\quad
 I=0.0503258\text{ bits},\quad
 E=1.2345679,\quad D=2.2839506.
\]

The best-scaled torus RMSE is 3.33%. An observable common-probe protocol
recovers all five permutations from 17,982 accepted memory events in 22,500
trials. The learner receives only opaque anchor, action, and future Hesse-report
tokens—never translated-state or successor-state labels. The maps are
bijective and commuting, generate a transitive group of order nine, and score
100% against hidden maps only after learning. A separate oracle-labelled
benchmark gives learned, exact, and state-marginal negative log likelihoods
1.3989, 1.3132, and 3.8964 bits per joint event; those NLL values validate
finite-state estimation conditional on labels and are not part of the
observable-only claim.

### Exactly Euclidean elementary squares

An elementary cell has Euclidean edge/diagonal ratio \(\sqrt2\).  Since

\[
 {D\over E}={3\mu+5\over4},
\]

the analytic choice

\[
 \boxed{\mu_\square={4\sqrt2-5\over3}=0.2189514165\ldots}
\]

makes every four-state elementary cell an exact Euclidean square.  It retains
\(0.196535\) bits of immediate information.  This is an exact local result,
not an exact open \(3\times3\) plane: periodic wraparound is already visible
on a three-by-three torus.

## 7. Why the geometry is a torus, not a plane

The nearest-neighbor relation is recovered from the smaller nonzero Bellman
shell, producing the Cayley graph of two commuting order-three generators.
This is a two-dimensional topological/compositional statement.  It does not
imply that the full distance matrix embeds in \(\mathbb R^2\).

For the corrected Hesse metric \((0,4,5)\), Schoenberg double centering gives
four eigenvalues 17, four eigenvalues \(7/2\), and one zero.  The metric is
Euclidean only in dimension eight.  The selected high-memory torus is not an
exact Euclidean distance matrix in any dimension; its Schoenberg negative
eigenmass is 0.1957 and its 2D MDS stress is 0.3731.  These are expected
global effects of a tiny periodic space, not failures of its learned torus
topology.

The exact local square is therefore the strongest flat statement supported by
this experiment.  Larger \(\mathbb Z_m^2\) phase orbits are needed to obtain
open patches whose radius is small compared with the periodicity scale.

## 8. Nonselective closure and the limits of a fiber claim

For the nonselective channel \(\Phi\), covariance gives

\[
 \Phi_{a_t}\cdots\Phi_{a_1}
 =\Phi^t\operatorname{Ad}_{U_{a_1+\cdots+a_t}}.
\]

Hence the **nonselective**, outcome-discarded channels for same-length words
with the same displacement coincide analytically; a numerical audit from one
Hesse start through length four has residual \(10^{-15}\). Words of different
length can have the same net displacement but different powers of \(\Phi\).
The audited maximum trace-distance residual is 0.31866.

This is the state of an observer who suppresses the available branch record.
For the fully observing agent, every conditional memory or reset branch ends
on a pure Hesse state, so there is no additional noise-age coordinate in this
exact model. Only under outcome coarse-graining may one write

\[
 \text{coarse-grained ensemble state}\simeq
 (\text{toroidal place},\ \text{noise age/purity}).
\]

The separate Lüders experiment supplies stronger evidence for a genuinely
predictive internal coordinate: its same-token posteriors differ in future
laws. Because that difference is detected by an oracle filter and not learned
by the suffix estimator, ``candidate predictive fiber'' is the strongest
current description.

## 9. Opaque learning and skeptical controls

The independent learning strand receives only shuffled action/outcome strings.
For rank-one Hesse measure-and-prepare actions it recovers five bijective,
commuting permutations, a transitive group of order nine, and learned shells
approximately \((0,3.96,4.92)\).

Weak higher-rank Lüders instruments also recover a nine-element commuting
action group at \(\eta=0.55\) and \(0.80\), with token-aggregated shells
approximately \((0,6.63,7.09)\) and \((0,5.11,5.91)\).  However, the last
token is not a sufficient predictive state.  The exact quantum filter improves
held-out NLL from 3.149 to 3.114 bits at \(\eta=0.55\), and from 3.074 to
3.017 bits at \(\eta=0.80\).

A naive two-event suffix estimator is slightly worse than the last-token
model.  Thus the experiment detects retained memory through the oracle gap but
does **not** yet learn that fiber successfully.  A controlled spectral PSR is
the appropriate next estimator.

The controls separate common false positives:

- null outcomes and an external nine-node DFA have no valid quantum action
  group;
- Haar-random controls can change future statistics but do not close into the
  translation algebra;
- action and token reshuffling leave group order and torus topology invariant;
- the memory-only limit is spatial but uninformative; and
- the report-only limit is informative but has the least word-metric-like
  shell ratio.

## 10. What is and is not established

Established exactly:

- a strict state-hitting \(\mathbb Z_3^2\) Bellman metric from informative
  integrated qutrit actions;
- analytic CP, trace-preservation, covariance, information, and Bellman
  formulas;
- numerical opaque recovery of the translation group from a common future
  probe in the exact family and from histories in the separate Lüders study;
- an exact Euclidean elementary square with positive information;
- analytic same-length displacement closure for nonselective exact-family
  channels; and
- a general theorem giving sufficient conditions for covariant hitting costs
  to form a group metric.

Not established:

- a globally planar nine-state embedding;
- an open exact \(3\times3\) Euclidean patch;
- full outcome-conditioned path closure beyond the finite Hesse branch process;
- successful data-driven learning of an internal quantum-memory fiber; or
- a necessary-and-sufficient classification of all informative instruments
  giving spatial hodology.

## 11. Most promising next experiment

The next step is a larger odd torus \(\mathbb Z_m^2\), preferably
\(m=5,7,9\), with a covariant higher-rank instrument and an opaque controlled
spectral PSR.  It should be evaluated on a central open patch of radius less
than \(m/2\), so periodic shortcuts are held out.  Planning should operate in
the learned predictive state rather than the last outcome token.  The decisive
tests are:

1. held-out controlled-string calibration against a quantum-filter oracle;
2. learned two-generator group closure and local neighborhood topology;
3. state-hitting metricity and Schoenberg rank on successively larger open
   patches;
4. separation of base coordinate from memory/purity fiber;
5. loop-dependent fiber transport at fixed base displacement; and
6. null, random-control, external-memory, and token-gauge controls.

This is a concrete bridge from an exact finite torus to a learned local atlas,
and from an oracle-detected candidate memory fiber to operational connection
and holonomy.

## 12. Reproduction

From `low_dimensional_hodology/covariant_memory_geometry/`:

```bash
python -m unittest discover -s search/tests -v
MPLBACKEND=Agg python search/run_search.py

cd learning
python -m unittest -v test_opaque_learning.py
MPLBACKEND=Agg python opaque_learning.py \
  --train 12000 --test 3000 --seed 20260812
```

The search contributes 380 exact candidates, analytic and numerical Bellman
tables, exhaustive Bellman verification, restricted path-closure audits,
observable-only action trials, and a summary figure. The
learning study contributes 72,000 training and 18,000 held-out seven-step
sequences across six models, controlled kernels, action-algebra tables,
Hankel spectra, and two figures. The focused search and learning suites pass.

Detailed derivations are in `theory/THEORY_COVARIANT_MEMORY.md`; exact-search
methods and data are in `search/RESULTS_SEARCH.md`; opaque-learning methods and
controls are in `learning/RESULTS_LEARNING.md`.  The standalone pedagogical
paper `COVARIANT_MEMORY_GEOMETRY.tex` integrates all three strands.
