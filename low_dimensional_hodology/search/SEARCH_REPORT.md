# Low-dimensional sequence goals: a skeptical search

## Executive conclusion

Nine basis states are **not** necessary for nine goals or for a two-dimensional
hodological geometry. A qutrit gives an exact and analytically tractable
counterexample: two unitary controls generate a nine-state
\(\mathbb Z_3\times\mathbb Z_3\) orbit, one common nine-outcome POVM supplies
nine nonorthogonal goals, and optimal goal-hitting costs inherit the periodic
lattice distance. Longer translated outcome sequences preserve essentially the
same geometry.

There are, however, three different statements hiding in the phrase “the agent
has learned a 2D lattice,” and only the third is strong enough for the larger
research program:

1. A **goal automaton** can have a 3x3 grid even when the quantum state never
   changes. This is a false positive.
2. Nine conditional quantum states can have an approximately planar distance
   matrix. A qubit already permits this locally, because a small patch of the
   Bloch sphere is nearly flat.
3. The **controlled predictive process itself** can carry two translation
   directions and induce lattice-ordered hitting costs. A qutrit supports an
   exact finite example; a qubit supports good local approximations but cannot
   support the same faithful ternary translation symmetry.

The strongest exact result here is a periodic 3x3 lattice, not an open square
with Euclidean distances. Its control graph is intrinsically a 2D torus, but
the shortest-path metric is not exactly embeddable in \(\mathbb R^2\). That
distinction is scientifically important: “two translation generators,” “a
two-dimensional manifold,” and “a Euclidean distance matrix of rank two” are
not equivalent criteria.

## Reproduction

From this directory:

```bash
python -m unittest -v test_search_low_dimensional.py
python search_low_dimensional.py
```

The run is deterministic and takes a few seconds. It writes numerical matrices
and summaries to `results/` and two figures to `figures/`. The production run
used 180 qubit rotation angles between 0.01 and 1.50 radians. All eight tests
pass.

## 1. Why Hilbert-space dimension does not simply count goals

A density operator on \(\mathbb C^d\) varies continuously in a real affine
space of dimension \(d^2-1\). Thus even a qubit has infinitely many distinct
states and can support arbitrarily many named target effects or history-based
goals. What dimension does limit is **perfect distinguishability**. If nine
positions must be identified without error in one shot, their supports must be
mutually orthogonal, and therefore \(d\geq9\). The previous nine-level model
satisfies this strong classical-position requirement.

The low-dimensional question relaxes it. A position may instead be:

- a nonorthogonal predictive state;
- an equivalence class of action-observation histories;
- a node in the controlled transition graph;
- or an externally stored goal-progress state.

Only the first three can plausibly count as environment-borne emergent space.
The fourth can imitate any geometry without the quantum system carrying it.

This motivates three distance objects. For conditional quantum states
\(\rho_i\), state distinguishability may be measured by trace distance

\[
 D_{\rm tr}(i,j)=\tfrac12\lVert\rho_i-\rho_j\rVert_1.
\]

For pure states this is

\[
 D_{\rm tr}(i,j)=\sqrt{1-|\langle\psi_i|\psi_j\rangle|^2}.
\]

For controlled dynamics, let \(C(i,g)\) be the optimal expected number of
unit-cost interventions needed to achieve goal \(g\) from predictive state
\(i\). The baseline-subtracted hodological cost is

\[
 H(i,g)=C(i,g)-C(g,g).
\]

Finally, a deterministic finite automaton (DFA) used to recognize goal
sequences has its own graph distance. That distance belongs to the agent's
external memory unless its progress states can be recovered from the physical
predictive process.

## 2. Deliberate false positive: a 3x3 automaton over a null qubit

Assign nine DFA nodes the labels \((x,y)\in\{0,1,2\}^2\) and let two classical
counters advance them. Their Manhattan distance is exactly

\[
 d_{\rm DFA}((x,y),(x',y'))=|x-x'|+|y-y'|.
\]

Now couple this DFA to a qubit on which every “control” is the identity. All
nine histories have the same density matrix, so every pairwise trace distance
is zero, yet the goal automaton still draws a perfect square grid. No learned
embedding or hitting-time plot alone can rule this out.

The automated null result records:

- 9 goal nodes;
- 1 distinct quantum state;
- maximum trace distance 0;
- nonzero Manhattan goal distances.

This yields the first non-negotiable diagnostic:

> Repeat every geometry analysis after replacing the instruments by a
> state-independent outcome process while leaving the goal recognizer intact.
> Geometry that survives unchanged is not evidence that the quantum process
> carries space.

## 3. A genuinely quantum sequence-counter ablation

The next model keeps the same external 3x3 counter DFA but replaces independent
fair coins by projective qubit \(X\) and \(Z\) measurements. A goal \((i,j)\)
requires accumulating \(i\) observed \(X+\) events and \(j\) observed \(Z+\)
events. The agent chooses which basis to measure at each unit-cost step.

For independent fair coins the solution is additive:

\[
 C_{\rm coin}(i,j)=2(i+j).
\]

The computed surface is

\[
 \begin{pmatrix}
 0&2&4\\
 2&4&6\\
 4&6&8
 \end{pmatrix}.
\]

Qubit backaction changes the problem. After an \(X-\) result, another \(X\)
measurement returns minus with certainty; the agent must measure \(Z\) to
restore a chance of \(X+\). The exactly solved four-state post-measurement MDP
gives

\[
 C_{\rm qubit}(i,j)=
 \begin{pmatrix}
 0&3&4\\
 3&5&6\\
 4&6&7
 \end{pmatrix}.
\]

Its maximum additive residual

\[
 \max_{i,j}|C(i,j)-C(i,0)-C(0,j)|
\]

is 1, versus numerical zero for the coin. Thus quantum backaction measurably
warps an otherwise product goal space. Nevertheless, the counters are still
external memory. This is a useful model of an **internal fiber over a hand-made
base**, not yet an emergence of the base itself.

## 4. Qubit tangent-plane construction

The Bloch sphere is two-dimensional, so a qubit can certainly carry local 2D
state geometry. Start at \(|0\rangle\) and define nine sequence-labelled states

\[
 |\psi_{xy}\rangle=R_x(\theta)^x R_y(\theta)^y|0\rangle,
 \qquad x,y\in\{0,1,2\}.
\]

These are not nine basis states. They are nine nonorthogonal endpoints of
control sequences. The search compares their pairwise trace distances with an
ordinary 3x3 Euclidean patch.

With a minimum-separation constraint of 0.12, the selected compromise was
\(\theta=0.309665\) radians. It has:

| diagnostic | result |
|---|---:|
| minimum pairwise trace separation | 0.125572 |
| Pearson correlation with Euclidean grid distance | 0.989749 |
| Spearman correlation with Manhattan distance | 0.954801 |
| classical-MDS 2D stress | 0.007361 |
| positive-Gram variance in first two dimensions | 0.992494 |
| mean translation-equivariance defect | 0.009424 |
| maximum translation-equivariance defect | 0.049083 |
| one-cell commutator defect | 0.010249 |

This is a strong approximate result, but it has a revealing singular limit.
As \(\theta\to0\), curvature and noncommutativity errors vanish, while all nine
states coalesce. Increasing \(\theta\) improves distinguishability but reveals
Bloch-sphere curvature and the failure of \(R_x\) and \(R_y\) to commute. The
plot in `figures/low_dimensional_search.png` displays this
resolution--flatness tradeoff.

There is also a symmetry obstruction to turning this construction into the
exact ternary lattice below. Two commuting order-three unitaries on a qubit
are simultaneously diagonalizable. After removal of a global phase, their
action supplies only one independent relative phase, so a ray orbit has at
most three elements. For the nontrivial Weyl relation

\[
 XZ=\omega ZX,\qquad \omega=e^{2\pi i/3},
\]

taking determinants gives \(\omega^d=1\); hence \(3\mid d\). A faithful
two-coordinate ternary projective translation representation therefore cannot
live in dimension two. This rules out this **particular exact covariance
scheme**, not every imaginable qubit instrument or every nine-goal geometry.

## 5. Exact qutrit construction I: Weyl--Hesse phase space

Let

\[
 X|j\rangle=|j+1\bmod3\rangle,\qquad
 Z|j\rangle=\omega^j|j\rangle,
\]

and choose the fiducial state

\[
 |\psi_{00}\rangle=(0,1,-1)^T/\sqrt2.
\]

The nine states

\[
 |\psi_{mn}\rangle=X^mZ^n|\psi_{00}\rangle
\]

form the qutrit Hesse SIC orbit. Their projectors obey

\[
 \sum_{m,n}\frac13\Pi_{mn}=I,
 \qquad
 |\langle\psi_{mn}|\psi_{m'n'}\rangle|^2=\frac14
 \quad ((m,n)\ne(m',n')).
\]

The physical action repertoire consists of four single-Kraus unitary
instruments \(X,X^\dagger,Z,Z^\dagger\) and one common measurement with Kraus
operators

\[
 K_{mn}=\frac1{\sqrt3}\Pi_{mn}.
\]

Conditional on outcome \((m,n)\), that measurement resets the qutrit to
\(|\psi_{mn}\rangle\). The unitary controls translate the labels. Although
\(X\) and \(Z\) commute only up to phase, their conjugation actions commute on
density matrices, giving an exact \(\mathbb Z_3^2\) orbit.

Goal \(g\) is not a target basis state. It is the event “the common SIC
measurement reports outcome \(g\).” From SIC state \(s\),

\[
 p(g|s)=
 \begin{cases}
 1/3,&g=s,\\
 1/12,&g\ne s.
 \end{cases}
\]

The stochastic-shortest-path Bellman equation is

\[
 V_g(s)=\min\left\{
 1+V_g(X^{\pm1}s),\;
 1+V_g(Z^{\pm1}s),\;
 1+\sum_{o\ne g}p(o|s)V_g(o)
 \right\}.
\]

It has the exact solution

\[
 V_g(s)=6+d_T(s,g),
\]

where

\[
 d_T((m,n),(m',n'))=
 \min(|m-m'|,3-|m-m'|)+
 \min(|n-n'|,3-|n-n'|)
\]

is the 3x3 toroidal Manhattan distance. At the goal, one measurement costs one
and its eight failures contribute
\((4\cdot7+4\cdot8)/12=5\), proving \(V_g(g)=6\). Away from the goal, a shortest
unitary move decreases \(d_T\) by one and satisfies the remaining Bellman
inequalities. The numerical solver independently recovers baseline 6 and
excess costs exactly 1 or 2, with Pearson and Spearman correlation 1.0 against
\(d_T\).

This example is maximally clean dynamically, but all distinct SIC states are
equiangular. Therefore static trace distance sees no lattice at all. The
geometry resides in the controlled transition relation, not in pairwise state
distinguishability.

## 6. Exact qutrit construction II: two relative phases

A still simpler orbit makes the two coordinates explicit:

\[
 |\phi_{mn}\rangle=
 \frac{|0\rangle+\omega^m|1\rangle+\omega^n|2\rangle}{\sqrt3}.
\]

Use commuting controls

\[
 U=\operatorname{diag}(1,\omega,1),\qquad
 V=\operatorname{diag}(1,1,\omega).
\]

Then \(U|\phi_{mn}\rangle=|\phi_{m+1,n}\rangle\) and
\(V|\phi_{mn}\rangle=|\phi_{m,n+1}\rangle\), exactly modulo three. The nine
effects \(E_{mn}=|\phi_{mn}\rangle\langle\phi_{mn}|/3\) again sum to identity,
so they define one common covariant POVM. The completeness residual in the run
is \(4.9\times10^{-16}\).

Unlike the Hesse SIC, distinct phase-grid states have fidelities 0 or 1/3. The
state trace distance therefore contains some spatial information (Pearson
correlation 0.577 with control distance), while the optimal goal-hitting cost
tracks the control distance extremely closely:

| diagnostic | result |
|---|---:|
| baseline goal cost | 5.5882352941 |
| control vs excess-cost Pearson correlation | 0.996116 |
| control vs excess-cost Spearman correlation | 0.942809 |
| distinct off-diagonal excess costs | 1, 1.8823529412, 2 |

The slight split at control distance two is not numerical noise. Some
measurement outcomes provide a shortcut because the POVM overlaps are
direction-dependent. This construction is thus less metrically exact than the
Hesse model but richer observationally.

### Translated sequence goals

To test genuinely elaborate goals, the same calculation was repeated when
success requires \(L\) **consecutive** reports of the designated POVM outcome.
Any different outcome or intervening control resets progress. This augments the
physical state by goal-progress \(k\in\{0,\ldots,L-1\}\), and the Bellman
recursion is solved on that product space.

| confirmations \(L\) | aligned baseline cost | correlation with control distance |
|---:|---:|---:|
| 1 | 5.588235 | 0.996116 |
| 2 | 22.352941 | 0.996116 |
| 3 | 72.647059 | 0.996116 |

The translated goal family retains the same relative geometry while difficulty
grows rapidly. This is precisely the desired separation between “where the
goal is” and “how internally demanding the goal template is.” In later work,
the sequence-progress coordinate is a natural candidate fiber over the qutrit
phase-space base.

## 7. In what sense is the exact example two-dimensional?

The qutrit constructions are exact two-generator periodic lattices:

- node set \(\mathbb Z_3\times\mathbb Z_3\);
- two independent translation controls and their inverses;
- homogeneous local transition structure;
- a translated family of goal effects or goal sequences;
- optimal hitting costs ordered by graph displacement.

But the toroidal Manhattan distance is not a two-dimensional Euclidean distance
matrix. For any proposed Euclidean distance matrix \(D\), classical distance
geometry forms

\[
 B=-\tfrac12 J D^{\circ2}J,
 \qquad J=I-\frac1N\mathbf1\mathbf1^T.
\]

Exact embedding in \(\mathbb R^k\) is equivalent to \(B\succeq0\) and
\(\operatorname{rank}B\leq k\). For the 3x3 torus control metric, \(B\) has
four positive eigenvalues equal to 3.5 and four negative eigenvalues equal to
-1. Its raw two-dimensional classical-MDS stress is 0.383. It should therefore
be described as a **discrete 2D periodic control topology**, not as an exact
Euclidean plane.

The same warning applies to an open square with Manhattan shortest-path costs:
ordinary square coordinates are two-dimensional, but graph geodesic distance
is \(L^1\), not Euclidean \(L^2\). A future theorem must state whether “maps
onto ordinary space” refers to topology, local adjacency, a Riemannian metric,
or literal equality of the scalar hodological cost with Euclidean distance.

## 8. Candidate necessary conditions

No useful necessary-and-sufficient theorem is possible until the target notion
of emergence is fixed. For environment-borne \(k\)-dimensional spatial
hodology, the following conditions are plausible and numerically testable.

### 8.1 Predictive support

Histories assigned different spatial coordinates should not all be
action-conditionally predictively equivalent. Define two histories equivalent
when every future intervention policy gives the same distribution over future
observations. Spatial coordinates should factor through these causal/predictive
equivalence classes, not merely through goal-progress memory.

Numerical test: train a discriminator for future observation strings from each
putative position under a common bank of probe policies. Report pairwise total
variation or an integral probability metric. Apply the same test after erasing
the goal-progress variables.

### 8.2 Independent controllable directions

There should be \(k\) locally independent action-induced displacement fields
on the predictive-state manifold. Estimate the Jacobian of predictive features
under each action and measure the rank of their span after quotienting gauge or
normalization directions.

For a finite homogeneous lattice, fit permutations or stochastic kernels
\(T_a\) and test closure, inverse relations, and commutators. The qutrit models
have exact closure and two independent order-three generators. The qubit patch
has small but nonzero equivariance defects.

### 8.3 Goal covariance

Goals should be translates of a common operational template rather than nine
unrelated hand-tuned recognizers. Seek a group or semigroup action satisfying

\[
 T_h G_g T_h^{-1}\simeq G_{h\cdot g}.
\]

For sequence goals, the entire recognition automaton—not just its terminal
effect—must transform covariantly. The phase-grid confirmation goals satisfy
this condition by construction.

### 8.4 Hodological compatibility

After subtracting goal-dependent terminal baselines, hitting costs should be
approximately stationary:

\[
 H(i,g)\approx h(g-i).
\]

Report symmetry error, triangle-inequality violations, translation-stationarity
residual, and correlations with control-graph displacement. Do not silently
replace an asymmetric cost by its average: report both directed and
symmetrized versions.

### 8.5 Dimensional compatibility

Use at least three independent notions:

- Euclidean distance-matrix PSD/rank and stress;
- intrinsic dimension of predictive states;
- number and rank of controllable displacement directions.

Agreement is strong evidence. Disagreement, as in the equiangular Hesse orbit,
identifies where the geometry actually resides.

### 8.6 Automaton ablations and memory accounting

At minimum run:

1. state-independent outcome replacement;
2. instrument-label permutation while holding the goal DFA fixed;
3. goal-progress erasure before geometric analysis;
4. physical-state reset between every goal-symbol observation;
5. matched goal automata with different quantum instruments.

Also report the number of physical predictive states, external DFA states, and
recurrent memory dimensions separately. A 3x3 geometry requiring nine external
counter states is not a low-dimensional physical realization merely because a
qubit was attached to it.

## 9. A useful sufficient construction theorem

The experiments suggest the following finite theorem schema.

Let a finite group \(G\) act projectively on a Hilbert space through unitaries
\(U_g\). Let \(\rho_g=U_g\rho_0U_g^\dagger\) be a closed orbit. Let
\(\{E_g\}_{g\in G}\) be a covariant POVM with an instrument that conditionally
resets to \(\rho_g\). Give the agent unit-cost controls corresponding to a
symmetric generating set \(S\subset G\), and let the goals be group translates
of one outcome or sequence template. Then:

1. the controlled predictive process has Cayley-graph covariance;
2. all goal value functions are translates of one value function;
3. baseline-subtracted hodological cost depends only on group displacement;
4. if the Bellman-optimal nonterminal actions follow shortest generator paths,
   hodological ordering agrees with Cayley distance.

The Hesse construction realizes the stronger identity
\(H(i,g)=d_{\rm Cayley}(i,g)\). The phase construction realizes a small
measurement-induced deformation of it. This schema is a promising exact basis
for later perturbation theory: weaken covariance, add noisy controls, or attach
internal sequence templates and calculate how the geometry deforms.

The schema is sufficient for homogeneous discrete space, not necessary for all
emergent geometries. Boundaries, curvature, inhomogeneous costs, and stochastic
local charts deliberately break global group covariance.

## 10. Recommended next theoretical program

1. **Promote the Hesse model to the canonical solvable baseline.** Prove the
   Bellman solution and covariance theorem formally, including uniqueness and
   proper-policy conditions.
2. **Use the phase-grid model for observational geometry.** Derive its rational
   hitting costs exactly after identifying the stable Bellman policy, rather
   than relying on converged floating-point iteration.
3. **Separate topology from metric.** Decide whether the primary target is a
   2D transition complex, a discrete Riemannian metric, or exact Euclidean EDM.
4. **Create an open-grid perturbation.** Add weak dissipative boundary markers
   or position-dependent goal penalties to break the torus without introducing
   an external nine-state register. Quantify covariance breaking.
5. **Develop a bundle model.** Use phase-space label \((m,n)\) as base and
   translated sequence progress \(k\) as fiber. Compare product goals with
   state-dependent internal transitions to introduce a nontrivial connection
   or holonomy.
6. **Run causal-state ablations on learned agents.** Verify that learned 2D
   coordinates remain decodable from future quantum observations after goal
   identifiers and tracker progress are hidden.

## Artifacts

- `search_low_dimensional.py`: exact Bellman solvers, qubit angle search,
  diagnostics, tables, and plotting.
- `test_search_low_dimensional.py`: eight tests for POVM completeness,
  covariance, exact torus distance, sequence geometry, and ablations.
- `results/summary.json`: machine-readable headline results.
- `results/qubit_rotation_search.csv`: complete 180-angle scan.
- `results/qutrit_*`: probabilities, policies, cost matrices, and embeddings.
- `results/*counter*`: fair-coin and projective-qubit sequence-counter surfaces.
- `figures/low_dimensional_search.png`: qubit tradeoff, exact qutrit cost
  classes, and automaton null.
- `figures/qutrit_and_counter_comparison.png`: Hesse versus phase-grid state
  geometry, measurement shortcuts, and qubit counter backaction.

## References

- J. M. Renes, R. Blume-Kohout, A. J. Scott, and C. M. Caves,
  “Symmetric informationally complete quantum measurements,” *Journal of
  Mathematical Physics* **45**, 2171–2180 (2004), arXiv:quant-ph/0310075.
- A. J. Scott, “Tight informationally complete quantum measurements,”
  *Journal of Physics A* **39**, 13507–13530 (2006), arXiv:quant-ph/0604049.
- M. L. Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic
  Programming*, Wiley (1994), for stochastic-shortest-path Bellman theory.

