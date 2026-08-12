# Covariant quantum memory and operational torus geometry

## Executive result

This note develops outcome-conditioned, group-covariant qutrit instruments
whose branches retain quantum memory. The target is an operationally learned
two-dimensional torus, or a locally flat patch of its universal cover. A
global isometric embedding of a finite torus in the Euclidean plane is not
required.

The principal conclusions are:

1. Terminal semantics changes the earlier Hesse result. Under the earlier
   “report the goal again” convention, the sharp integrated Hesse instrument
   has Bellman shells \((4,4,5)\), so self and edge collapse. Under standard
   state-hitting semantics, \(V_g(g)=0\) before another intervention, the same
   instrument has the exact shells
   \[
   \boxed{(v_0,v_1,v_2)=(0,4,5).}
   \]
   This is a genuine translation-invariant metric on \(\mathbb Z_3^2\).
2. That metric recovers the torus topology exactly: distance-four pairs are
   precisely Cayley neighbors. It is not planar. Its Schoenberg Gram matrix
   is positive semidefinite of rank eight, with four eigenvalues \(17\) and
   four eigenvalues \(7/2\).
3. More generally, positive-cost translation-covariant stochastic
   shortest-path problems induce a group metric when controls admit
   equal-cost inverse-reflected policies and the goal is a hitting boundary.
4. A positive Choi seed gives a complete parametrization that guarantees CP,
   trace preservation, and covariance by construction.
5. The exactly soluble two-Kraus family
   \[
   \mathcal J_o^\lambda(\rho)=
   \frac{\lambda}{3}\Pi_o\rho\Pi_o+
   \frac{1-\lambda}{9}\rho
   \]
   interpolates between a uniformly reported memory-preserving channel and
   the sharp Hesse reset. For \(0<\lambda<1\), each branch has Choi rank two,
   its outcome is immediately informative, and its posterior retains a
   nonzero copy of the incoming state.
6. Higher-rank branches remove the outcome-only reset mechanism, but do not
   alone guarantee path closure, metricity, or local flatness. Those are
   separate falsifiable conditions.

Hilbert dimension, branch Choi rank, posterior density rank, predictive
Hankel rank, and Euclidean embedding rank are distinct resources.

---

## 1. Controlled instruments and operational states

Let \(G=\mathbb Z_m\times\mathbb Z_m\), written additively, and let
\(g\mapsto U_g\) be a qutrit unitary representation. An action
\(a\in A\subseteq G\) selects an instrument
\[
\mathfrak I^a=\{\mathcal E_o^a:o\in O\},\qquad
\sum_o\mathcal E_o^a\ \text{trace preserving}.
\]
For state \(\rho\),
\[
p(o\mid\rho,a)=\operatorname{Tr}\mathcal E_o^a(\rho),\qquad
\rho_{a,o}=\frac{\mathcal E_o^a(\rho)}{p(o\mid\rho,a)}.
\]

A history \(h=(a_1,o_1)\cdots(a_t,o_t)\) is predictively equivalent to \(h'\)
when every allowed future controlled test has the same conditional
probability after both. The causal state \([h]\) is therefore a row of the
controlled Hankel matrix, not an assigned latent label. Classical controller
memory may enlarge histories, but it is not automatically quantum predictive
geometry.

### Definition 1 (learned translation structure)

Opaque actions realize a learned \(G\)-translation structure on a predictive
quotient \(S\) if future statistics identify bijections \(T_g:S\to S\) with
\[
T_0=\operatorname{id},\qquad T_gT_h=T_{g+h},
\]
up to a common relabeling of \(S\) and an automorphism of \(G\). This gauge is
unavoidable: data cannot determine which recovered generator should be named
east.

---

## 2. Parametrizations guaranteeing physicality and covariance

### 2.1 Base instrument followed by a control

Suppose the outcomes form a \(G\)-orbit. A base instrument is covariant when
\[
\mathcal J_{o+g}(U_g\rho U_g^\dagger)
=U_g\mathcal J_o(\rho)U_g^\dagger.
\tag{2.1}
\]
Define integrated move-and-observe actions by
\[
\mathcal E_o^a=\mathcal J_o\circ\operatorname{Ad}_{U_a}.
\tag{2.2}
\]
Then
\[
p(o+g\mid U_g\rho U_g^\dagger,a)=p(o\mid\rho,a),
\tag{2.3}
\]
so absolute labels may be shuffled while relational response fields remain
learnable.

### Proposition 2 (Choi-seed construction)

Let \(R_g=U_g\otimes\overline{U_g}\) act on output-input Choi space. Choose a
positive seed \(C_0\succeq0\), and set
\[
C_o=R_oC_0R_o^\dagger.
\tag{2.4}
\]
If
\[
\operatorname{Tr}_{\rm out}\sum_{o\in G}C_o=I,
\tag{2.5}
\]
then the CP maps having Choi matrices \(C_o\) form a CPTP covariant
instrument. Conversely, every transitive covariant instrument has this form,
with the seed invariant under the reference outcome's stabilizer.

#### Proof

Unitary conjugation preserves \(C_o\succeq0\), hence complete positivity.
Equation (2.5) is exactly trace preservation of the sum. Conjugating by
\(R_g\) maps \(C_o\) to \(C_{o+g}\), which is (2.1) in Choi form. Conversely,
covariance transports the reference Choi matrix around the orbit; ambiguity
in orbit representatives is precisely stabilizer invariance. ∎

For numerical optimization, write \(C_0=LL^\dagger\) and enforce (2.5)
exactly. Physicality and covariance then cannot be lost during training.

### 2.2 Kraus-orbit construction and retained memory

An explicit alternative starts from seed Kraus operators \(L_\mu\):
\[
K_{o,\mu}=U_oL_\mu U_o^\dagger,\qquad
\sum_{o,\mu}K_{o,\mu}^\dagger K_{o,\mu}=I.
\tag{2.6}
\]
The branch Choi rank is at most the number of linearly independent seeds.

### Definition 3 (retained operational quantum memory)

A branch \(o\) retains memory on a reachable set \(\mathcal R\) if there are
\(\rho,\sigma\in\mathcal R\) such that the normalized posteriors conditioned
on the same outcome are separated by an allowed future test:
\[
\frac{\mathcal J_o(\rho)}{\operatorname{Tr}\mathcal J_o(\rho)}
\not\sim
\frac{\mathcal J_o(\sigma)}{\operatorname{Tr}\mathcal J_o(\sigma)}.
\tag{2.7}
\]
Mixed posterior rank does not prove retained memory; future separation does.

---

## 3. Covariance versus predictive path closure

Let \(\Phi=\sum_o\mathcal J_o\). Covariance implies
\[
\Phi\operatorname{Ad}_{U_g}
=\operatorname{Ad}_{U_g}\Phi.
\tag{3.1}
\]
The nonselective integrated actions
\(\Phi_a=\Phi\operatorname{Ad}_{U_a}\) consequently satisfy
\[
\Phi_{a_t}\cdots\Phi_{a_1}
=\Phi^t\operatorname{Ad}_{U_{a_1+\cdots+a_t}}.
\tag{3.2}
\]
Thus covariance gives equal-length displacement closure. Full closure across
different lengths additionally requires \(\Phi^t\) to be predictively
invisible or idempotent on the relevant quotient. Otherwise elapsed
measurement depth is a genuine fiber or clock coordinate.

### Proposition 4 (finite path-closure criterion)

For a finite reachable predictive set and a finite test set that separates
it, words \(w,w'\) are path equivalent if and only if
\[
p(\tau\mid h,w)=p(\tau\mid h,w')
\]
for every separating history \(h\) and core test \(\tau\). A proposed
\(G\)-quotient is valid exactly when this holds for every pair assigned the
same displacement.

#### Proof

Necessity is causal equivalence. Sufficiency follows because equality of all
separating core-test coordinates is equality in the predictive quotient. ∎

The empirical residual
\[
\epsilon_{\rm path}(L)=
\max_{\substack{|w|,|w'|\le L\\\Delta(w)=\Delta(w')}}
\max_{h,\tau}|p(\tau\mid h,w)-p(\tau\mid h,w')|
\tag{3.3}
\]
should be compared with bootstrap sampling error and random-unitary controls.
Outcome-conditioned closure requires comparing complete word/outcome kernels,
not only nonselective channels.

---

## 4. Exact Hesse geometry under state-hitting semantics

For \(m=3\), let \(\Pi_s\), \(s\in G\), be the Hesse SIC projectors:
\[
\frac13\sum_s\Pi_s=I,\qquad
\operatorname{Tr}(\Pi_s\Pi_t)=
\begin{cases}1&s=t,\\1/4&s\ne t.\end{cases}
\]
Let controls be identity and the four cardinal Weyl translations. The
integrated branches are
\[
\mathcal E_o^a(\rho)=
\frac13\Pi_oU_a\rho U_a^\dagger\Pi_o.
\tag{4.1}
\]
On input \(\Pi_s\),
\[
p_a(o\mid s)=
\begin{cases}
1/3,&o=s+a,\\
1/12,&o\ne s+a,
\end{cases}
\qquad \rho_{a,o}=\Pi_o.
\tag{4.2}
\]

### 4.1 Terminal semantics are part of the goal

Two tasks must be distinguished.

- **Report-again:** even at \(g\), pay for another action that reports \(g\).
  This produces the earlier shells \((4,4,5)\).
- **State hitting:** arrival in predictive state \(g\) terminates immediately,
  so
  \[
  V_g(g)=0.
  \tag{4.3}
  \]

Report-again models costly certification to an external observer. State
hitting models occupying the operational goal state. Neither is universally
correct, but they cannot be mixed in one distance matrix.

### Theorem 5 (exact Hesse state-hitting values)

With state-hitting semantics and unit action cost, the optimal Bellman values
depend only on torus shell and are
\[
\boxed{v_0=0,\qquad v_1=4,\qquad v_2=5.}
\tag{4.4}
\]

#### Proof

The boundary gives \(v_0=0\). From an edge state, choose the cardinal move
whose likelihood peak is \(g\). The goal outcome has probability \(1/3\) and
terminates. The four edge and four diagonal failures each have probability
\(1/12\). Hence
\[
v_1=1+\frac13v_1+\frac13v_2.
\tag{4.5}
\]
From a diagonal state, an optimal move peaks at an edge. The off-peak goal
outcome has probability \(1/12\). The peak plus three other edge outcomes have
total coefficient \(1/3+3/12=7/12\); four diagonal outcomes total \(1/3\).
Thus
\[
v_2=1+\frac7{12}v_1+\frac13v_2.
\tag{4.6}
\]
Solving gives \(v_1=4,v_2=5\). Comparing the five symmetric action rows
verifies the stated minimizers. ∎

### Corollary 6 (a genuine torus metric)

The hitting cost \(D(s,t)=V_t(s)\) is a translation-invariant metric on
\(\mathbb Z_3^2\): it has values \(0,4,5\).

#### Proof

Translation covariance and inverse-symmetric controls give symmetry.
Off-diagonal distances are positive. The only nontrivial triangle bound is
\(5\le4+4\). ∎

### Proposition 7 (exact Euclidean classification)

The metric of Corollary 6 is Euclidean, but its minimal Euclidean embedding
dimension is eight, not two.

#### Proof

\(D^2\) equals \(16\) on four cardinal displacements and \(25\) on four
diagonal displacements. For a character \((k,l)\in\mathbb Z_3^2\), let
\(A_0=2\), \(A_1=A_2=-1\). The Fourier eigenvalue of \(D^2\) is
\[
\widehat{D^2}(k,l)=16(A_k+A_l)+25A_kA_l.
\]
For a nontrivial character with exactly one zero coordinate this is \(-34\);
with both coordinates nonzero it is \(-7\). On the centered subspace, the
Schoenberg Gram eigenvalues are minus half these numbers: \(17\) four times
and \(7/2\) four times. All are positive, so the Euclidean embedding rank is
eight. ∎

This is an exact two-dimensional torus topology and translation algebra, not
a global planar Euclidean distance matrix.

---

## 5. A general hitting-cost metric theorem

### Theorem 8 (covariant stochastic hitting costs induce a group metric)

Consider a controlled Markov process on a finite group \(G\). Assume:

1. transition laws are translation covariant;
2. each nonterminal action costs at least \(c_{\min}>0\);
3. every goal is reached under some proper policy;
4. every policy has an equal-cost inverse-reflected policy;
5. goals are absorbing hitting boundaries: \(V_g(g)=0\).

Then \(D(s,g)=V_g(s)\) is a translation-invariant metric on \(G\).

#### Proof

Nonnegativity is immediate. If \(s\ne g\), every successful trajectory uses a
nonterminal action, so \(V_g(s)\ge c_{\min}>0\). Translation covariance gives
\(V_g(s)=v(g-s)\), while inverse reflection gives symmetry. Concatenate an
\(\varepsilon\)-optimal policy from \(s\) to \(u\) with one from \(u\) to
\(g\). By the strong Markov property at the hitting time of \(u\),
\[
V_g(s)\le V_u(s)+V_g(u)+2\varepsilon.
\]
Letting \(\varepsilon\) vanish proves the triangle inequality. ∎

The theorem gives metricity, not Euclidean dimension or locality.

### Proposition 9 (report-again kernel-cloning obstruction)

Under report-again semantics, suppose an edge state \(s\) has an action \(a\)
whose complete outcome-and-posterior kernel equals an action \(b\)'s kernel at
the goal, both actions have equal cost, and both are optimal. Then
\(V_g(s)=V_g(g)\).

#### Proof

The two Bellman right-hand sides are identical term by term. ∎

This statement is independent of branch Choi rank. Higher rank helps only if
it breaks the operational kernel equality.

---

## 6. An exact higher-rank retained-memory family

Define
\[
\boxed{\mathcal J_o^\lambda(\rho)=
\frac{\lambda}{3}\Pi_o\rho\Pi_o+
\frac{1-\lambda}{9}\rho},\qquad 0\le\lambda\le1,
\tag{6.1}
\]
and
\[
\mathcal E_o^{a,\lambda}
=\mathcal J_o^\lambda\circ\operatorname{Ad}_{U_a}.
\tag{6.2}
\]
Kraus operators are
\[
K_{o,1}^{a,\lambda}=\sqrt{\lambda/3}\,\Pi_oU_a,\qquad
K_{o,2}^{a,\lambda}=\sqrt{(1-\lambda)/9}\,U_a.
\tag{6.3}
\]

### Proposition 10 (physicality, covariance, and memory)

Equations (6.1)--(6.3) define a covariant CPTP instrument. For
\(0<\lambda<1\), each branch has Choi rank two and retains operational input
memory whenever future tests are informationally complete.

#### Proof

Complete positivity follows from the Kraus form. Summing effects gives
\[
\sum_o\left[\frac{\lambda}{3}U_a^\dagger\Pi_oU_a+
\frac{1-\lambda}{9}I\right]=\lambda I+(1-\lambda)I=I.
\]
Hesse covariance transports \(\Pi_o\). The posterior is proportional to
\[
\frac{\lambda}{3}\Pi_oU_a\rho U_a^\dagger\Pi_o+
\frac{1-\lambda}{9}U_a\rho U_a^\dagger.
\tag{6.4}
\]
The second term is nonzero and source dependent. Informationally complete
future tests separate generic posteriors from distinct inputs. Finally,
\(\Pi_oU_a\) and \(U_a\) are linearly independent, so the branch Choi matrix
has rank two. ∎

On SIC input \(\Pi_s\), the exact likelihood is
\[
p_\lambda(o\mid s,a)=
\begin{cases}
(1+2\lambda)/9,&o=s+a,\\
(4-\lambda)/36,&o\ne s+a.
\end{cases}
\tag{6.5}
\]
For a uniform prior, immediate mutual information is
\[
I_\lambda=\log_2 9-
H\left(
\frac{1+2\lambda}{9},
\underbrace{\frac{4-\lambda}{36},\ldots,
\frac{4-\lambda}{36}}_{8}
\right).
\tag{6.6}
\]
It grows from zero to \(0.251629\) bits. At \(\lambda=0\), outcomes are
uniform but the posterior is exactly the translated input: immediate
information vanishes while future-operational action meaning remains.

For \(0<\lambda<1\), off-peak branches generate mixed states outside the nine
SIC orbit. A Bellman model tracking only the last token is therefore invalid;
the exact state is the conditional density operator or an equivalent PSR.
Theorem 8 still guarantees a metric when true reachable goal states use
state-hitting semantics and the stated properness and inverse conditions hold.
Whether memory improves shell ratios and local flatness is a computational,
not automatic, consequence.

---

## 7. Information, disturbance, and useful memory

Immediate mutual information is not action meaning. Action identifiability
asks whether shifted future Hankel blocks differ, and may hold at
\(\lambda=0\). A direct retained-memory statistic is
\[
M_\lambda=\mathbb E_{o,s,t}\operatorname{JS}\left(
p(\mathcal T\mid\rho_{s,a,o}),
p(\mathcal T\mid\rho_{t,a,o})\right),
\tag{7.1}
\]
for a separating battery of future tests \(\mathcal T\). It vanishes for an
outcome-only reset and is positive for retained memory.

Measure disturbance relative to intended motion:
\[
F_{\rm move}(\rho,a)=
F\left(\sum_o\mathcal E_o^a(\rho),U_a\rho U_a^\dagger\right),
\tag{7.2}
\]
or by excess optimal goal cost relative to a unitary benchmark. A suitable
multiobjective design is
\[
\max\{I_{\rm immediate},I_{\rm future},M,\Delta_{\rm edge}\},\qquad
\min\{1-F_{\rm move},\epsilon_{\rm path},\epsilon_{\rm flat}\}.
\tag{7.3}
\]
Exact unitary nonselective motion and input-dependent immediate outcomes
cannot coexist on the full state space: a unitary Choi matrix has rank one,
so every positive branch Choi matrix summing to it is proportional to it.
Approximate information-disturbance tradeoffs are unavoidable.

---

## 8. Torus topology and local geometry

Opaque learned actions recover a discrete torus when:

- two generators have order \(m\);
- they commute on the predictive quotient;
- the generated group has order \(m^2\);
- its action on learned landmarks is regular and transitive;
- inverse actions are identifiable;
- the four-generator Cayley graph is connected.

These invariants survive outcome relabeling and group-automorphism gauge.

A periodic square lattice is locally flat but globally not a planar subset
with its intrinsic shortest-path metric. For \(m\ge5\), every graph ball of
radius
\[
r<m/2
\tag{8.1}
\]
lifts uniquely to \(\mathbb Z^2\), before wraparound occurs.

A qutrit supports larger nonorthogonal phase tori:
\[
|\phi_{xy}^{(m)}\rangle=
\frac{|0\rangle+\zeta^x|1\rangle+\zeta^y|2\rangle}{\sqrt3},
\quad
\zeta=e^{2\pi i/m},
\tag{8.2}
\]
under commuting controls
\[
U_x=\operatorname{diag}(1,\zeta,1),\qquad
U_y=\operatorname{diag}(1,1,\zeta).
\tag{8.3}
\]
Hilbert dimension three does not bound the orbit to nine states, although
operational discrimination becomes harder as \(m\) grows.

For every learned cost matrix:

1. test symmetry, strict off-diagonal positivity, and all triangles;
2. test translation homogeneity;
3. compute the global Schoenberg matrix
   \(B=-\frac12JD^{\circ2}J\);
4. repeat Schoenberg or norm fitting on nonwrapping patches;
5. fit local coordinates and test whether generator increments are
   state-independent after one common Procrustes gauge.

Square-grid shortest-path cost is locally Manhattan, hence Finsler rather
than Euclidean. Exact planar Euclidean geometry needs a quadratic energy law,
an isotropic diffusion limit, or sufficiently rich directions. The norm must
be reported honestly rather than hidden by a two-dimensional plot.

---

## 9. Necessary, sufficient, and no-go conditions

### Necessary operational conditions

1. Candidate causal states have distinct finite test rows.
2. Every intended action pair differs on a shifted future test.
3. Learned transformations obey group relations within uncertainty.
4. Every goal has a finite proper hitting policy.
5. \(V_g(s)>V_g(g)\) for \(s\ne g\).
6. Symmetry and triangle inequalities hold.
7. Lowest-cost neighbors coincide with the generator graph.
8. Translated nonwrapping patches share the same local Euclidean or Finsler
   model.

### Sufficient finite certificate

For a finite predictive quotient it is sufficient that:

- finite core tests separate states and shifted action kernels;
- learned actions generate a regular \(\mathbb Z_m^2\) action;
- a positive-cost, proper, inverse-symmetric hitting task is covariant;
- the lowest-cost pairs are exactly generator neighbors;
- every patch below injectivity radius fits one common declared local metric
  with zero residual.

The first two items learn topology, Theorem 8 gives the metric, and the last
two identify its local geometric type.

### No-go boundaries

1. Universal exact unitary motion implies state-independent branch
   probabilities.
2. Under report-again semantics, cloned operational kernels force self-edge
   collapse regardless of Choi rank.
3. Outcome-only preparation cannot retain source memory.
4. Covariance alone does not imply path closure, metricity, or Euclidean rank.
5. Finitely many homogeneous directions generically produce polygonal unit
   balls, not a Riemannian norm.

---

## 10. Falsifiable experiments

### A. Terminal-semantic regression test

Recompute the sharp Hesse model under both conventions.

- Report-again target: \((4,4,5)\).
- State-hitting target: \((0,4,5)\).
- Bellman residual tolerance: \(10^{-12}\).
- Hitting Schoenberg spectrum: \(17\) four times, \(7/2\) four times, one
  zero.
- Nearest-neighbor graph: exactly the four Weyl edges.

### B. Retained-memory interpolation

For \(\lambda=0,.05,\ldots,1\), implement (6.1):

1. verify completeness and covariance below \(10^{-12}\);
2. compare analytic (6.5)--(6.6) with Monte Carlo data;
3. estimate predictive Hankel rank with bootstrap intervals;
4. measure same-outcome future separation \(M_\lambda\);
5. solve belief-state hitting problems for every translated goal;
6. record strict edge gap, triangle violations, shell variance, topology
   recovery, and patch flatness;
7. repeat with report-again semantics to test whether memory breaks kernel
   cloning.

Choi rank two is not acceptance. The confidence interval for operational
same-outcome future separation must exclude zero.

### C. Larger qutrit torus

Use (8.2)--(8.3) with \(m=5,7,9\), a covariant phase POVM, and a weak
full-rank instrument.

- Randomly permute action and outcome identifiers.
- Fit a spectral PSR on words through length six and test through length ten.
- Recover orders, inverses, commutators, group size, and orbit.
- Compare global torus distortion with patches satisfying \(r<m/2\).
- Test generator-increment consistency across patches.
- Bootstrap rank, likelihood, covariance, and Bellman uncertainties.

### D. Decisive ablations

- sharp reset \(\lambda=1\);
- uninformative memory limit \(\lambda=0\);
- identical-action null;
- distinguishable but noncovariant Haar unitaries;
- two cloned action identifiers;
- episode-wise action relabeling;
- external classical counter with null quantum predictive rank;
- post hoc outcome shuffle preserving marginals but destroying branches.

### Acceptance table

| Layer | Statistic | Acceptance |
|---|---|---|
| Physical | completeness/covariance residual | \(<10^{-10}\) |
| Prediction | held-out NLL/calibration | beats marginal null |
| Memory | same-outcome future separation | confidence interval \(>0\) |
| Action | shifted-Hankel separation | every intended pair |
| Algebra | order/commutator/orbit | \(\mathbb Z_m^2\) within error |
| Hitting | Bellman solution | finite, residual \(<10^{-10}\) |
| Metric | strictness/triangle/symmetry | no significant violations |
| Topology | low-cost neighbor graph | torus isomorphism |
| Local geometry | patch residual/increment variance | below preregistered limit |
| Controls | null/random/cloned tests | fail at predicted layer |

---

## 11. Recommended immediate direction

The next step should be the analytic rank-two family (6.1), not an
unconstrained neural search.

1. Reproduce the \((0,4,5)\) state-hitting theorem and rank-eight global
   embedding. This is the positive operational-torus baseline.
2. Scan \(\lambda\) to learn whether retained branch memory reduces goal cost,
   changes shell ratios, and breaks report-again kernel cloning.
3. If report-again equality persists, inspect the full operational kernel
   equality and introduce a covariant coherent branch term whose posterior
   remains source dependent after an aligning move.
4. Move to \(m=5\) or \(7\), where nonwrapping patches contain plaquettes and
   local-flatness tests are meaningful.
5. Only then optimize a positive Choi seed, keeping physicality and covariance
   exact while optimizing predictive and Bellman objectives.

The earlier negative result was not a blanket impossibility of integrated
qutrit geometry. It concerned a particular certification semantics and an
outcome-only reset. Standard state hitting yields an exact metric on the same
learned torus. Retained quantum memory is the next controlled degree of
freedom, while the separation between topology, intrinsic metric, and global
Euclidean embedding keeps future claims honest.

---

## 12. Provenance

This note starts from the controlled-instrument, predictive-equivalence,
action-gauge, Bellman, and Hesse-SIC work in the informative-actions
miniproject. New derivations here are:

- the exact state-hitting Hesse shells \((0,4,5)\);
- their exact Schoenberg spectrum and rank-eight classification;
- the covariant hitting-cost metric theorem;
- the branch-rank-independent kernel-cloning obstruction;
- the exact retained-memory family (6.1), with Kraus form, likelihood,
  information, and memory proof;
- the local-torus versus global-EDM acceptance framework.
