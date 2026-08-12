# Informative actions and operationally learned spatial meaning

## Predictive equivalence, finite identification, and low-dimensional quantum hodology

**Status.** This note develops a strict notion of learned action meaning. An
action is not “north” because its name, source code, or hidden Kraus operator
says so. Its meaning is the transformation it induces in conditional
distributions of future outcomes under future experiments.

The main conclusions are:

1. Histories, states, and actions must be quotiented by **predictive
   equivalence**: two objects mean the same thing exactly when no allowed
   continuation distinguishes them.
2. A \(d\)-dimensional Markovian quantum process has predictive Hankel rank at
   most \(d^2\). Reachable-state and observable-effect spaces stabilize after
   finitely many steps, giving exact finite observability and action-equivalence
   tests.
3. Learned predictive operators are identifiable only up to similarity gauge,
   token permutations, and Kraus-representation freedom. Future probabilities,
   composition laws, invariant spectra, Bellman values, and goal geometry are
   meaningful; arbitrary matrix entries are not.
4. Mutual information, Fisher information, channel distinguishability, and
   causal displacement answer different questions. A random-unitary move may
   have no immediate state information yet have a clear meaning through later
   probes.
5. Predictive separation, shifted-Hankel action identification, Bellman
   realizability, a rank-two positive Schoenberg Gram matrix, and
   action-consistent local increments give a necessary-and-sufficient finite
   criterion for the desired operational 2D atlas.
6. A qubit supports an exact four-goal square whose Pauli actions are learned
   from tetrahedral-SIC response permutations. It cannot support a robust
   smooth plane through commuting phases: that family has Fisher rank one.
7. A qutrit is minimal for two independent commuting relative phases. A
   covariant phase POVM gives an exactly soluble response field from which
   translation meanings can be learned without coordinate labels.

The recommended experiment randomly renames every action and outcome, learns a
finite predictive-state representation from sequence frequencies, reconstructs
the qutrit translation algebra up to gauge, and only then analyzes goal
geometry. Cloned-action, null-probe, outcome-rescrambling, and twirled
random-unitary controls must fail at predetermined points.

---

## 1. Why labels cannot carry semantics

Buttons named “north,” “south,” “east,” and “west” already contain a spatial
interpretation. The same leakage occurs when an analysis:

- plots action indices in a preferred compass order;
- reads a privileged Kraus matrix to decide which coordinate changed;
- lets a goal automaton update a hand-coded \((x,y)\) counter;
- aligns learned gates to target gates before evaluating geometry;
- uses ordered outcome symbols as hidden coordinates.

Renaming all action and outcome tokens by fixed unknown permutations must
change none of the scientific conclusions. An unknown button earns a spatial
meaning only from relations such as:

- after a reference history it shifts a characteristic future response;
- repeated use composes the same shift;
- another button reverses it;
- two learned axis transformations commute;
- its response transformation is homogeneous across operational places;
- its effect on many goal values is a consistent local displacement.

Direction names may be attached after these facts are learned. Even then, a
global rotation, reflection, translation, or relabeling is pure gauge.

---

## 2. Controlled quantum processes

### 2.1 Instruments, histories, and tests

Let \(\mathcal H\cong\mathbb C^d\). Action \(a\) is an instrument with observed
outcomes \(o\) and branch maps

\[
\mathcal E_o^a(\rho)=\sum_kK_{o,k}^a\rho K_{o,k}^{a\dagger},
\qquad
\sum_{o,k}K_{o,k}^{a\dagger}K_{o,k}^a=I_d.
\]

The probability and conditional state are

\[
p(o\mid\rho,a)=\operatorname{Tr}\mathcal E_o^a(\rho),
\qquad
\rho'=\frac{\mathcal E_o^a(\rho)}{p(o\mid\rho,a)}.
\]

A history and a contemplated future test are

\[
h=(a_1,o_1,\ldots,a_t,o_t),
\qquad
\tau=(b_1,y_1,\ldots,b_L,y_L).
\]

Writing \(\mathcal E_h\) and \(\mathcal E_\tau\) for composed branch maps,

\[
p(h,\tau)=
\operatorname{Tr}\!\left[\mathcal E_\tau\mathcal E_h(\rho_0)\right].
\tag{1}
\]

For \(p(h)>0\),

\[
p(\tau\mid h)=\frac{p(h,\tau)}{p(h)}.
\tag{2}
\]

The agent can estimate these probabilities by repeated interaction without
knowing a state, Hilbert dimension, or Kraus operator.

### 2.2 Dual continuation effects

The Hilbert--Schmidt adjoint satisfies

\[
\operatorname{Tr}[F\mathcal E(\rho)]
=\operatorname{Tr}[\mathcal E^*(F)\rho].
\]

Each future test defines an effect

\[
F_\tau=\mathcal E_\tau^*(I),
\]

and therefore

\[
p(\tau\mid h)=\operatorname{Tr}[F_\tau\rho_h].
\tag{3}
\]

Histories generate reachable states; future tests generate observable effects;
their bilinear pairing contains all accessible meaning.

---

## 3. Predictive equivalence

### Definition 1 (causal-state equivalence)

\[
h\sim_{\mathrm{pred}}h'
\quad\Longleftrightarrow\quad
p(\tau\mid h)=p(\tau\mid h')
\quad\text{for every allowed future test }\tau.
\tag{4}
\]

The class \([h]\) is an operational causal state. Agreement on only the next
outcome is insufficient: a hidden coherence can become visible after a later
rotation.

For a restricted goal test family \(\mathcal T_G\), define

\[
h\sim_Gh'
\quad\Longleftrightarrow\quad
p(\tau\mid h)=p(\tau\mid h')
\quad\forall\tau\in\mathcal T_G.
\tag{5}
\]

Goal-relative state can be cheaper than full predictive state, but may merge
places that a common atlas should distinguish.

### Proposition 1 (posterior sufficiency)

In a known Markovian model,

\[
\rho_h=\rho_{h'}\Longrightarrow h\sim_{\mathrm{pred}}h'.
\]

The converse holds only if future effects separate the relevant density
operators.

#### Proof

Equal states give equal pairings in (3). If the future-effect span is
incomplete, a nonzero \(\rho_h-\rho_{h'}\) can be orthogonal to every accessible
effect and remain unobservable. \(\square\)

Thus the operational state is the posterior density operator modulo an
observable null space, not necessarily the privileged density operator.

---

## 4. Hankel matrices and predictive-state representations

### 4.1 Finite linear dimension

Define the controlled Hankel matrix

\[
\mathsf H_{h,\tau}=p(h,\tau).
\tag{6}
\]

In Liouville notation,

\[
\mathsf H_{h,\tau}
=
\langle\!\langle F_\tau\mid\widetilde\rho_h\rangle\!\rangle,
\tag{7}
\]

so a \(d\)-dimensional Markovian quantum model obeys

\[
\operatorname{rank}\mathsf H\le d^2.
\tag{8}
\]

This does not bound the number of distinct causal states. A qubit can have
infinitely many posterior states while its linear predictive dimension is at
most four.

### 4.2 Core tests and action operators

If \(\operatorname{rank}\mathsf H=r\), choose \(r\) independent core tests and
define

\[
q(h)=
\big(p(\tau_1\mid h),\ldots,p(\tau_r\mid h)\big)^\top.
\tag{9}
\]

Every other test probability is linear in \(q(h)\). Each action--outcome symbol
has an unnormalized operator \(B_{ao}\):

\[
\widetilde q(hao)=B_{ao}q(h),
\qquad
q(hao)=
\frac{B_{ao}q(h)}
{\mathbf n^\top B_{ao}q(h)}.
\tag{10}
\]

This is a linear predictive-state representation (PSR). It describes state in
terms of observable predictions rather than hidden labels. The foundational
construction is due to Littman, Sutton, and Singh
([primary paper](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)).

Define shifted Hankel blocks

\[
\mathsf H^{ao}_{h,\tau}=p(h,ao,\tau).
\tag{11}
\]

An invertible rank-\(r\) history/test basis block and the shifted blocks
determine \(B_{ao}\) in a chosen factorization gauge. Action meanings are then
learned through:

- equality or distinction of shifted operators;
- inverse, order, and idempotence relations;
- commutators and group composition;
- fixed predictive states and spectra;
- permutations of landmark response fields;
- changes in goal-conditioned value.

---

## 5. Finite observability

Define nested reachable-state spaces

\[
\mathcal R_0=\operatorname{span}\{\rho_0\},
\]

\[
\mathcal R_{L+1}
=
\operatorname{span}\left(
\mathcal R_L
\cup
\{\mathcal E_o^a(R):R\in\mathcal R_L,\ a,o\}
\right),
\tag{12}
\]

and observable-effect spaces

\[
\mathcal O_0=\operatorname{span}\{I\},
\]

\[
\mathcal O_{L+1}
=
\operatorname{span}\left(
\mathcal O_L
\cup
\{\mathcal E_o^{a*}(F):F\in\mathcal O_L,\ a,o\}
\right).
\tag{13}
\]

### Proposition 2 (finite closure)

Both sequences stabilize after at most \(d^2-1\) strict dimension increases.
Once equality occurs at one step, all longer states or effects remain in the
same space.

#### Proof

The spaces are nested and have dimension at most \(d^2\). Each strict
inclusion raises dimension. At equality the recursion makes the space
invariant under every branch map; induction gives closure at all later
lengths. \(\square\)

### Theorem 3 (finite state-equivalence criterion)

Let \(\mathcal O_\infty\) be the stabilized observable space. Reachable states
\(\rho,\sigma\) are predictively equivalent iff

\[
\operatorname{Tr}[F(\rho-\sigma)]=0
\tag{14}
\]

for every member of a finite basis of \(\mathcal O_\infty\).

#### Proof

Necessity follows from equal test probabilities. Every continuation effect is
a linear combination of the stabilized basis, proving sufficiency. \(\square\)

A state family \(\{\rho_x\}\) is globally observable when every distinct pair
is separated by some \(F\in\mathcal O_\infty\). Full tomography
\(\mathcal O_\infty=\operatorname{Herm}(\mathcal H)\) is sufficient but not
necessary; separation on the accessible manifold is enough.

---

## 6. Four meanings of informativeness

### 6.1 State information

For an ensemble \(X\sim\mu\), immediate information from action \(a\) is

\[
I_\mu(X;O\mid a)
=
\sum_{x,o}\mu(x)p(o\mid x,a)
\log
\frac{p(o\mid x,a)}
{\sum_{x'}\mu(x')p(o\mid x',a)}.
\tag{15}
\]

This is prior dependent. For a future schedule \(\beta\), delayed information
is

\[
I_\mu(X;O_{1:L}\mid a,\beta).
\tag{16}
\]

An action can have no informative immediate outcome but rotate a hidden
coherence into a basis revealed later.

### 6.2 Fisher observability

For a smooth predictive family \(p_\tau(x)\),

\[
J(x)
=
\sum_\tau p_\tau(x)
\nabla\log p_\tau(x)\nabla\log p_\tau(x)^\top.
\tag{17}
\]

Resolving a local \(k\)-dimensional coordinate requires

\[
\operatorname{rank}J(x)=k.
\tag{18}
\]

The quantum Fisher information bounds all measurement Fisher information.
Full quantum Fisher rank is necessary for any probe to see all local
directions, but a restricted catalog may still fail to attain it.

### 6.3 Action identity information

If a hidden variable \(A\) selects a button, then

\[
I(A;Y_{\rm future}\mid\text{preparation,test})
\tag{19}
\]

quantifies experimental action identification. It is distinct from (15): a
state-independent movement flag may reveal no location but the transformation
can be recognized by later landmark statistics.

### 6.4 Instrument distinguishability

Represent an instrument as a channel with a classical register:

\[
\mathfrak I_a(\rho)
=
\sum_o\proj{o}\otimes\mathcal E_o^a(\rho).
\tag{20}
\]

With equal priors and unrestricted preparations, ancillas, and measurements,

\[
P^\star_{\rm discr}(a,b)
=
\frac12+\frac14
\|\mathfrak I_a-\mathfrak I_b\|_\diamond.
\tag{21}
\]

This device-level diamond distance can overstate the resources available to
the project agent. Define accessible distinguishability

\[
\delta_{\rm acc}(a,b)
=
\sup_{\rho\in\mathcal R_{\rm phys}}
\sup_{F\in\mathcal O_{\rm phys}}
\left|
\operatorname{Tr}
\left[F(\mathfrak I_a-\mathfrak I_b)(\rho)\right]
\right|.
\tag{22}
\]

It vanishes exactly when the experimental grammar cannot distinguish the
actions. For diamond-norm operational interpretations see
[Regula, Takagi, and Gu](https://arxiv.org/abs/2102.07773).

### 6.5 Causal displacement

For predictive metric \(d_{\rm pred}\), define

\[
\Delta_{ao}(z)=d_{\rm pred}(z,T_{ao}z).
\tag{23}
\]

A retry movement can satisfy

\[
I(X;O\mid a)=0,
\qquad
\Delta_{a,\mathrm s}(z)>0.
\]

Information gained, action identity, channel distinguishability, and causal
movement must therefore be reported separately.

---

## 7. Action equivalence, finite identification, and gauge

### Definition 2 (operational action equivalence)

Actions \(a,b\) are equivalent up to outcome permutation \(\pi\) when

\[
p(o,\tau\mid h,a)
=
p(\pi(o),\tau\mid h,b)
\tag{24}
\]

for every reachable history, branch, and future test.

### Theorem 4 (finite action-equivalence test)

Let \(\{R_i\}\) span \(\mathcal R_\infty\) and \(\{F_j\}\) span
\(\mathcal O_\infty\). Actions \(a,b\) are operationally equivalent under
\(\pi\) iff

\[
\operatorname{Tr}\left[
F_j(\mathcal E_o^a-\mathcal E_{\pi(o)}^b)(R_i)
\right]=0
\tag{25}
\]

for all \(i,j,o\).

#### Proof

Necessity is immediate. Every reachable state and continuation effect is a
linear combination of the bases, so bilinearity extends the equality to every
history and test. \(\square\)

If both spaces are full operator space, (25) identifies branch superoperators.
Otherwise only the operational restriction is identifiable.

### 7.1 Similarity gauge

With probabilities \(e_\tau^\top B_{ao}s_h\), any invertible \(G\) gives

\[
s_h\mapsto Gs_h,\qquad
e_\tau^\top\mapsto e_\tau^\top G^{-1},\qquad
B_{ao}\mapsto GB_{ao}G^{-1},
\tag{26}
\]

without changing an experiment. This is the PSR/gate-set similarity gauge.
Self-consistent gate-set tomography has the same fundamental issue; see
[Nielsen et al.](https://arxiv.org/abs/2009.07301) and the operational
gauge-free approach of
[Di Matteo et al.](https://arxiv.org/abs/2007.01470).

Other unavoidable freedoms include:

- permutations of action and outcome tokens;
- unitary mixing of Kraus operators within one observed CP branch;
- automorphisms of a learned group with no external landmark;
- translations, rotations, and reflections of a recovered Euclidean atlas.

### Proposition 5 (operational invariants)

A minimal exact PSR is unique up to similarity. Hence future probabilities,
predictive classes, rank, operational action equality, operator spectra,
polynomial composition relations, fixed-subspace dimensions, Bellman values,
and the Euclidean congruence class of goal geometry are identifiable.
Individual latent coordinates and raw matrix entries are not.

---

## 8. Two-dimensional observability and learned vector fields

For candidate place coordinate \(x=(x^1,x^2)\), define finite predictive
features after closure:

\[
\Phi(x)=
\big(p(\tau_1\mid x),\ldots,p(\tau_M\mid x)\big).
\tag{27}
\]

The family is:

- globally predictive if \(\Phi\) is injective;
- locally 2D if \(\operatorname{rank}D\Phi(x)=2\);
- statistically regular if accessible Fisher information is positive
  definite;
- robust if its smallest Fisher eigenvalue or Jacobian singular value stays
  bounded away from zero.

Finite injectivity is much weaker than a robust 2D manifold. A dense 1D curve
can contain arbitrarily many distinct labelled points while its differential
rank remains one.

An identified action branch maps causal states \(T_{ao}:z\mapsto z'\). After
recovering coordinates \(X(z)\), define

\[
v_{ao}(z)=X(T_{ao}z)-X(z).
\tag{28}
\]

Homogeneous translation requires \(v_{ao}(z)\) or its stochastic distribution
to be independent of \(z\) away from boundaries. Locality bounds its norm.
Action direction is inferred after geometry and is unique only up to the
atlas's Euclidean gauge.

---

## 9. Necessary and sufficient finite conditions for a 2D operational atlas

### Definition 3

For selected causal states \(Z=\{z_i\}_{i=1}^n\), goals \(\{g_j\}\), and
identified actions, an exact 2D operational spatial atlas has:

1. injective coordinates \(X_i\in\mathbb R^2\) with affine span two;
2. optimal costs \(D_{ij}=\|X_i-X_j\|_2\);
3. proper reachability;
4. statistically identifiable local action displacement laws;
5. state/action distinctions derived from sequence probabilities, not labels.

### Theorem 6 (finite operational-atlas criterion)

After exact Hankel closure, such an atlas exists iff:

#### P. Predictive separation

\[
i\ne j\Longrightarrow
\exists\tau:
p(\tau\mid z_i)\ne p(\tau\mid z_j).
\tag{P}
\]

#### A. Action identifiability

Shifted Hankel blocks determine branch transformations on the reachable
predictive quotient, and differently interpreted action classes fail (25):

\[
\delta_{\rm acc}(a,b)>0.
\tag{A}
\]

#### B. Bellman realizability

Each cost column is the minimal nonnegative solution

\[
D_{jj}=0,
\]

\[
D_{ij}
=
\min_a\left[c(i,a)+\sum_kP_a(i,k)D_{kj}\right].
\tag{B}
\]

#### M. Metric conditions

\[
D_{ii}=0,\quad D_{ij}=D_{ji}>0,\quad
D_{ik}\le D_{ij}+D_{jk}.
\tag{M}
\]

#### E. Exact planar Euclidean condition

\[
J=I-\frac1n\mathbf1\mathbf1^\top,
\qquad
B_D=-\frac12JD^{\circ2}J,
\]

\[
B_D\succeq0,\qquad \operatorname{rank}B_D=2.
\tag{E}
\]

#### L. Learned local action consistency

Coordinates reconstructed from \(B_D\) obey

\[
P_{ao}(i,k)>0
\Longrightarrow
X_k-X_i\in S_{ao},
\tag{L1}
\]

where the response-derived displacement law \(S_{ao}\) is local and
site-independent away from boundaries. Distinct semantic actions have
different operational response/displacement signatures unless declared
equivalent.

#### Proof

An operational atlas necessarily separates its points and actions, satisfies
Bellman optimality, has Euclidean metric costs, and supports local consistent
increments, giving P--L. Conversely P supplies operational points and A
well-defined action transformations. B supplies actual optimal costs, M makes
them a metric, and Schoenberg's theorem E constructs

\[
X=Q_2\Lambda_2^{1/2}
\]

from the two positive eigenpairs with
\(\|X_i-X_j\|=D_{ij}\). L makes the identified transformations local motions
in those coordinates. Every ingredient came from finite probabilities, costs,
and goal rules. \(\square\)

If the points affinely span the plane, the result is unique up to translation,
rotation, and reflection. P--E without L gives a 2D goal metric but not motion:
a teleportation table can have Euclidean costs.

---

## 10. Minimal qubit example and obstruction

### 10.1 Exactly soluble Pauli square

Choose qubit Bloch vector

\[
r_{00}=(1,1,1)/\sqrt3.
\]

Pauli conjugations generate

\[
\begin{aligned}
r_{00}&=(+,+,+)/\sqrt3,\\
r_{10}&=(+,-,-)/\sqrt3,\\
r_{01}&=(-,-,+)/\sqrt3,\\
r_{11}&=(-,+,-)/\sqrt3.
\end{aligned}
\tag{29}
\]

They form a tetrahedron and satisfy

\[
\operatorname{Tr}(\rho_i\rho_j)=1/3\quad(i\ne j).
\]

Use tetrahedral SIC effects

\[
E_j=\frac14(I+r_j\cdot\sigma).
\tag{30}
\]

The exact response matrix is

\[
p(j\mid i)=
\begin{cases}
1/2,&j=i,\\
1/6,&j\ne i.
\end{cases}
\tag{31}
\]

Each place has a unique response maximum. The unknown Pauli buttons are learned
as response permutations:

\[
X:(u,v)\mapsto(u+1,v),\qquad
Z:(u,v)\mapsto(u,v+1)\pmod2.
\tag{32}
\]

Use deterministic \(X,Z\) cost-one moves and retry diagonal \(Y\simeq XZ\):

\[
K_{\rm s}=2^{-1/4}Y,\qquad
K_{\rm f}=\sqrt{1-2^{-1/2}}\,I.
\tag{33}
\]

The diagonal mean cost is \(\sqrt2\), giving

\[
D=
\begin{pmatrix}
0&1&1&\sqrt2\\
1&0&\sqrt2&1\\
1&\sqrt2&0&1\\
\sqrt2&1&1&0
\end{pmatrix}.
\tag{34}
\]

This is exactly a unit square: \(B_D\succeq0\) and
\(\operatorname{rank}B_D=2\). It is the minimal finite example with
nonorthogonal predictive places, statistically learned actions, and exact 2D
hodology. Its projective translation group closes at \(\mathbb Z_2^2\), so it
does not scale.

### 10.2 Irrational qubit phases are physically rank one

Consider

\[
\ket{\psi_{xy}}
=
\frac{\ket0+e^{i(x\alpha+y\beta)}\ket1}{\sqrt2},
\tag{35}
\]

with \(\alpha,\beta,2\pi\) rationally independent. The integer map is
injective, but the physical state depends only on
\(\theta=x\alpha+y\beta\). Its quantum Fisher matrix is

\[
F_Q=
\begin{pmatrix}\alpha\\\beta\end{pmatrix}
\begin{pmatrix}\alpha&\beta\end{pmatrix},
\tag{36}
\]

of rank one. No measurement can robustly resolve two local coordinates. The
orbit is dense on one circle, its inverse coordinate map is discontinuous,
and finite-tolerance goals admit near-alias shortcuts.

A qubit pure-state manifold is the 2D Bloch sphere, so noncommuting controls
can explore two curved directions. The no-go is specifically for a robust flat
plane of two commuting phases: a maximal connected commuting qubit unitary
subgroup has only one relative phase.

---

## 11. Robust qutrit phase atlas

### 11.1 States and translations

For odd \(m\ge5\), let \(\omega=e^{2\pi i/m}\) and

\[
\ket{\psi_{xy}}
=
\frac{\ket0+\omega^x\ket1+\omega^y\ket2}{\sqrt3},
\qquad (x,y)\in\mathbb Z_m^2.
\tag{37}
\]

The commuting actions

\[
U=\operatorname{diag}(1,\omega,1),\qquad
V=\operatorname{diag}(1,1,\omega)
\tag{38}
\]

translate \(x\) and \(y\). There are \(m^2\) distinct nonorthogonal states in
fixed Hilbert dimension three.

### 11.2 Full-rank phase information

For continuous phases and generators

\[
N_1=\proj1,\qquad N_2=\proj2,
\]

the pure-state quantum Fisher matrix is four times their covariance:

\[
F_Q=\frac49
\begin{pmatrix}
2&-1\\
-1&2
\end{pmatrix}.
\tag{39}
\]

Its eigenvalues are \(4/9\) and \(4/3\). Both phase directions are locally
observable in principle. A \(d\)-level diagonal pure-state orbit has at most
\(d-1\) relative phases, so a qutrit is minimal for two independent commuting
phase coordinates.

### 11.3 Covariant phase probe

Define

\[
E_{uv}=\frac3{m^2}\proj{\psi_{uv}}.
\tag{40}
\]

Root-of-unity cancellation yields \(\sum_{u,v}E_{uv}=I_3\). The response is

\[
q_{xy}(u,v)
=
\frac{
|1+\omega^{x-u}+\omega^{y-v}|^2
}{3m^2}.
\tag{41}
\]

It is covariant:

\[
q_{x+r,y+s}(u,v)=q_{xy}(u-r,v-s).
\tag{42}
\]

Its unique maximum is at \((u,v)=(x,y)\), because the triangle inequality
reaches three only when both phases are one. Thus all response vectors are
distinct and every translation creates a distinct response permutation.
Outcome symbols may be randomly renamed; their relational transition
structure remains.

### 11.4 Learned algebra

Anchor, unknown-button, and future-probe experiments identify response shifts.
Longer words identify

\[
UV=VU,\qquad U^m=V^m=I,\qquad U^{-1},V^{-1}
\tag{43}
\]

as relations of shifted Hankel operators. The recovered object is the action
of \(\mathbb Z_m^2\), up to token permutations, generator changes, group
automorphisms, and geometric orientation. No absolute compass is identifiable.

---

## 12. Exact and local qutrit geometries

### 12.1 Exact square

Select predictive states

\[
\{(0,0),(1,0),(0,1),(1,1)\}.
\]

Use deterministic \(U,V\) moves of cost one and a retry \(UV\) diagonal with
success probability \(1/\sqrt2\). Goals are predictive response classes
\(q_{xy}\), not basis outcomes. The cost matrix is (34).

Compared with a labelled grid:

1. places are defined by future response fields;
2. actions are learned as transformations of those fields;
3. commutation and composition are tested from sequence probabilities;
4. square coordinates come from Bellman costs;
5. buttons receive direction vectors only after MDS.

### 12.2 Exact \(3\times3\) patch

For \(m\ge7\), select \(\{0,1,2\}^2\). Exact Euclidean all-pairs cost follows
if the catalog contains all primitive patch slopes and reversals, with retry

\[
p_{rs}=\frac1{\sqrt{r^2+s^2}}.
\tag{44}
\]

This is exactly soluble and action semantics remain learned through response
shifts. It is not yet a derivation of Euclidean norm: represented primitive
lengths are encoded in retry rates.

### 12.3 Fixed local catalog

Using only axial and diagonal translations tests a stronger notion of
locality. Predictive states and action meanings remain genuinely 2D, while
the cost becomes octile/Finsler rather than exactly Euclidean. This is an
important negative control: learned action meaning is necessary but not
sufficient for ordinary geometry.

---

## 13. Decisive controls

### 13.1 Cloned actions

Give differently named buttons identical branch maps. Equation (25) and
\(\delta_{\rm acc}\) must report equivalence. Any learner that separates them
is reading labels, noise, or privileged metadata.

### 13.2 Null probe

Replace every effect by

\[
E_o=I/|\mathcal O|.
\tag{45}
\]

If branch states are also input independent and all later probes are null,
predictive places have identical Hankel rows. Geometry can still be tracked by
a hand-coded history counter, exposing memory rather than quantum-statistical
provenance.

### 13.3 Twirled random-unitary actions

For translation group \(G\), define

\[
\mathcal T(\rho)
=
\frac1{|G|}\sum_{g\in G}W_g\rho W_g^\dagger.
\tag{46}
\]

Replace every move by

\[
\widetilde{\mathcal E}^{a}
=
\mathcal T\circ\mathcal U_a.
\]

Group invariance gives

\[
\mathcal T\circ\mathcal U_a=\mathcal T
\]

for every \(a\). All actions become operationally identical:
\(\delta_{\rm acc}(a,b)=0\). This preserves valid random-unitary physics while
destroying persistent directional meaning.

### 13.4 Fixed versus episode-wise outcome permutation

A fixed unknown permutation of probe symbols is gauge and must not hurt.
Independently rescrambling symbols every episode destroys cross-episode
landmark identity and should destroy learned semantics.

### 13.5 Relational anchor

Under a transitive uniform prior, a translation may only permute hidden
components and have zero immediate marginal signature. Compare:

1. anchor--move--probe;
2. uniform start--move--probe;
3. uniform start--probe--move--probe;
4. uniform start--null probe--move--probe.

The first and third provide a stable relational reference. This shows that
action meaning belongs to an experimental grammar, not an isolated button.

### 13.6 Label-permutation audit

Randomly permute action and outcome tokens independently in every seed, while
keeping each seed's permutation fixed. All invariant metrics must be stable.
Privileged alignment is allowed only after analysis is frozen.

---

## 14. Experiment specification: finite-Hankel reconstruction

### Environment

- Hilbert dimension \(d=3\).
- Phase order \(m=7\), then \(m=5,9,11\).
- Orbit and POVM from (37)--(41).
- Hidden movement tokens: a fixed random permutation of
  \(\{U,U^{-1},V,V^{-1}\}\).
- Probe outcomes: one fixed random permutation of \(m^2\) symbols.
- Reference: heralded anchor \(\rho_{00}\), later replaced by a learned
  reference history.
- Unit intervention costs.

### Data grammar

Collect:

- anchor--probe;
- anchor--action--probe;
- anchor--action--action--probe;
- repeated “germ” words that amplify action differences;
- probe--action--probe transitions;
- held-out random words longer than training words.

Increase word length until cross-validated Hankel rank and predictive
likelihood stabilize. Exact qutrit rank is at most nine.

### Learned objects

1. joint and shifted Hankel matrices;
2. minimal predictive rank;
3. core histories/tests and a spectral PSR;
4. action-equivalence classes;
5. inverse, order, and commutator relations;
6. action Cayley graph on predictive landmarks;
7. goal-conditioned Bellman costs;
8. 2D atlas and post-hoc action vectors.

### Acceptance metrics

#### Prediction

- held-out negative log likelihood;
- calibration and total-variation error on unseen words;
- Hankel singular-value confidence intervals;
- rank stability across maximum word lengths.

#### State observability

- minimum pairwise response divergence;
- smallest singular value of the landmark response matrix;
- Fisher eigenvalues and condition number;
- causal-state confusion.

#### Action identifiability

- pairwise accessible distinguishability;
- shifted-Hankel residual;
- inverse residual
  \(\|B_aB_{a^{-1}}-I\|\);
- commutator residual
  \(\|B_UB_V-B_VB_U\|\);
- order residual \(\|B_U^m-I\|\);
- clustering accuracy only after privileged permutation alignment.

#### Geometry

- Bellman residual and cost calibration;
- symmetry and triangle violations;
- Schoenberg negative eigenmass and Gram rank;
- 1D/2D/3D MDS stress;
- privileged Procrustes recovery after freezing analysis;
- displacement covariance and locality.

#### Gauge robustness

- unchanged predictions under random similarity transformations;
- unchanged scientific results under token permutations;
- invariant spectra and algebraic relations before alignment;
- explicit record of any visualization gauge.

### Exact checks

In the analytic model:

1. POVM/Kraus completeness and covariance residuals below \(10^{-12}\);
2. all qutrit response vectors distinct;
3. action permutations satisfy (43);
4. the minimal square equals (34) below \(10^{-10}\);
5. its Schoenberg Gram matrix is PSD of rank two;
6. action directions recover only up to square symmetry;
7. fixed token permutations change no invariant;
8. cloned and twirled actions have zero accessible distinguishability;
9. null probes collapse predictive place rank;
10. episode-wise rescrambling destroys cross-episode semantics.

Finite-data nulls must show no significant semantic separation after
multiple-comparison correction.

---

## 15. Experiment specification: learned-action hodology

### Stage A: minimal cell

Run both the qubit Pauli square and qutrit four-state square. Randomize tokens,
learn response transformations, then solve costs. Compare predictive rank,
response conditioning, probe-strength sample complexity, hidden-slip
robustness, and exact 2D recovery.

### Stage B: nine-goal patch

Use \(m=7\) and \(\{0,1,2\}^2\). Compare:

1. exact patch-slope retries;
2. local axial/diagonal actions;
3. learned retry probabilities;
4. twirled actions;
5. null probe;
6. cloned action;
7. episode-rescrambled outcomes.

The exact model should satisfy P--L. The local model should satisfy predictive
and action criteria but fail exact Euclidean E. Twirling should fail A; null
probing should fail P.

### Stage C: hidden start and active sensing

Hide the initial state and offer multiple inequivalent weak phase probes.
Compare an exact Bayesian filter, finite-history learner, spectral PSR, and
recurrent predictive model. Policies choose sense, move, or commit. Report:

- preparation-label success;
- present-state fidelity;
- sensing, movement, and total interventions separately;
- posterior calibration and held-out predictive likelihood;
- action identification under belief uncertainty.

This asks whether an action retains a learned direction when its effect is a
transformation of a posterior over predictive places.

---

## 16. Minimality results

### Proposition 7 (one-shot place bound)

If \(N\) place states must be recognized without error by one common POVM,
their supports are mutually orthogonal and \(N\le d\). Sequence statistics
evade the premise, not the one-shot discrimination bound.

### Proposition 8 (commuting phase dimension)

A pure orbit under commuting diagonal unitaries in dimension \(d\) has at most
\(d-1\) independent relative-phase directions.

#### Proof

There are \(d\) diagonal phases and one projectively irrelevant common phase.
\(\square\)

### Proposition 9 (qubit commuting-phase no-go)

Every smooth qubit orbit generated by commuting diagonal unitaries has quantum
Fisher rank at most one, so no probe grammar makes it a robust smooth 2D phase
manifold.

### Proposition 10 (finite qubit possibility)

The qubit construction (29)--(34) satisfies P--L and is an exact operational
2D atlas. Thus a finite 2D cell and a scalable 2D manifold have different
minimality requirements.

---

## 17. Interpretation rules and failure modes

1. **High mutual information without movement.** A measurement can reveal
   state then reset every branch to one state. Report information and causal
   displacement separately.
2. **Positive diamond distance but no accessible meaning.** Channels may
   differ only on preparations or effects unavailable to the agent. Use
   \(\delta_{\rm acc}\) for first-person semantics.
3. **Identifiable actions without space.** Buttons can be distinguished while
   Bellman geometry is a tree, directed graph, or high-dimensional metric.
4. **Euclidean costs without observable places.** A hand-coded counter can
   pass Schoenberg E under null outcomes. P/A/L reveal the leakage.
5. **Gauge-fixed overclaim.** Target-gate alignment can make incorrect action
   vectors look right. Make claims in invariants first.
6. **Too-short tests.** One-step equality does not imply causal equality.
   Increase test length through observable closure and validate longer words.
7. **Finite-sample rank hallucination.** Noise makes empirical Hankel matrices
   full rank. Use bootstrap singular-value intervals and held-out likelihood,
   not a visual elbow alone.

---

## 18. Data that must be retained

Save:

- raw sequence counts and train/test word splits;
- hidden token permutations for privileged audit only;
- Hankel singular values with uncertainty;
- core histories and tests;
- learned state/effect/action operators;
- all gauge transformations used for plots;
- shifted-Hankel and algebraic residuals;
- action equivalence/confidence matrices;
- predictive landmark response fields;
- Bellman cost matrices and failure rules;
- Schoenberg spectra and MDS coordinates;
- action increments in recovered coordinates;
- cloned, null, twirled, and rescrambled controls;
- exact targets, seeds, and manifests.

This preserves future theoretical context and prevents a geometric claim from
depending on an unrecoverable plotting alignment.

---

## 19. Outlook

### Spectral PSR before recurrent learning

Solve the qutrit benchmark first with a spectral PSR. It exposes rank,
shifted operators, and gauge. Then train a recurrent model and compare
held-out predictions, calibration, action algebra, and navigation regret
against the exact representation.

### Learn a continuous action family

Replace discrete headings with an unlabeled continuous control. Learn its
composition and future-response shifts, then test whether the family has
circle topology and an elliptical infinitesimal velocity body. Its coordinate
parameterization is gauge; topology, composition, and induced cost are not.

### Compare candidate local metrics

Natural tensors include:

- Fisher information of future outcomes;
- quantum Fisher information of controllable states;
- quadratic control energy;
- Hessian of local goal value;
- diffusion/heat-kernel geometry of the learned transition operator.

The important question is when these independently motivated tensors agree.

### Noncommutativity, connection, and curvature

The learned loop

\[
B_aB_bB_a^{-1}B_b^{-1}
\]

is similarity-gauge covariant. Its invariant spectrum and effect on future
predictions provide an operational holonomy observable. A flat qutrit phase
atlas should give trivial base commutators; internal transformations around
loops can define a connection and eventually curvature.

### Base and fiber

The spatial base should quotient predictive states by internal differences
that leave place-like movement values unchanged. The fiber retains posterior
uncertainty, measurement context, goal progress, and internal quantum state.
Horizontal, vertical, and coupled actions can then be learned from their
future-statistical effects rather than assigned by architecture.

---

## 20. Recommended first production study

1. Implement the \(m=7\) qutrit phase orbit and covariant POVM.
2. Randomly rename four moves and 49 outcomes.
3. Collect anchor/action/probe words through length six and held-out words
   through length ten.
4. Fit a spectral PSR with bootstrap rank selection.
5. Recover inverses, commutation, order, and response permutations.
6. Build the exact four-goal square, then the nine-goal patch.
7. Run fixed-permutation, cloned-action, null-probe, episode-rescrambled, and
   twirled controls.
8. Evaluate P--L without privileged alignment.
9. Align to hidden coordinates only for final Procrustes validation.

The decisive success statement is:

> From randomly named interventions and outcomes alone, a finite predictive
> model reconstructs an action algebra, identifies operational landmarks,
> predicts held-out sequences, induces Bellman costs with a rank-two Euclidean
> Gram matrix, and assigns consistent local displacement vectors to action
> classes, while matched operational nulls fail exactly where predicted.

That would show how an agent learns what its actions mean before interpreting
their goal geometry as space.

---

## 21. References and provenance

- M. L. Littman, R. S. Sutton, and S. Singh, “Predictive representations of
  state,” *NeurIPS 14* (2001).
  [Primary paper](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf).
- E. Nielsen et al., “Gate Set Tomography,” arXiv:2009.07301 (2020).
  [Primary preprint](https://arxiv.org/abs/2009.07301).
- O. Di Matteo et al., “Operational, gauge-free quantum tomography,”
  arXiv:2007.01470 (2020).
  [Primary preprint](https://arxiv.org/abs/2007.01470).
- B. Regula, R. Takagi, and M. Gu, “Operational applications of the diamond
  norm and related measures,” *Quantum* **5**, 522 (2021).
  [Primary preprint](https://arxiv.org/abs/2102.07773).
- K. Kraus, *States, Effects, and Operations* (Springer, 1983).
- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
  Information* (Cambridge University Press, 2010).
- J. Watrous, *The Theory of Quantum Information* (Cambridge University
  Press, 2018).
- A. S. Holevo, *Probabilistic and Statistical Aspects of Quantum Theory*
  (2011).
- C. W. Helstrom, *Quantum Detection and Estimation Theory* (1976).
- I. J. Schoenberg, “Remarks to Maurice Fréchet's article ...,” *Annals of
  Mathematics* **36**, 724--732 (1935).
- M. L. Puterman, *Markov Decision Processes* (Wiley, 1994).

The qubit response calculation, qutrit covariant response field, finite
operational-atlas theorem, and proposed controls are derived here for this
project.
