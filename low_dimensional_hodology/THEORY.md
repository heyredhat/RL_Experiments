# Low-dimensional quantum hodology

## Exact lattice geometry from a qubit or qutrit

**Status.** This note develops the theory needed to decide whether a small
Hilbert space can support a much larger spatial goal geometry. It gives two
exactly solvable constructions:

1. a **qubit Pauli square**, which is the smallest nondegenerate lattice cell;
2. a **qutrit phase torus**, whose arbitrarily large finite subgroups contain
   exact \(3\times3\) (and larger) square-lattice patches.

The conclusions are deliberately qualified. Low Hilbert dimension does **not**
bound the number of distinguishable density operators, and sequence goals can
indeed have a larger state space than the quantum system. On the other hand,
unrestricted history automata can manufacture any finite geometry even with a
trivial one-dimensional system. A useful construction must therefore report
both the physical Hilbert dimension and the memory used to recognize goals.

The qutrit construction below is the most promising *robust manifold* model.
An exact qubit construction also scales to every finite patch by encoding
\(\mathbb Z^2\) densely in one phase circle; it is algebraically valid but
unstable under finite-resolution physical verification. The qutrit model's
places are nonorthogonal phase-orbit states, not basis states. Its movement
instruments are random-unitary Kraus instruments. Its optimal all-pairs cost
matrix is known in closed form and is exactly the Euclidean metric on any
small square patch. It therefore gives simulation a ground truth stringent
enough to expose estimation, planning, and representation-learning errors.

---

## 1. Three dimensions that must not be conflated

### 1.1 Physical Hilbert dimension

The hidden quantum system has Hilbert space

\[
\mathcal H\cong\mathbb C^d.
\]

An action (a) is a quantum instrument

\[
\mathcal I^{a}=\{\mathcal E^{a}_{o}\}_{o\in\mathcal O_a},
\qquad
\mathcal E^{a}_{o}(\rho)
=\sum_k K^{a}_{o,k}\rho K^{a\dagger}_{o,k},
\]

with

\[
\sum_{o,k}K^{a\dagger}_{o,k}K^{a}_{o,k}=I_d.
\]

The probability and conditional state are

\[
p(o\mid\rho,a)=\operatorname{Tr}\mathcal E^a_o(\rho),
\qquad
\rho'=\frac{\mathcal E^a_o(\rho)}{p(o\mid\rho,a)}.
\]

The real vector space of Hermitian operators has dimension (d^2), while the
affine set of normalized density operators has dimension (d^2-1). A qubit
therefore has a three-dimensional continuum of states; it is not limited to
two possible physical states. What *is* limited to (d) is the number of
states that can be perfectly distinguished in one shot.

### 1.2 Predictive or causal-state dimension

The agent observes a history

\[
h_t=(a_0,o_0,\ldots,a_{t-1},o_{t-1}).
\]

Two histories are predictively equivalent when every possible future action
string produces the same distribution of future outcomes:

\[
h\sim_{\rm pred}h'
\iff
p(\mathbf o\mid h,\mathbf a)
=p(\mathbf o\mid h',\mathbf a)
\quad\text{for all }(\mathbf a,\mathbf o).
\]

The equivalence classes are operational causal states. In a known Markovian
quantum model, the posterior density operator is a sufficient predictive
state. There may be infinitely many such states even for a qubit. At the same
time, the linear dimension of the controlled process is bounded: the Hankel
matrix of continuation probabilities factors through operator space and has
rank at most (d^2). This is a linear-rank bound, not a bound on the number of
distinct posterior states.

### 1.3 Goal-automaton or controller dimension

A sequence goal is naturally a language over action--outcome symbols. A
regular goal can be recognized by a deterministic finite automaton

\[
q_{t+1}=\delta(q_t,a_t,o_t),
\qquad q_t\in Q,
\]

with accepting states (F_g\subseteq Q). The correct Markov state of the
goal-conditioned control problem is then

\[
(\rho_t,q_t)\in\mathcal D(\mathcal H)\times Q.
\]

The automaton size (M=|Q|) is independent of (d). If the goal condition
depends on the last (L) symbols, (M) can grow exponentially with (L).
A recurrent controller can represent the same information continuously rather
than as an explicit table, but the information has not disappeared.

This distinction is the key to the low-dimensional question:

> A qutrit can support nine or nine hundred sequence goals because goal space
> is a space of controlled histories, not a set of mutually orthogonal basis
> vectors. Whether the resulting geometry is physically informative depends
> on how much of its coordinate is carried by the quantum process and how much
> is carried only by the goal automaton.

---

## 2. The universality result—and why it is too easy

### Proposition 1 (finite-automaton universality)

Let (G=(V,E)) be any finite directed graph with positive edge costs. If goal
recognition may use an unrestricted finite automaton, then a system with
physical Hilbert dimension \(d=1\) can realize the shortest-path hodological
geometry of (G) exactly.

### Proof

Take one controller state (q_v) for each vertex (v\in V). For every edge
(e=(v,w)), provide an action symbol (a_e), and define the goal automaton to
update (q_v\mapsto q_w) when (a_e) is legal. Use the one-dimensional
identity quantum channel for every action, with a single deterministic outcome.
Assign action (a_e) its edge cost. Let goal (g_w) accept exactly at (q_w).
The Bellman equation of this controller is precisely the shortest-path
equation on (G). The physical quantum system does nothing. \(\square\)

If only unit action costs are permitted, replace an edge of integer cost (k)
by a chain of (k) auxiliary automaton states. Rational costs can instead be
implemented as exactly solvable retry processes after a common rescaling.

### Consequence

The bare statement “a qubit plus sequence goals realizes a \(3\times3\) grid”
is not enough. It could simply mean that a nine-state classical counter was
attached to an idle qubit. Every low-dimensional construction should report:

- (d), the physical Hilbert dimension;
- the number and form of quantum instruments;
- (M), the exact goal-monitor memory, or a recurrent-memory estimate;
- whether coordinates are recoverable from future quantum statistics;
- whether coordinates are recoverable merely by replaying the agent's own
  chosen action labels;
- whether the geometry survives uncertain initialization and hidden
  disturbances, when open-loop bookkeeping no longer suffices.

These measurements turn an otherwise vacuous existence proof into a
scientific comparison of where spatial information is stored.

---

## 3. Genuine obstructions

### 3.1 Perfect one-shot place recognition

### Proposition 2 (orthogonality obstruction)

Suppose (N) place goals are physical states \(\rho_1,\ldots,\rho_N\), and a
single final POVM \(\{E_i\}_{i=1}^N\) must recognize them without error:

\[
\operatorname{Tr}(E_i\rho_j)=\delta_{ij}.
\]

Then their supports are mutually orthogonal, and consequently

\[
N\le d.
\]

### Proof sketch

For (i\ne j), positivity and
\(\operatorname{Tr}(E_i\rho_j)=0\) imply that (E_i) vanishes on the support
of \(\rho_j\). Since \(\operatorname{Tr}(E_j\rho_j)=1\) and
\(0\le E_j\le I\), (E_j) acts as the identity on that support. Distinct
supports must therefore be orthogonal. Their dimensions sum to at most (d).
\(\square\)

Thus nine perfectly readable places do require (d\ge9) **if** each place
must be a deterministic outcome of one common final measurement. Sequence
goals evade this premise: a controller may recognize a path, an accumulated
transformation, a repeated statistical test, or a posterior-confidence event.
The evasion is legitimate, but it should be stated explicitly.

No finite protocol acting on a single unknown copy can perfectly discriminate
nonorthogonal alternatives. Repeated *fresh preparations* can make the error
arbitrarily small, but not identically zero at finite sample size unless the
states become orthogonal in the relevant tensor power.

### 3.2 Reversible motion

If a CPTP channel has a CPTP inverse on the entire matrix algebra, it is a
unitary channel. Therefore exact reversible, memoryless translations of the
whole quantum state space must be implemented by unitaries (or by isometries
when input and output dimensions differ).

For a qubit, continuous unitary conjugations act as rotations of the Bloch
sphere. A maximal connected commuting subgroup has rank one: it supplies one
independent periodic phase coordinate, not two. This obstructs a faithful
continuous (U(1)\times U(1)) translation action on a qubit. There is an
important finite exception. Conjugation by Pauli (X) and (Z) commutes,
because their matrix anticommutation differs only by a global phase. It gives
the finite group \(\mathbb Z_2\times\mathbb Z_2\), exactly enough for one
square cell.

A qutrit has two independent relative phases. Its diagonal projective-unitary
subgroup is a two-torus. This is why \(d=3\) is the minimal Hilbert dimension
for a **robust two-dimensional commuting phase manifold** of the type used in
Section 7. The manifold qualification matters. A qubit phase circle can host
a faithful but dense *countable* copy of \(\mathbb Z^2\); Section 6.6 explains
why that algebraic injection is not a stable two-dimensional physical chart.

### 3.3 Open boundaries versus homogeneous reversible translations

A reversible action on a finite state set is a permutation. Repeated
homogeneous translation on a finite set therefore produces cycles, not an
open boundary. An exact finite open grid must do at least one of the following:

1. break homogeneity at the boundary;
2. make boundary actions irreversible;
3. store boundary location in an auxiliary controller;
4. embed the desired open grid as a patch of a larger periodic or continuous
   space.

The qutrit model uses option 4. A \(3\times3\) set of goals is a local patch of
a larger flat phase torus. No boundary rule is needed to define distances
among those goals.

### 3.4 Local discrete actions do not automatically give Euclidean distance

Four unit-cost nearest-neighbor moves on a square lattice produce the
Manhattan metric, not the Euclidean metric. Adding unit-cost diagonal moves
produces a Chebyshev metric. Giving diagonals expected cost \(\sqrt2\) produces
the octile metric, which is exact for offsets ((1,0)) and ((1,1)) but not
for \((2,1)\):

\[
1+\sqrt2\ne\sqrt5.
\]

An exact finite Euclidean all-pairs metric therefore requires either a richer
action repertoire, a continuous control limit with a Riemannian action cost,
or a different definition of difficulty. This explains why the current
nine-level nearest-neighbor construction has low but nonzero 2D stress.

---

## 4. Exact criteria for Euclidean hodology

Let (Z=\{1,\ldots,n\}) be a finite operational state set. An action (a)
has positive cost (c(i,a)) and transition kernel (P_a(i,j)). The optimal
cost to reach goal state (g) is

\[
D_{ig}=\inf_\pi
\mathbb E_\pi\!\left[
\sum_{t=0}^{\tau_g-1}c(z_t,a_t)
\,\middle|\,z_0=i
\right].
\]

### 4.1 Control condition: Bellman realizability

For proper stochastic-shortest-path problems, a proposed matrix (D) is the
optimal hitting-cost matrix if and only if, for every goal (g), its column is
the minimal nonnegative solution of

\[
D_{gg}=0,
\qquad
D_{ig}=\min_a\left[
c(i,a)+\sum_jP_a(i,j)D_{jg}
\right]
\quad(i\ne g).
\tag{B}
\]

Equation (B) is the necessary-and-sufficient *control* test. It does not yet
say that (D) is symmetric or Euclidean.

### 4.2 Metric condition

Ordinary distance additionally requires

\[
D_{ij}=D_{ji},\qquad D_{ii}=0,\qquad
D_{ij}>0\ (i\ne j),\qquad
D_{ik}\le D_{ij}+D_{jk}.
\]

Directed costs, information-gathering costs, and risk-sensitive costs can
violate these conditions. Their failure is scientifically meaningful: it says
the hodology is not an ordinary metric without an additional symmetrization or
state augmentation.

### 4.3 Geometric condition: Schoenberg's theorem

Let (D^{\circ2}) denote elementwise squared distances and

\[
J=I-\frac1n\mathbf1\mathbf1^\top,
\qquad
B=-\frac12JD^{\circ2}J.
\]

A finite symmetric distance matrix embeds **exactly** in Euclidean
\(\mathbb R^k\) if and only if

\[
B\succeq0,
\qquad
\operatorname{rank}(B)\le k.
\tag{E}
\]

When (E) holds, an eigendecomposition
\(B=V\Lambda V^\top\) gives centered coordinates
\(X=V_k\Lambda_k^{1/2}\). Thus (B), the metric axioms, and (E) are a finite
necessary-and-sufficient test for an exact Euclidean hodological point set.

### 4.4 Trajectory condition

Distances alone do not ensure that actions look like motion. Given recovered
coordinates (x_i), one should also test whether each action has a consistent
local displacement law:

\[
P_a(i,j)>0
\Longrightarrow
x_j-x_i\in S_a(i).
\]

For homogeneous space, (S_a(i)) should be approximately independent of
(i) away from boundaries. A strong stochastic form requires the distribution
of increments (x_{t+1}-x_t), conditioned on action (a), to be translation
invariant. This action-consistency condition separates a spatial atlas from an
arbitrary low-stress arrangement of goal costs.

---

## 5. A general retry-channel theorem

The following construction turns group geometry into exactly solvable quantum
hodology.

### Theorem 3 (projective-orbit retry construction)

Let (G) be a finite group with a left-invariant metric (d_G), rescaled so
that

\[
\min_{h\ne e}d_G(e,h)=1.
\]

Suppose (G) has a projective unitary representation

\[
U:G\longrightarrow PU(d)
\]

and a fiducial density operator \(\rho_0\) whose orbit labels are distinct:

\[
U_g\rho_0U_g^\dagger=U_h\rho_0U_h^\dagger
\Longrightarrow g=h.
\]

For each nonidentity displacement (h\), define a two-outcome instrument

\[
K^{(h)}_{\rm s}=\sqrt{p_h}\,U_h,
\qquad
K^{(h)}_{\rm f}=\sqrt{1-p_h}\,I,
\qquad
p_h=\frac1{d_G(e,h)}.
\tag{R}
\]

The outcomes `success` and `failure` are observed. Let goal (g\) mean that
the accumulated successful group product equals (g). Then the optimal
expected number of unit-cost interventions from (x) to (g) is exactly

\[
D(x,g)=d_G(x,g).
\]

### Proof

Instrument (R) is valid because its Kraus Gram sum is (I). On failure the
group coordinate remains (x); on success it becomes (hx) (or (xh),
according to the fixed convention). Repeating the direct displacement
(h=gx^{-1}) succeeds after a geometric number of trials with mean

\[
\frac1{p_h}=d_G(x,g),
\]

so the optimal cost is no larger than the metric distance.

For the reverse inequality, use (V(x)=d_G(x,g)). For any displacement (h),
the one-step Bellman expression is

\[
1+(1-p_h)V(x)+p_hV(hx).
\]

The triangle inequality and left invariance imply

\[
V(x)-V(hx)\le d_G(x,hx)=d_G(e,h)=\frac1{p_h}.
\]

Therefore the Bellman expression is at least (V(x)). No action or adaptive
mixture of actions can improve on (V), while the direct retry policy attains
it. \(\square\)

This theorem is intentionally transparent. It makes the geometry exact by
putting its edge lengths into success probabilities, just as the existing
spatial experiment puts \(\sqrt2\) into the expected cost of a diagonal move.
Its value is analytic control: any departure in learned geometry can be
assigned to finite data, partial observability, model error, or policy error.

---

## 6. Minimal construction: the qubit Pauli square

### 6.1 Physical orbit

Take \(\mathcal H=\mathbb C^2\) and the pure fiducial state whose Bloch vector
is

\[
r_{00}=\frac1{\sqrt3}(1,1,1).
\]

Let (X,Y,Z) be the Pauli matrices. Conjugation by (X) and (Z) commutes,
even though the matrices anticommute, because global phase disappears from a
density operator. The orbit is

\[
\begin{aligned}
r_{00}&=(+,+,+)/\sqrt3,\\
r_{10}&=(+,-,-)/\sqrt3,\\
r_{01}&=(-,-,+)/\sqrt3,\\
r_{11}&=(-,+,-)/\sqrt3.
\end{aligned}
\]

These are the vertices of a regular tetrahedron in the Bloch ball. For any
distinct pair,

\[
\operatorname{Tr}(\rho_{uv}\rho_{u'v'})
=\frac{1+r_{uv}\cdot r_{u'v'}}2
=\frac13.
\]

Thus the four physical states are nonorthogonal and cannot be four outcomes
of a perfectly discriminating qubit measurement. Their *hodological* geometry
will nevertheless be a square, not a tetrahedron. This cleanly demonstrates
that control geometry need not reproduce ambient quantum-state geometry.

### 6.2 Instruments

Use deterministic axial actions

\[
\mathcal X(\rho)=X\rho X,
\qquad
\mathcal Z(\rho)=Z\rho Z,
\]

and a stochastic diagonal action

\[
K_{\rm s}=\sqrt{p}\,Y,
\qquad
K_{\rm f}=\sqrt{1-p}\,I,
\qquad p=\frac1{\sqrt2}.
\]

Successful (X), (Z), and (Y\simeq XZ) toggle the history coordinate by
((1,0)), ((0,1)), and ((1,1)) modulo two, respectively. Failure leaves
both the density operator and coordinate unchanged.

### 6.3 Sequence goals

Let the goal monitor keep the parities of successful toggles:

\[
q_t=(u_t,v_t)\in\mathbb Z_2^2.
\]

Goal (g_{uv}) is the regular language of histories whose accumulated
successful Pauli product lies in projective class (X^uZ^v). The minimal DFA
has four states. No goal is “observe computational-basis state (i).” Each is
a property of the entire action--outcome sequence.

### 6.4 Exact cost

Ordering the goals as (00,10,01,11), the optimal movement-cost matrix is

\[
D=
\begin{pmatrix}
0&1&1&\sqrt2\\
1&0&\sqrt2&1\\
1&\sqrt2&0&1\\
\sqrt2&1&1&0
\end{pmatrix}.
\]

Axial neighbors cost one deterministic action. Opposite corners can be reached
by (X) then (Z) at cost two, or by retrying the diagonal instrument at
mean cost (1/p=\sqrt2). The latter is optimal. Matrix (D) is exactly the
Euclidean distance matrix of

\[
(0,0),(1,0),(0,1),(1,1).
\]

Its double-centered squared-distance Gram matrix is positive semidefinite of
rank two. One-dimensional stress is positive; two-dimensional stress is zero.

### 6.5 In what sense is it minimal?

- Four points are the smallest repertoire containing a complete square cell.
- A one-dimensional Hilbert space could realize the same language only by
  storing the whole square in a classical automaton; it would have no
  nontrivial state orbit.
- A qubit is the smallest quantum system with the projective Pauli
  \(\mathbb Z_2^2\) orbit.
- The four orbit states are affinely independent and saturate the qubit's
  four-dimensional Hermitian operator space.

The example is exact and genuinely low-dimensional, but not scalable: the
qubit's commuting projective translations close after two steps. The qutrit
removes this obstruction.

### 6.6 Important qualification: an irrational qubit phase encoding

There is a mathematically faithful qubit encoding of an unbounded integer
lattice that prevents an unqualified dimension no-go claim. Let

\[
U=\operatorname{diag}(1,e^{i\alpha}),
\qquad
V=\operatorname{diag}(1,e^{i\beta}),
\]

where \(\alpha,\beta,2\pi\) are rationally independent, and take a fiducial
qubit state with nonzero amplitudes in both basis directions, such as
\(\lvert+\rangle\). Then

\[
(x,y)\longmapsto U^xV^y\lvert+\rangle
\]

is injective on \(\mathbb Z^2\): equality up to global phase would imply
\(x\alpha+y\beta\equiv0\pmod{2\pi}\), contradicting rational independence
unless \(x=y=0\). Applying the retry-channel theorem to displacement unitaries
\(U^rV^s\), with success probability \(1/\sqrt{r^2+s^2}\), therefore gives an
exact Euclidean **history-label** metric on any finite lattice patch using only
a qubit.

This construction is exact but physically fragile. All orbit states lie on
one phase circle, and the irrational integer orbit is dense on that circle.
Consequently:

- arbitrarily distant history coordinates produce arbitrarily close qubit
  states;
- the inverse map from physical state to \((x,y)\) is discontinuous;
- no finite-resolution measurement can localize the unbounded coordinate
  robustly;
- target projectors have nonzero acceptance probability on many incorrect
  nonorthogonal states;
- if goal success is defined by a finite-tolerance physical test rather than
  exact history equality, dense near-aliases create shortcuts.

Thus a qubit suffices for an exact finite arithmetic construction, while a
qutrit is minimal for the stronger object wanted here: a smooth, locally
invertible, two-dimensional commuting phase manifold whose two coordinates
remain operationally resolvable at finite scale. This example should be kept
as a simulation control because it separates algebraic faithfulness from
robust emergent dimension.

---

## 7. Scalable construction: an exact square phase lattice in a qutrit

### 7.1 A fiducial state and orthogonal phase generators

Take

\[
\lvert\psi_0\rangle
=\sqrt{\frac38}\lvert0\rangle
+\frac12\lvert1\rangle
+\sqrt{\frac38}\lvert2\rangle.
\]

Define two commuting diagonal generators

\[
A=\operatorname{diag}(0,1,0),
\qquad
B=\operatorname{diag}\!\left(0,\frac12,1\right).
\]

For the two-parameter orbit

\[
\lvert\psi(\alpha,\beta)\rangle
=e^{i(\alpha A+\beta B)}\lvert\psi_0\rangle,
\]

the Fubini--Study line element is the covariance metric of the generators:

\[
ds^2
=\operatorname{Var}(A)d\alpha^2
+\operatorname{Var}(B)d\beta^2
+2\operatorname{Cov}(A,B)d\alpha d\beta.
\]

In \(\lvert\psi_0\rangle\), direct calculation gives

\[
\operatorname{Var}(A)=\operatorname{Var}(B)=\frac3{16},
\qquad
\operatorname{Cov}(A,B)=0.
\]

Hence

\[
\boxed{
ds^2=\frac3{16}(d\alpha^2+d\beta^2)
}
\tag{Q}
\]

everywhere on the orbit. The metric is flat, isotropic, and exactly square in
the \((\alpha,\beta)\) coordinates. This choice of amplitudes and generators is
not cosmetic: a generic pair of qutrit phase coordinates has a cross term and
produces an oblique lattice. Here the cross term has been canceled exactly.

### 7.2 A finite square torus of nonorthogonal states

Choose any odd integer \(m\ge5\) and set

\[
\epsilon=\frac{4\pi}{m},
\qquad
U=e^{i\epsilon A},
\qquad
V=e^{i\epsilon B}.
\]

Then

\[
U^m=V^m=I,
\qquad UV=VU.
\]

For odd \(m\), the projective representation

\[
(x,y)\longmapsto U^xV^y,
\qquad (x,y)\in\mathbb Z_m^2,
\]

is faithful. To see this, identity of the relative phases requires

\[
2y\equiv0\pmod m,
\qquad
2x+y\equiv0\pmod m.
\]

Because 2 is invertible modulo odd \(m\), this implies \(x=y=0\). The qutrit
therefore has \(m^2\) distinct orbit states

\[
\rho_{xy}=U^xV^y\rho_0V^{-y}U^{-x}.
\]

They are not mutually orthogonal; indeed \(m^2>3\) makes that impossible. The
number of place-like goals can nevertheless be arbitrarily large while the
physical Hilbert dimension remains three.

Normalize distance so that adjacent phase steps have length one. For a residue
\(r\in\mathbb Z_m\), let \(\bar r\) be its centered representative in
\(\{-(m-1)/2,\ldots,(m-1)/2\}\). Define the desired left-invariant
square-torus **control metric**

\[
d_m\big((x,y),(x',y')\big)
=\sqrt{\bar r^2+\bar s^2},
\quad
r=x'-x,\quad s=y'-y.
\tag{T}
\]

The common physical scale \(\epsilon\sqrt3/4\) has been divided out. On a
small chart, (T) agrees with the path length induced by (Q). Globally, the
continuous qutrit phase torus has a skew identification lattice in these
coordinates, so its shortest ambient Fubini--Study paths need not equal (T)
for widely separated points. The exact global equality below is between (T)
and optimal control cost, because the retry probabilities implement (T).

### 7.3 Random-unitary displacement instruments

For each nonzero centered displacement \((r,s)\), define

\[
W_{rs}=U^rV^s,
\qquad
L_{rs}=\sqrt{r^2+s^2},
\qquad
p_{rs}=\frac1{L_{rs}},
\]

and the two Kraus operators

\[
K^{(r,s)}_{\rm s}=\sqrt{p_{rs}}\,W_{rs},
\qquad
K^{(r,s)}_{\rm f}=\sqrt{1-p_{rs}}\,I_3.
\tag{K}
\]

For axial unit displacements, \(p=1\). For a unit diagonal,
\(p=1/\sqrt2\). For displacement \((2,1)\), \(p=1/\sqrt5\). The action outcome reveals whether
the displacement occurred. Theorem 3 then gives

\[
\boxed{
D\big((x,y),(x',y')\big)
=d_m\big((x,y),(x',y')\big)
}
\]

exactly. This is an analytically solved reinforcement-learning environment:
the optimal value function, policy, success probability, and geometry are all
known before simulation.

### 7.4 The exact \(3\times3\) open-lattice patch

Take, for example, (m=11), and choose the nine goals

\[
\mathcal G_{3\times3}=\{(x,y):x,y\in\{0,1,2\}\}.
\]

All pairwise coordinate differences have magnitude at most two, far below the
wrap distance (m/2). Therefore restriction of (T) to these goals is exactly

\[
D\big((x,y),(x',y')\big)
=\sqrt{(x-x')^2+(y-y')^2}.
\]

This is the ordinary Euclidean distance matrix of a \(3\times3\) square
lattice, with zero 2D embedding stress. Yet every physical goal state lives in
the same three-dimensional Hilbert space, and the nine states are
nonorthogonal phase transforms of one fiducial superposition.

There is no physical wall around the selected patch. It is a local chart of a
larger periodic space. This is an advantage for the present purpose: the open
grid geometry is obtained without a privileged nine-dimensional position
basis or boundary-dependent transition rule.

### 7.5 Why the standard qutrit Weyl orbit is not already the answer

The qutrit shift and clock operators

\[
X\lvert j\rangle=\lvert j+1\bmod3\rangle,
\qquad
Z\lvert j\rangle=\omega^j\lvert j\rangle,
\qquad \omega=e^{2\pi i/3},
\]

satisfy \(ZX=\omega XZ\). Their conjugation channels therefore commute and
give a projective \(\mathbb Z_3^2\) action. A generic fiducial has a nine-state
orbit, so this elegantly defeats the state-counting objection.

It does not give an open planar \(3\times3\) grid. Its natural topology is a
torus: residues \(+1\) and \(-1\) are both one step away. With the natural
axial/diagonal torus metric, the nine-point centered Gram matrix has Euclidean
rank greater than two; zero-stress planar embedding is impossible. One must
study it as an intrinsically 2D torus, select a patch of a larger group, or
break homogeneity at a boundary. It is a valuable toroidal control, not the
desired planar answer by itself.

### 7.6 Goal semantics and memory accounting

The cleanest exact goal monitor keeps the accumulated successful displacement

\[
q_t=(x_t,y_t)\in\mathbb Z_m^2.
\]

Goal (g_{xy}) accepts histories whose successful displacement sum is
((x,y)) modulo (m). It is a sequence goal with an (m^2)-state DFA. The
physical state transforms in lockstep:

\[
q_t=(x,y)
\Longrightarrow
\rho_t=\rho_{xy},
\]

provided initialization is known and no hidden disturbances occur.

This exact synchronization is both a strength and a limitation. It proves
existence without storing place in orthogonal basis states, but a controller
can navigate by dead reckoning from its own action outcomes. The decisive
next experiments should progressively remove that crutch:

1. hide the initial phase coordinate;
2. insert unobserved displacement noise;
3. provide only weak, nonorthogonal qutrit measurements;
4. require the recurrent agent to infer a posterior over phase coordinates;
5. compare the learned predictive-state rank and memory with the explicit
   (m^2)-state monitor.

Because the orbit states are nonorthogonal, no single measurement can recover
the coordinate perfectly. Localization becomes a genuine Bayesian quantum
filtering problem while the exact control geometry remains available as ground
truth.

### 7.7 Hybrid provenance control: one quantum axis, one history axis

A transparent intermediate model stores horizontal position in the three
orthogonal qutrit basis states and vertical position in a three-state DFA:

\[
(\rho,q_y)=(\lvert x\rangle\!\langle x\rvert,y),
\qquad x,y\in\{0,1,2\}.
\]

East/west instruments update the qutrit coordinate; north/south symbols update
only the DFA coordinate. Retry displacement instruments can impose exact
Euclidean all-pairs costs. The nine augmented states then form an exact grid,
but only one axis is physically readable from the qutrit. This is not a final
emergence model. It is a provenance calibration: the analysis ought to detect
that one dimension is quantum-supported and the other is purely historical.

---

## 8. What is quantum here, and what is supplied by design?

The constructions establish a possibility result, not yet an emergence result
from generic dynamics.

### Supplied explicitly

- a commuting group action whose parameters already have two components;
- success probabilities chosen to encode Euclidean displacement lengths;
- a sequence-goal monitor that recognizes cumulative displacement;
- in the finite exact version, one action for every displacement class.

### Derived rather than supplied

- the qutrit fiducial state and generator choice yield an exactly orthogonal,
  equal-scale Fubini--Study phase metric;
- arbitrarily many distinct place states arise in fixed Hilbert dimension;
- Bellman optimality turns retry probabilities into the exact geodesic metric;
- a local \(3\times3\) square chart arises inside a homogeneous boundary-free
  phase torus;
- the Euclidean rank-two result follows independently from the cost matrix.

The full displacement action catalog is the analytic benchmark, analogous to
having exact geodesic moves. A more austere local-action model should be tested
next. With only a finite set of nearby directions its finite-step geometry is
generally Finsler or polygonal, but it should converge toward the Riemannian
metric as directional resolution increases. Measuring that convergence is a
natural bridge between the exact model and learned inverse design.

---

## 9. Necessary and sufficient conditions: a proposed hierarchy

There is unlikely to be one useful theorem without specifying what “space”
means. The following hierarchy makes the eventual classification problem
precise.

### Level I: exact Euclidean cost geometry

For a finite operational MDP and goal repertoire, the conditions are exactly:

1. proper reachability and Bellman equations (B);
2. symmetry and the metric axioms;
3. a positive-semidefinite centered Gram matrix of rank at most two or three.

These are necessary and sufficient for goals to have exact point coordinates
in ordinary Euclidean space.

### Level II: spatially consistent dynamics

In addition, there must be coordinates for which action-conditioned increment
laws are local and approximately translation invariant. This prevents an
arbitrary teleportation table from masquerading as space merely because its
costs were tuned to a Euclidean matrix.

### Level III: operationally identifiable place

The coordinate must be inferable, to the required accuracy, from the agent's
available intervention--outcome history. Exact one-shot identification obeys
the orthogonality bound (N\le d); approximate and sequential identification
does not. A useful quantitative condition is a localization-error or posterior-
entropy bound at a fixed sensing budget.

### Level IV: low memory provenance

The spatial coordinate should not reside entirely in a hand-coded goal
automaton. Candidate measures include:

- minimal DFA size for exact goal recognition;
- predictive Hankel rank;
- recurrent-state dimension required at fixed prediction error;
- mutual information (I(z_t;q_t)) between learned state and coordinate;
- the additional predictive loss incurred when action labels are erased but
  quantum outcomes are retained;
- robustness to unknown starts and hidden moves.

### Level V: manifold and bundle structure

For increasing goal repertoires, local charts should agree on overlaps and
estimated metric tensors should converge. Internal variables can then be
tested as fibers over the learned base. In the qutrit model, phase position is
the base candidate; posterior uncertainty, measurement context, or additional
internal quantum degrees of freedom can supply a first fiber.

This hierarchy turns “necessary and sufficient conditions for space” into a
sequence of finite, testable statements rather than one ambiguous criterion.

---

## 10. Simulation-ready specification

The following experiment can be implemented without numerical optimization.

### Environment constants

- Hilbert dimension: \(d=3\).
- Torus order: (m=11).
- Fiducial state:
  \(\lvert\psi_0\rangle=(\sqrt{3/8},1/2,\sqrt{3/8})^\top\).
- Generators:
  (A=\operatorname{diag}(0,1,0)),
  (B=\operatorname{diag}(0,1/2,1)).
- Step angle: \(\epsilon=4\pi/11\).
- Coordinate unitaries: (U=e^{i\epsilon A}), (V=e^{i\epsilon B}).
- Goals: the nine sequence targets ((x,y)\in\{0,1,2\}^2).
- Movement outcomes: `success`, `failure`.
- Unit cost per attempted instrument.

### Exact action catalog

For every centered displacement

\[
(r,s)\in\{-5,\ldots,5\}^2\setminus\{(0,0)\},
\]

use instrument (K). This is 120 actions and is intentionally overcomplete. A
smaller goal-patch catalog needs only the 24 nonzero displacements in
\(\{-2,-1,0,1,2\}^2\), provided excursions outside the patch are not allowed
to introduce shorter wrap routes. The full homogeneous catalog is preferable
for the first theorem check.

### Ground-truth assertions

Automated tests should verify:

1. every Kraus Gram sum equals (I_3);
2. (U) and (V) commute and have order 11;
3. all 121 orbit density matrices are distinct numerically;
4. the empirical success frequency of ((r,s)) is
   (1/\sqrt{r^2+s^2});
5. exact value iteration matches (T);
6. the nine-goal submatrix equals ordinary square-lattice Euclidean distance;
7. its Schoenberg Gram matrix is PSD of rank two;
8. metric MDS has zero stress to numerical precision;
9. one-dimensional MDS has nonzero stress;
10. privileged Procrustes recovery is one up to numerical tolerance.

### Dialectical ablations

Each ablation answers a theoretical objection.

| Question | Ablation | Predicted result |
|---|---|---|
| Is \(d=3\) essential for scaling? | Compare the Pauli square and irrational qubit phase encoding | The qubit gives an exact history metric, but only a dense 1D physical orbit |
| Where is each coordinate stored? | Use the qutrit-basis \(\times\) three-state-DFA hybrid | One axis is physically readable and one is history-supported |
| Does a nine-state orbit imply a plane? | Use the qutrit Weyl \(\mathbb Z_3^2\) orbit | It yields a torus, not an open zero-stress planar grid |
| Is the geometry only in the cost schedule? | Set every success probability to one | Goal metric becomes the discrete action word metric, not Euclidean |
| Is the full action catalog doing the work? | Keep only axial and diagonal local moves | Zero stress is lost for offsets such as \((2,1)\) |
| Is the agent merely dead reckoning? | Randomize the initial coordinate secretly | Sequence tracking alone fails; sensing becomes necessary |
| Do quantum outcomes carry position information? | Compare informative qutrit POVM with a null instrument | Only the informative condition should relocalize after hidden moves |
| Are goals secretly basis states? | Compute pairwise fidelities and discrimination error | The nine targets remain nonorthogonal despite exact spatial cost |
| Is the geometry local? | Evaluate action increments across all torus sites | Displacement laws are exactly translation invariant |
| Is two-dimensionality global or local? | Embed the whole torus versus the \(3\times3\) patch | The patch is planar; the periodic torus is intrinsically 2D but not globally planar |

### Learning experiment

Run four increasingly difficult stages:

1. **Known model, known source.** Confirm exact Bellman and MDS results.
2. **Learned transition table, known source.** Estimate retry probabilities
   from landmark-anchored surveys and measure sample complexity.
3. **Unknown source, informative sensing.** Add a fixed qutrit informationally
   complete POVM or a family of weak phase measurements. Learn a belief filter.
4. **Hidden slips.** Apply unobserved random displacements. Train a recurrent
   controller and test whether its learned state forms a base coordinate plus
   an uncertainty fiber.

The exact model supplies a clean loss decomposition:

\[
\text{geometric error}
=\text{model estimation}
+\text{belief error}
+\text{planning error}
+\text{policy execution error},
\]

with each term isolated by an oracle control.

---

## 11. Predictions and falsifiers

The theory makes several sharp predictions.

1. The exact qutrit benchmark will recover the nine-goal planar lattice to
   floating-point precision; failure indicates an implementation error.
2. The physical qutrit states will not have the same pairwise ambient quantum
   distances as the square-lattice hodological distances. Equality is neither
   expected nor required.
3. Restricting the action catalog to a few local directions will introduce a
   systematic polygonal metric, not random embedding noise.
4. A known-start controller will require very little learned state because it
   can integrate observed displacements. Unknown starts and hidden slips will
   sharply increase the value of sensing and recurrent memory.
5. A qubit can reproduce the four-goal square and inject \(\mathbb Z^2\) into
   an irrational phase circle, but it cannot make that dense encoding a robust
   two-dimensional commuting phase manifold.
6. If a claimed low-dimensional model supports nine deterministic one-shot
   place outcomes on a qutrit, some classical side information or enlarged
   effective Hilbert space must have entered the implementation.

These statements make the project falsifiable: zero-stress geometry alone is
not sufficient evidence of emergent space.

---

## 12. Research outlook

### 12.1 Remove hand-tuned distances

The retry probabilities currently encode the target Riemannian norm. The next
inverse-design question is whether a small local instrument repertoire can
*learn* or dynamically induce the same metric. Candidate objectives include
the rank-two Schoenberg residual, local isotropy, Bellman calibration, and
penalties for action-catalog size.

### 12.2 Derive the metric from control energy

For continuous controls

\[
U(t)=e^{i(\alpha(t)A+\beta(t)B)},
\]

the Fubini--Study speed follows (Q). If action cost is integrated speed, the
minimum control cost is the intrinsic Riemannian geodesic length without
introducing one retry action per displacement. Discretizing this variational
problem would connect reinforcement learning to quantum optimal control and
provide a less engineered route to Euclidean geometry.

### 12.3 Learn the phase chart from outcomes only

Replace explicit coordinate tracking with weak qutrit measurements whose
statistics vary smoothly over the torus. The agent should learn a circular
two-coordinate predictive representation. Topological diagnostics should
detect periodicity before a local chart is unfolded. Persistent homology,
diffusion maps, and successor-feature eigenfunctions are natural comparisons,
but all should be judged against the exact metric and group action.

### 12.4 From base to fiber

Add an internal degree of freedom or measurement context at each phase point.
If transport around a loop changes the internal state, the learned transition
model defines a discrete connection. Loop holonomy is then an operational
curvature observable. The qutrit phase lattice is attractive because its flat
base is exactly solved: any measured holonomy can be attributed to the added
fiber coupling rather than uncertainty about the base geometry.

### 12.5 Toward necessary-and-sufficient structural theorems

The project should seek classification results under explicit assumptions,
for example:

- finite physical dimension (d);
- bounded goal-monitor memory (M);
- reversible or general CPTP movement;
- exact or approximate localization;
- homogeneous local actions;
- fixed action-catalog cardinality;
- metric, risk-sensitive, or information-augmented cost.

Without these qualifiers, automaton universality makes the answer trivial.
With them, the Bellman, Schoenberg, representation-theoretic, and
discrimination constraints above provide the beginning of a genuine theory.

---

## 13. References and mathematical provenance

- I. J. Schoenberg, “Remarks to Maurice Fréchet's article ‘Sur la définition
  axiomatique d'une classe d'espace distanciés vectoriellement applicable sur
  l'espace de Hilbert,’” *Annals of Mathematics* **36**, 724–732 (1935).
  [DOI](https://doi.org/10.2307/1968654). The centered-Gram criterion is the
  finite Euclidean-distance theorem used in Section 4.
- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
  Information*, 10th anniversary ed. (Cambridge University Press, 2010).
  [Publisher page](https://doi.org/10.1017/CBO9780511976667). Background on
  instruments, state discrimination, and unitary dynamics.
- I. Bengtsson and K. Życzkowski, *Geometry of Quantum States*, 2nd ed.
  (Cambridge University Press, 2017).
  [Publisher page](https://doi.org/10.1017/9781139207010). Background on the
  Fubini--Study metric, quantum-state geometry, and phase orbits.
- M. L. Puterman, *Markov Decision Processes* (Wiley, 1994).
  [Publisher page](https://doi.org/10.1002/9780470316887). Bellman optimality
  and stochastic shortest-path control.
- A. S. Holevo, *Probabilistic and Statistical Aspects of Quantum Theory*, 2nd
  English ed. (Edizioni della Normale, 2011).
  [Publisher page](https://doi.org/10.1007/978-88-7642-378-9). Statistical
  distinguishability and quantum measurements.

The qubit Pauli square and the orthogonal qutrit phase-generator choice in
Section 7 are derived explicitly here and should be treated as testable model
proposals rather than cited standard constructions.
