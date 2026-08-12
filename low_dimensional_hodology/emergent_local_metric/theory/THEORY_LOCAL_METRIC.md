# Emergent Euclidean geometry from local quantum controls

## What remains after displacement-specific retry actions are removed?

**Status.** This note gives the theoretical basis for the next low-dimensional
hodology experiment. The previous exact qubit and qutrit constructions provide
one retry instrument for every desired source--goal displacement. They prove
that a two- or three-dimensional Hilbert space can carry an arbitrarily large
goal orbit, but the Euclidean norm is then explicitly encoded in the success
probability of a goal-sized macro-action.

Here the action catalog is fixed, small, local, goal-independent, and
translation-covariant. Long displacements must be composed from the same
primitive interventions. The main result is partly an obstruction:

> A fixed finite set of local translation directions with additive
> intervention cost has a polygonal stable norm. It is generically Finsler,
> not Euclidean. Refining the spatial lattice does not remove this anisotropy
> if the direction set remains fixed.

This is not a negative result for the project. It identifies the additional
ingredient that ordinary Euclidean space requires. There are several
possibilities: increasing directional resolution, a continuous two-generator
control field with an isotropic resource cost, a diffusion-derived metric
rather than mean hitting time, or learned temporally extended actions whose
direction set grows with scale.

The concrete simulation proposed below uses a qutrit phase orbit and three
nested action families with 4, 8, and 16 directions. Every family has a closed
form optimal metric. The 16-direction family is exactly Euclidean on a
\(3\times3\) goal patch but ceases to be exact on larger patches. This gives a
particularly clean dialectical experiment: a finite result can look like
emergent Euclidean space while its scaling law reveals a persistent polygonal
geometry.

---

## 1. Operational setting

### 1.1 Translation orbit

Let the operational coordinate group be

\[
\Gamma=\mathbb Z_m^2
\]

for a large odd \(m\), or \(\mathbb Z^2\) when wraparound is irrelevant.
Assume a projective unitary representation

\[
x\longmapsto W_x\in PU(d),
\qquad
W_xW_y\simeq W_{x+y},
\]

and a fiducial state \(\rho_0\) whose relevant orbit labels are distinct:

\[
\rho_x=W_x\rho_0W_x^\dagger.
\]

The construction in Section 8 uses \(d=3\). The theory in Sections 2--7
depends only on the translation law, not on the Hilbert dimension.

### 1.2 A finite local action catalog

Let

\[
S\subset\mathbb Z^2\setminus\{0\}
\]

be a finite symmetric set of primitive displacement vectors:
\(v\in S\Rightarrow -v\in S\). An attempted action \(v\) has unit
intervention cost and two Kraus outcomes

\[
K^{(v)}_{\mathrm s}=\sqrt{p_v}\,W_v,
\qquad
K^{(v)}_{\mathrm f}=\sqrt{1-p_v}\,I,
\qquad 0<p_v\le1.
\tag{1}
\]

On success the coordinate changes \(x\mapsto x+v\); on failure it remains
unchanged. Both outcomes are observed. The action is translation-covariant:

\[
\mathcal E_o^{(v)}(\rho_x)
=W_x\,\mathcal E_o^{(v)}(\rho_0)\,W_x^\dagger
\]

up to the harmless ordering convention for commuting translations. No action
is indexed by a goal or by the current site.

More generally, an attempt may have external cost \(c_v>0\). The central
quantity is then the effective successful-displacement cost

\[
\ell_v=\frac{c_v}{p_v}.
\tag{2}
\]

The proposed experiment uses \(c_v=1\) and
\(p_v=1/\lVert v\rVert_2\), so \(\ell_v=\lVert v\rVert_2\) for the few
local directions that are actually provided. This still calibrates primitive
direction lengths, but it no longer provides a direct action for every
source--goal displacement.

### 1.3 Goal semantics

For a known source, the observed successes define an accumulated displacement.
A sequence goal \(g\) accepts when this total first reaches \(g\). On a finite
torus this monitor is a finite automaton with \(m^2\) coordinate states. On an
unbounded lattice it is a counter automaton, or a finite automaton after an
episode horizon is imposed.

As in the parent low-dimensional study, physical Hilbert dimension and
goal-monitor memory must be reported separately. The present question is not
whether history can name a lattice; it is what metric local physical controls
induce on that lattice.

---

## 2. Retry instruments reduce exactly to a weighted word metric

### Proposition 1 (effective edge length)

For the retry instrument (1), the expected cost of obtaining one successful
displacement \(v\), if the action is repeated after failure, is

\[
\ell_v=\frac{c_v}{p_v}.
\]

#### Proof

The number of attempts is geometric with success parameter \(p_v\), hence has
mean \(1/p_v\). Each attempt costs \(c_v\). \(\square\)

### Proposition 2 (optimal local hodology)

Assume failures reveal no new hidden state information and leave the physical
and operational state unchanged. The optimal expected cost from \(x\) to \(g\)
is the weighted word metric

\[
D_S(x,g)
=
\inf\left\{
\sum_{k=1}^n\ell_{v_k}:
v_k\in S,\ 
\sum_{k=1}^n v_k=g-x
\right\}.
\tag{3}
\]

On \(\mathbb Z_m^2\), the displacement equality is interpreted modulo \(m\).

#### Proof

Any prescribed successful word \(v_1,\ldots,v_n\) can be implemented by
retrying each action until success. Proposition 1 gives expected total cost
\(\sum_k\ell_{v_k}\), so the right side of (3) is achievable.

For the converse, let \(V(x)\) be the right side. The triangle inequality for
the word metric gives

\[
V(x)-V(x+v)\le\ell_v=\frac{c_v}{p_v}.
\]

The Bellman expression for attempting \(v\) and behaving optimally afterward
is

\[
c_v+(1-p_v)V(x)+p_vV(x+v)\ge V(x).
\]

Thus no first action, and hence no adaptive policy, has expected cost below
\(V(x)\). Properness and positivity make \(V\) the minimal nonnegative
Bellman solution. \(\square\)

### Consequences

1. Retry noise does not by itself round a lattice metric. It simply converts
   success probability into effective edge length.
2. Randomizing among actions does not enlarge the asymptotic velocity set
   beyond its convex hull.
3. Once \((S,\ell)\) is specified, exact values can be computed by Dijkstra's
   algorithm without simulating density matrices.
4. The quantum implementation remains important for operational
   observability, localization, disturbance, and the provenance of the
   coordinate, but the known-source control metric is a weighted Cayley-graph
   metric.

---

## 3. When can a local catalog be exactly Euclidean?

Suppose every primitive is priced at its Euclidean length:

\[
\ell_v=\lVert v\rVert_2.
\tag{4}
\]

The Euclidean triangle inequality immediately gives

\[
D_S(0,z)\ge\lVert z\rVert_2.
\tag{5}
\]

### Proposition 3 (exactness on a displacement)

Under (4), equality in (5) holds for a nonzero displacement \(z\) if and only
if \(z\) can be written as a sum of available action vectors that all lie on
the same oriented ray as \(z\).

#### Proof

For any word summing to \(z\),

\[
\sum_k\ell_{v_k}
=\sum_k\lVert v_k\rVert_2
\ge
\left\lVert\sum_kv_k\right\rVert_2
=\lVert z\rVert_2.
\]

Euclidean norm is strictly convex. Equality in its triangle inequality holds
exactly when all nonzero \(v_k\) are nonnegative scalar multiples of one
another. Therefore an exact word exists precisely under the stated ray
condition. \(\square\)

### Corollary 3.1 (finite-direction obstruction)

A fixed finite action catalog cannot reproduce Euclidean distance exactly on
all of \(\mathbb Z^2\). It is exact only on the finite set of rays represented
by its primitive directions.

This statement is stronger than saying that a particular diagonal is absent.
Even if the catalog is tuned with arbitrary positive lengths, global exactness
forces every primitive not to undercut its own Euclidean displacement and
forces every target direction to satisfy the equality condition above. The
integer lattice contains infinitely many primitive slopes.

### Corollary 3.2 (catalog size required for an exact finite patch)

Consider a square goal patch with coordinate differences
\(|\Delta x|,|\Delta y|\le R\). Exact Euclidean all-pairs distance requires at
least one action ray for every coprime pair

\[
(a,b),\qquad
0\le a,b\le R,\qquad
\gcd(a,b)=1,
\]

together with reflections and reversals. The number of required directions is
quadratic in \(R\), because the density of coprime integer pairs is
\(6/\pi^2\).

Long actions along an already represented ray are unnecessary: repeated
primitive moves preserve equality. The obstruction is directional, not
radial. Nevertheless, exactness on ever larger patches requires an unbounded
direction catalog.

---

## 4. The stable norm is polygonal

The finite-patch ray argument has an asymptotic counterpart that identifies
the continuum geometry.

### 4.1 Convex velocity body

For a symmetric spanning action family define

\[
P_S
=
\operatorname{conv}
\left\{
\frac{v}{\ell_v}:v\in S
\right\}
\subset\mathbb R^2.
\tag{6}
\]

The associated Minkowski functional or gauge is

\[
\lVert x\rVert_{P_S}
=
\inf\{\lambda>0:x\in\lambda P_S\}.
\tag{7}
\]

Equivalently,

\[
\lVert x\rVert_{P_S}
=
\inf_{\alpha_v\ge0}
\left\{
\sum_v\alpha_v\ell_v:
\sum_v\alpha_vv=x
\right\}.
\tag{8}
\]

Because \(S\) is symmetric, (7) is a norm. Without symmetry it is a directed
Finsler gauge.

### Theorem 4 (stable local-control norm)

If \(S\) generates \(\mathbb Z^2\), then for every
\(z\in\mathbb Z^2\),

\[
\lim_{n\to\infty}\frac{D_S(0,nz)}{n}
=
\lVert z\rVert_{P_S}.
\tag{9}
\]

More generally, integer points approaching a fixed real direction converge
after rescaling to the same gauge.

#### Proof sketch

Every integer word is feasible in the linear program (8), so
\(D_S(0,nz)\ge n\lVert z\rVert_{P_S}\). Conversely, take an optimal fractional
decomposition of \(z\). Multiplying its coefficients by \(n\), rounding all
but a fixed generating subset to integers, and correcting the bounded
rounding residue with a bounded word gives

\[
D_S(0,nz)
\le n\lVert z\rVert_{P_S}+C_z,
\]

where \(C_z\) is independent of \(n\). Divide by \(n\) and take the limit.
\(\square\)

### Theorem 5 (finite actions imply a Finsler polygon)

For finite \(S\), \(P_S\) is a polygon. Consequently its gauge cannot equal the
Euclidean norm in all directions.

#### Proof

A convex hull of finitely many points is a polytope, hence in two dimensions a
polygon. The Euclidean unit ball is a disk with infinitely many extreme
points and a continuously curved boundary. A finite polygon cannot equal that
disk. By (7), equality of norms would require equality of unit balls.
\(\square\)

The obstruction persists as the lattice spacing tends to zero if \(S\) is
held fixed. Spatial refinement changes the sampling of the polygonal norm; it
does not turn the polygon into a circle. The continuum limit is a flat
Finsler plane.

### 4.2 Quantitative angular bound

Assume (4), so all normalized velocities \(v/\ell_v\) lie on the Euclidean
unit circle. Sort their angles, including reversals, and let
\(\Delta_{\max}\) be the largest angular gap between adjacent directions.
The polygon is inscribed in the unit circle. In the middle of the largest
gap its radial extent is

\[
\cos(\Delta_{\max}/2).
\]

Therefore

\[
1
\le
\frac{\lVert x\rVert_{P_S}}{\lVert x\rVert_2}
\le
\sec(\Delta_{\max}/2),
\tag{10}
\]

and the worst relative anisotropy is exactly

\[
\varepsilon_\infty
=\sec(\Delta_{\max}/2)-1.
\tag{11}
\]

For \(K\) evenly spaced oriented directions,
\(\Delta_{\max}=2\pi/K\), hence

\[
\varepsilon_\infty
=\sec(\pi/K)-1
=\frac{\pi^2}{2K^2}+O(K^{-4}).
\tag{12}
\]

Directional refinement can therefore converge to Euclidean geometry, but a
fixed catalog cannot.

---

## 5. Three exactly soluble local families

### 5.1 Four axial directions

Let

\[
S_4=\{\pm(1,0),\pm(0,1)\},
\qquad \ell_v=1.
\]

Then

\[
D_4(0,(a,b))=|a|+|b|,
\]

the Manhattan norm. Its unit ball is a diamond and its maximum distortion
relative to Euclidean distance is

\[
\sqrt2-1\approx0.4142.
\]

### 5.2 Eight axial and diagonal directions

Add

\[
\pm(1,1),\qquad \pm(1,-1),
\]

with effective cost \(\sqrt2\), implemented by
\(p=1/\sqrt2\). If

\[
A=\max(|a|,|b|),\qquad B=\min(|a|,|b|),
\]

then the exact word metric is the octile norm

\[
\boxed{
D_8(0,(a,b))
=A+(\sqrt2-1)B.
}
\tag{13}
\]

Indeed, use \(B\) diagonal successes and \(A-B\) axial successes. Any other
word is no cheaper by the supporting facets of the regular-octagonal unit
ball.

The largest angular gap is \(\pi/4\), so

\[
\varepsilon_\infty^{(8)}
=\sec(\pi/8)-1
\approx0.0823922.
\]

On a \(3\times3\) patch the only inequivalent offending displacement is
\((2,1)\):

\[
D_8(0,(2,1))=1+\sqrt2,
\qquad
\lVert(2,1)\rVert_2=\sqrt5,
\]

with relative excess

\[
\frac{1+\sqrt2}{\sqrt5}-1
\approx0.0796691.
\tag{14}
\]

Thus the eight-action model should look convincingly two-dimensional while
still failing a precise Euclidean test.

### 5.3 Sixteen directions with knight moves

Add all sign changes and coordinate swaps of \((2,1)\):

\[
S_{16}
=S_8
\cup
\{(\pm2,\pm1),(\pm1,\pm2)\},
\]

with effective cost \(\sqrt5\), implemented by \(p=1/\sqrt5\).

Every primitive direction occurring among differences of a \(3\times3\)
patch is now present. Proposition 3 implies

\[
D_{16}(x,g)=\lVert g-x\rVert_2
\]

for all 81 ordered source--goal pairs in that patch. Two-dimensional MDS must
have zero stress to numerical precision.

This exactness does not scale. On a \(5\times5\) patch, directions such as
\((3,1)\) and \((3,2)\) are absent. The largest angular gap is
\(\arctan(1/2)\), giving asymptotic worst distortion

\[
\varepsilon_\infty^{(16)}
=
\sec\!\left(\frac12\arctan\frac12\right)-1
\approx0.0274863.
\tag{15}
\]

The 16-direction model is therefore an important warning: exact recovery on
one small benchmark does not establish an asymptotically Euclidean space.

---

## 6. Why ordinary stochasticity does not automatically cure anisotropy

### 6.1 Policy randomization only convexifies

A randomized stationary policy chooses convex mixtures of primitive action
velocities. Those mixtures fill \(P_S\), which is already the convex hull in
(6). They do not add a curved boundary. Randomization converts a nonconvex
catalog into its polygonal relaxation, not into a disk.

### 6.2 General noisy increments

Suppose an action has several displacement outcomes rather than “move or
stay.” At large scales, controlled first-passage values satisfy an effective
Hamilton--Jacobi equation whose Hamiltonian is determined by the available
increment laws. With finitely many controls and additive time cost, the
leading ballistic geometry is generically a Finsler control metric. Noise
adds a second-order diffusion term and finite-scale corrections; it does not
force isotropy.

This broader statement needs assumptions for a full homogenization theorem,
so the proposed experiment initially uses retry instruments, where
Propositions 1--5 are exact.

### 6.3 Uncontrolled diffusion changes the notion of distance

A symmetric random walk has an isotropic covariance tensor when its direction
set is balanced. This can produce a Riemannian metric through the heat kernel:
for a diffusion with metric \(g\), Varadhan's short-time relation has the form

\[
d_g(x,y)^2
=
\lim_{t\downarrow0}
-4t\log p_t(x,y).
\tag{16}
\]

But this is not the same observable as expected unit-cost hitting time.
In two infinite dimensions, an unbiased random walk reaches a point almost
surely but has infinite mean hitting time. On a finite torus, mean hitting
time depends strongly on system volume and recurrence. Taking a square root
of a mean time can sometimes reveal a length scale, but it changes the
hodological definition and must be justified rather than silently substituted.

Diffusion geometry is nevertheless a promising alternative: ordinary space
might emerge from the agent's transition statistics or prediction kernel
rather than directly from mean interventions-to-goal.

---

## 7. Mechanisms that can yield a Riemannian metric

The polygonal obstruction identifies which assumption must change.

### 7.1 Increase directional resolution

Let \(S_K\) contain \(K\) approximately evenly spaced directions, normalized
by their physical length. Equation (12) predicts worst-case distortion
\(O(K^{-2})\). If learned options create new directions as experience grows,
the effective unit ball can converge toward a disk.

This mechanism has a measurable complexity cost: for exact Euclidean geometry
on an \(R\)-radius integer patch, direction count grows as \(O(R^2)\). For
fixed approximation tolerance \(\eta\), angular resolution requires only
\(K=O(\eta^{-1/2})\).

### 7.2 Continuous control from two Hamiltonian generators

The qutrit phase model has two commuting generators \(A\) and \(B\). Allow a
continuous local control

\[
H(u)=u_1A+u_2B,
\qquad
\lVert u\rVert_2\le1.
\]

The coordinate dynamics are

\[
\dot x=u.
\]

If cost is elapsed time, the reachable velocity body is the Euclidean disk,
not a finite polygon, and minimum time is exactly

\[
T^\star(x,g)=\lVert g-x\rVert_2.
\]

Only two physical control fields are required, but the action amplitude and
direction are continuous. This is a small *parametric* action family rather
than a finite action catalog.

If componentwise bounds \(|u_1|,|u_2|\le1\) are used instead, the velocity body
is a square and minimum time is the Chebyshev norm. Euclidean geometry comes
from isotropy of the admissible control resource, not merely from having two
coordinates.

### 7.3 Quadratic control energy

For fixed duration \(T\), define

\[
\mathcal E[u]
=\frac12\int_0^T u(t)^\top G\,u(t)\,dt.
\]

Under \(\dot x=u\), Cauchy--Schwarz gives the exact optimum

\[
\mathcal E^\star(x,g;T)
=
\frac1{2T}(g-x)^\top G(g-x),
\tag{17}
\]

attained by constant velocity. The square root of \(2T\mathcal E^\star\) is
the Riemannian distance with metric tensor \(G\).

This is theoretically attractive because the qutrit Fubini--Study metric
itself supplies a generator covariance matrix \(G\). It does, however, replace
unit intervention count with a physical effort functional.

### 7.4 Learned options and multiscale locality

An agent with only axial primitives can learn temporally extended options that
approximate straight motion in many rational directions. If option costs are
their expected primitive durations, the expanded catalog has the same
polyhedral theory, but its angular gaps shrink over learning. The scientific
question becomes whether useful directional options arise from prediction and
goal reuse without explicitly optimizing MDS stress.

### 7.5 Position-dependent controls

Allowing the local velocity body \(P_x\) to vary with operational position
produces a Finsler manifold; if each \(P_x\) approaches an ellipsoid, it
produces a Riemannian metric tensor \(g(x)\). This is the natural route toward
curvature. Translation covariance should be retained for the flat benchmark
and relaxed only after its errors are understood.

---

## 8. Concrete qutrit simulation

### 8.1 Physical phase orbit

Use

\[
\lvert\psi_0\rangle
=
\sqrt{\frac38}\lvert0\rangle
+\frac12\lvert1\rangle
+\sqrt{\frac38}\lvert2\rangle,
\]

with commuting generators

\[
A=\operatorname{diag}(0,1,0),
\qquad
B=\operatorname{diag}\!\left(0,\frac12,1\right).
\]

Their covariance in the fiducial state is

\[
\operatorname{Var}(A)
=\operatorname{Var}(B)
=\frac3{16},
\qquad
\operatorname{Cov}(A,B)=0.
\tag{18}
\]

Thus the Fubini--Study line element on the phase orbit is locally

\[
ds^2=\frac3{16}(d\alpha^2+d\beta^2).
\]

Choose

\[
m=101,
\qquad
\epsilon=\frac{4\pi}{m},
\qquad
U=e^{i\epsilon A},
\qquad
V=e^{i\epsilon B}.
\]

For odd \(m\), the map

\[
(x,y)\longmapsto U^xV^y
\]

is faithful on \(\mathbb Z_m^2\). The large order keeps all evaluation patches
far from wraparound while providing an exact finite state space.

### 8.2 Translation-covariant Kraus actions

For each \(v=(r,s)\) in a selected family \(S_4,S_8,\) or \(S_{16}\), use

\[
W_v=U^rV^s,
\qquad
p_v=\frac1{\sqrt{r^2+s^2}},
\]

and instrument (1). The largest physical displacement in \(S_{16}\) is
\(\sqrt5\) coordinate units. The catalog is fixed for every source, goal, and
patch size.

Compare four conditions:

1. **axial-4:** \(S_4\);
2. **octile-8:** \(S_8\);
3. **knight-16:** \(S_{16}\);
4. **direct-macro oracle:** one action for every target displacement, retained
   only as the previous exact upper benchmark.

The first three are the scientific conditions. The fourth must not be included
when reporting action locality or catalog complexity.

### 8.3 Goal patches and scaling

Use centered square patches of side

\[
L\in\{3,5,9,17\},
\qquad
\mathcal G_L
=
\left\{
-(L-1)/2,\ldots,(L-1)/2
\right\}^2.
\]

All are far from the \(m=101\) wrap scale. Evaluate all ordered source--goal
pairs. Compute exact values by Dijkstra on \(\mathbb Z_{101}^2\), and verify
the \(S_4\) and \(S_8\) closed forms.

For Monte Carlo policy validation, sample at least 2,000 episodes per
displacement class and seed, using at least five independent seeds. The exact
solver, not Monte Carlo, is the primary geometry source.

### 8.4 Optional learning stages

1. **Known-model planner.** Tests implementation against the proofs.
2. **Surveyed model.** Estimate \(p_v\) from action outcomes, then plan with
   learned \(\hat\ell_v=1/\hat p_v\).
3. **Model-free goal-conditioned learner.** Learn values from sequence-goal
   reward without coordinates.
4. **Unknown source and hidden slips.** Add weak qutrit sensing and recurrent
   belief state only after the local metric baseline is validated.

Keeping these stages separate prevents partial observability from obscuring
the finite-direction effect.

---

## 9. Acceptance metrics

No single MDS plot should decide whether geometry is spatial. Report the
following metric families.

### 9.1 Quantum validity

- maximum Kraus completeness residual
  \(\lVert\sum_oK_o^\dagger K_o-I\rVert_F\);
- maximum density-matrix trace, Hermiticity, and positivity error;
- empirical versus analytic primitive success probability;
- minimum and distribution of pairwise orbit-state trace distances or
  fidelities, establishing that goals are nonorthogonal rather than basis
  labels.

**Acceptance:** algebraic residuals below \(10^{-12}\); Monte Carlo success
frequencies inside simultaneous 95% binomial confidence intervals.

### 9.2 Translation covariance and locality

For each action compare kernels after coordinate translation:

\[
\varepsilon_{\rm cov}
=
\max_{a,x,x'}
\operatorname{TV}
\left(
P_a(\cdot-x\mid x),
P_a(\cdot-x'\mid x')
\right).
\]

Report maximum successful displacement radius, action count, number of outcome
symbols, and whether any action name or parameter contains the current goal.

**Acceptance:** \(\varepsilon_{\rm cov}<10^{-12}\) analytically and
numerically; radii \(1,\sqrt2,\sqrt5\) for \(S_4,S_8,S_{16}\); catalog sizes
4, 8, and 16 independent of \(L\).

### 9.3 Control optimality

- maximum Bellman residual;
- exact-versus-Monte-Carlo cost error;
- learned-model versus oracle cost error;
- success and intervention distributions, not means alone.

**Acceptance:** Bellman residual below \(10^{-10}\); analytic \(D_4,D_8\) and
Dijkstra values agree below \(10^{-10}\); rollout confidence intervals contain
the exact means at their nominal coverage rate.

### 9.4 Metric and Euclidean compatibility

For every condition and patch size report:

- asymmetry;
- triangle-inequality violation;
- Pearson and rank correlation with Euclidean distance;
- maximum, mean, and RMS relative distortion;
- normalized raw MDS stress in dimensions 1, 2, and 3;
- negative eigenvalue mass of the double-centered squared-distance Gram
  matrix;
- rank-two positive-spectrum residual;
- privileged Procrustes recovery after 2D MDS.

The primary obstruction statistic is

\[
\varepsilon_{\max}(L,S)
=
\max_{x\ne g}
\left[
\frac{D_S(x,g)}{\lVert g-x\rVert_2}-1
\right].
\tag{19}
\]

**Required exact checks:**

- \(S_8\) displacement \((2,1)\) has excess
  \(0.0796691275\) within \(10^{-10}\);
- \(S_{16}\) has zero distortion and zero 2D stress on \(L=3\) within
  numerical tolerance;
- \(S_{16}\) has strictly positive distortion for \(L\ge5\);
- large-patch maximum distortions approach the theoretical bounds
  \(0.0823922003\) and \(0.0274862967\) for \(S_8\) and \(S_{16}\).

### 9.5 Angular isotropy

Bin pairwise displacement directions and plot

\[
R_S(\theta)
=
\frac{D_S(0,r_\theta)}{\lVert r_\theta\rVert_2}.
\]

Report its range and Fourier harmonics. \(S_4\) and \(S_8\) should exhibit
fourfold and eightfold facet signatures. The maximum angular gap and
\(\sec(\Delta_{\max}/2)-1\) should predict the observed envelope without fit
parameters.

### 9.6 Complexity-adjusted performance

Report geometry error against:

- number of primitive actions;
- number of distinct directions;
- maximum displacement radius;
- number of learned transition parameters;
- training samples.

An exact \(3\times3\) result from \(S_{16}\) should not be described as better
without noting that it doubles the octile catalog. Conversely, \(S_{16}\)
remains fixed as \(L\) grows, unlike the direct-macro oracle.

---

## 10. Scaling predictions

### Prediction 1: fixed-direction error plateaus

For fixed \(S_8\) or \(S_{16}\),

\[
\lim_{L\to\infty}\varepsilon_{\max}(L,S)
=
\sec(\Delta_{\max}/2)-1>0.
\]

Increasing the number of goal sites or reducing physical phase spacing does
not remove the anisotropy.

### Prediction 2: directional refinement is second order

For approximately uniform \(K\)-direction families,

\[
\varepsilon_{\max}(K)\sim\frac{\pi^2}{2K^2}.
\]

Normalized distance residuals and 2D stress should be \(O(K^{-2})\) in leading
order; their squared numerators are \(O(K^{-4})\).

### Prediction 3: small-patch exactness can be misleading

\(S_{16}\) is exact for \(L=3\), fails for \(L=5\), and converges to a
nonzero Finsler distortion. Dimensional diagnostics should therefore always
cross patch sizes.

### Prediction 4: primitive probability estimation is \(N^{-1/2}\)

For \(N\) independent surveys of action \(v\),

\[
\operatorname{SE}(\hat p_v)
\approx
\sqrt{\frac{p_v(1-p_v)}{N}},
\]

and the delta method gives

\[
\frac{\operatorname{SE}(\hat\ell_v)}{\ell_v}
\approx
\sqrt{\frac{1-p_v}{Np_v}}.
\tag{20}
\]

Learned geometry should approach the exact polygonal metric at
\(N^{-1/2}\), not approach Euclidean geometry beyond the structural floor.

### Prediction 5: localization becomes harder as the phase grid is refined

For a small coordinate displacement \(\delta\), (18) gives

\[
1-\left|
\langle\psi_x\mid\psi_{x+\delta}\rangle
\right|^2
=
\frac3{16}\epsilon^2\lVert\delta\rVert_2^2
+O(\epsilon^4).
\tag{21}
\]

Since \(\epsilon=4\pi/m\), neighboring states become quadratically less
distinguishable as \(m\) grows. The copy or sensing budget required for fixed
localization accuracy should scale at least as \(m^2\) in the regular
estimation regime. Control geometry can remain exact while operational access
to its coordinates deteriorates.

---

## 11. Decision rules for the next research step

The experiment should be interpreted according to the following outcomes.

### If the predicted polygonal metrics are recovered

This validates the local-control theory and shows that the direct retry-macro
construction was not yet an emergence mechanism. The next principled model
should use either continuous bounded controls or an isotropic diffusion
observable.

### If fixed finite directions appear to become Euclidean with scale

First rule out normalization artifacts, boundary effects, MDS visualization
bias, and use of only represented slopes. Check the raw ratio (19) near the
midpoint of the largest angular gap. Under the assumptions of Theorem 5,
true convergence would contradict the convex-hull result.

### If a learned agent outperforms the oracle word metric

The implementation has introduced an additional transition, a state-dependent
shortcut, an unpriced sensing/reset operation, or an incorrect failure update.
The exact Bellman solver is a hard upper benchmark under the specified action
catalog.

### If unknown-start sensing changes the movement metric

Separate movement cost from sensing cost. The physical base geometry can
remain polygonal while total hodology acquires an epistemic fiber. A single
scalar total-cost MDS should not be used to infer deformation of the base
without this decomposition.

---

## 12. Theoretical outlook

The local-action question clarifies a hierarchy of claims:

1. **Orbit capacity:** low-dimensional quantum systems can carry many
   nonorthogonal goal states.
2. **Exact finite geometry:** a sufficiently rich finite action catalog can
   reproduce any finite set of Euclidean displacements.
3. **Local scalable geometry:** a fixed finite catalog produces a Finsler
   polygon, not a Euclidean plane.
4. **Riemannian emergence:** the admissible infinitesimal control or diffusion
   body must become ellipsoidal.
5. **Curved space:** that local ellipsoid may vary smoothly with operational
   position, defining a metric tensor.
6. **Fiber geometry:** internal belief or quantum variables can transform
   under transport over this base, defining connection and holonomy.

The qutrit phase orbit is still valuable because it supplies an exactly flat,
isotropic generator covariance in only three Hilbert dimensions. The open
question is no longer whether it can be assigned a Euclidean goal metric; it
can. The sharper question is:

> What operational resource principle makes the local admissible-control body
> a disk or ellipsoid, rather than a polygon chosen by an arbitrary action
> catalog?

Continuous quantum control with Fubini--Study or quantum-Fisher effort is the
most direct theoretical answer. Learned multiscale options and diffusion
geometry are the most interesting agent-centered alternatives. The proposed
finite-family experiment is the necessary baseline for distinguishing them.

---

## 13. Compact theorem-to-experiment map

| Theoretical statement | Experimental observable | Falsifier |
|---|---|---|
| Retry actions have effective cost \(1/p_v\) | primitive waiting-time mean | mean differs outside sampling uncertainty |
| Local optimum is a weighted word metric | Dijkstra versus Bellman/rollout cost | learned or simulated policy beats Dijkstra |
| \(S_8\) induces octile distance | all-pairs matrix and \((2,1)\) check | deviation from (13) |
| \(S_{16}\) is exact on \(3\times3\) | zero distortion and rank-two Gram matrix | any systematic residual |
| Finite directions give polygonal norm | angular cost profile | smooth disk with error below polygonal bound |
| Fixed-catalog anisotropy persists | \(L=3,5,9,17\) scaling | maximum distortion tends to zero |
| Direction refinement is \(O(K^{-2})\) | error versus angular gap/action count | incompatible convergence after finite-size control |
| Translation covariance is exact | shifted action kernels | nonzero site dependence away from numerical error |
| Phase goals are nonorthogonal | fidelity/trace-distance matrix | targets become basis-like or perfectly one-shot distinguishable |

This table should be copied into the simulation report before results are run,
so acceptance criteria cannot be changed after observing the data.
