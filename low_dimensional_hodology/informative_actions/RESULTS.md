# Informative quantum actions and emergent geometry: synthesis

## Executive conclusion

The informative-actions program has isolated a precise incompatibility rather
than yet producing a final minimal model. Low-dimensional quantum systems can
support each of the following:

1. operationally identifiable opaque action meanings;
2. positive immediate mutual information from genuinely nonunitary movement;
3. exact two-generator translation topology;
4. exact Bellman goal geometry;
5. predictive-state reconstruction without hidden labels.

The completed constructions do not support all five at once.

The exact qubit Pauli benchmark gives nonorthogonal predictive places and an
exact Euclidean square, but movement is unitary and has no immediate reported
information. The searched nonunitary qubit instruments robustly reveal a
two-axis action chart, but the channels fail path independence, so the exact
Manhattan chart lives in action history rather than predictive quantum state.
The integrated qutrit Hesse construction goes further: its outcomes are
informative, its last outcome is a sufficient predictive state, and its opaque
actions close exactly into \(\mathbb Z_3^2\). Yet measurement backaction makes
goal and nearest-neighbor Bellman values equal. Separating qutrit translation
from reporting restores the exact word metric, but returns to uninformative
movement.

The honest current statement is therefore:

> Observable action meaning, translation topology, and goal geometry can each
> emerge in a qubit or qutrit without basis-state goals or privileged action
> coordinates. In the minimal informative constructions studied here,
> however, measurement backaction prevents those properties from composing
> into one nondegenerate spatial predictive quotient.

This is a stronger and more useful result than a visually convincing lattice.
It identifies predictive compositionality and strict Bellman shell ordering as
the two missing constraints for the next model.

## 1. The exact scientific contract

### 1.1 Operational primitives

An action \(a\) is an instrument with observed branches

\[
 \mathcal E_o^a(\rho)
 =\sum_k K_{o,k}^a\rho K_{o,k}^{a\dagger},
 \qquad
 \sum_{o,k}K_{o,k}^{a\dagger}K_{o,k}^a=I.
\]

The learner observes action and outcome tokens, not a simulator state. A
history \(h\) is an action--outcome string. Its meaning is the complete vector
of probabilities of allowed future tests \(\tau\):

\[
 q(h)=\big(p(\tau_1\mid h),\ldots,p(\tau_r\mid h)\big).
\]

Two histories are predictively equivalent precisely when no allowed future
experiment distinguishes them:

\[
 h\sim_P h'
 \quad\Longleftrightarrow\quad
 p(\tau\mid h)=p(\tau\mid h')
 \quad\text{for every }\tau.
\]

For a \(d\)-dimensional Markovian quantum model the controlled Hankel rank is
at most \(d^2\). Reachable-state and observable-effect spans stabilize after
finitely many strict dimension increases. Consequently state equivalence and
action equivalence can be decided using finite spanning sets even though the
number of histories is infinite.

The learned predictive representation is defined only up to similarity gauge,
permutation of opaque tokens, Kraus representation freedom, and automorphisms
of a translation group. Probabilities, predictive classes, composition laws,
Bellman values, spectra, and the Euclidean congruence class of a geometry are
meaningful. Raw latent coordinates and compass names are not.

### 1.2 Six atlas conditions

The theory note proves a finite operational-atlas criterion. After predictive
closure, an exact two-dimensional atlas must satisfy:

#### P — predictive separation

Every pair of proposed places is distinguished by at least one future test:

\[
 i\ne j\Longrightarrow
 \exists\tau:\ p(\tau\mid z_i)\ne p(\tau\mid z_j).
\]

#### A — action identifiability

Shifted Hankel blocks determine the action branches on the reachable
predictive quotient. Actions assigned different meanings must be
operationally distinguishable there, not merely different as source-code
matrices on unavailable states.

#### B — Bellman realizability

The proposed cost to goal \(g\) must be the minimal nonnegative solution of

\[
 V_g(i)=\min_a\left[c(i,a)+\sum_kP_a(i,k)V_g(k)\right]
\]

with the specified terminal semantics. A separately designed distance matrix
does not count.

#### M — metric validity

The baseline-subtracted or terminal-state cost matrix must obey symmetry,
strict positivity off the diagonal, identity of indiscernibles, and the
triangle inequality. In particular, a perfect shell correlation is not enough
if self and edge costs coincide.

#### E — exact planar Euclidean geometry

For distance matrix \(D\), Schoenberg's centered Gram matrix

\[
 B_D=-\frac12JD^{\circ2}J,
 \qquad J=I-\frac1n\mathbf1\mathbf1^T,
\]

must be positive semidefinite of rank two. This is necessary and sufficient
for exact embedding in \(\mathbb R^2\).

#### L — learned local action consistency

After coordinates are reconstructed from costs, action-induced increments
must agree with response-derived, site-independent local displacement laws.
P--E without L could be a teleportation table rather than motion.

### 1.3 Additional requirements for informative spatial actions

The completed simulations show that P/A/B/M/E/L should be augmented by four
audits:

1. **Information provenance:** report separately immediate mutual information,
   future action identity information, and causal displacement. Any one can be
   nonzero while the others vanish.
2. **Predictive path closure:** if two words represent the same proposed
   displacement, they must lead to the same predictive class, at least over
   the intended base after quotienting explicitly declared internal fibers:

   \[
   w\sim_\Delta w'\Longrightarrow w\sim_P w'.
   \]

3. **Held-out prediction:** the learned state and action operators must predict
   longer unseen strings, so a chart cannot be inferred from a short,
   nonclosing signature alone.
4. **Matched nulls:** cloned actions, null observation, random/twirled controls,
   label permutations, and external counters must fail at predetermined
   criteria.

These clauses are the scientific contract for the next experiment.

## 2. Main constructions and exact results

### 2.1 Exact qubit predictive square

The theory strand gives the smallest exact finite spatial benchmark. Begin
with a qubit whose Bloch vector is \((1,1,1)/\sqrt3\). Pauli conjugations
generate four tetrahedral, mutually nonorthogonal states. A tetrahedral SIC
probe gives response law

\[
 p(j\mid i)=
 \begin{cases}
 1/2,&j=i,\\
 1/6,&j\ne i.
 \end{cases}
\]

The four predictive places are therefore exactly observable. Opaque Pauli
buttons are learned as permutations of their response fields. Projectively,
the Pauli actions close as \(\mathbb Z_2^2\).

Two unit-cost axial actions and a diagonal retry action with success
probability \(1/\sqrt2\) give

\[
 D=
 \begin{pmatrix}
 0&1&1&\sqrt2\\
 1&0&\sqrt2&1\\
 1&\sqrt2&0&1\\
 \sqrt2&1&1&0
 \end{pmatrix}.
\]

Its Schoenberg Gram matrix is positive semidefinite of rank two: this is an
exact unit square satisfying P/A/B/M/E/L. It proves that a qubit can carry a
finite exact 2D operational cell without four orthogonal basis states.

Its limitation is equally exact. The moves are deterministic or
random-unitary and provide no informative immediate movement outcome. The
translation group is only \(\mathbb Z_2^2\), so the construction does not
scale to an open plane. An irrational commuting-phase qubit does not solve the
problem: its quantum Fisher matrix has rank one, so two integer labels are
encoded on one physically resolvable phase direction.

### 2.2 Nonunitary qubit action chart

The qubit experiment searches genuinely informative instruments. Each opaque
button first rotates by hidden phase \(\theta_b\), then performs the same weak
\(X\) measurement:

\[
 E_o={I+osX\over2},\qquad
 K_{b,o}=\sqrt{E_o}\,e^{-i\theta_bZ/2}.
\]

The selected deterministic-search solution among 11,799 candidates is

\[
 \alpha=0.64,\qquad \beta=2.02,\qquad s=0.45,
\]

with phases \(\{\pm\alpha,\pm\beta\}\) and coherence retention 0.893029.
Immediate and common-future statistics obey

\[
 P(o=+\mid b)={1+s\cos\theta_b\over2},
\]

\[
 P(X+\mid b)={1+\cos\theta_b\over2},\qquad
 P(Y+\mid b)={1+\eta\sin\theta_b\over2}.
\]

Thus phase effects, inverse pairs, and two axes can be reconstructed from
observed frequencies alone, up to the correct square \(D_4\) gauge. Recovery
is 87.33% with two samples per signature component, 98.67% with five, and 100%
in all 300 replications at ten or more. Predictive tomography after stochastic
histories has mean trace-distance error 0.0350 using 300 samples for each
common Pauli probe.

Signed word counts define nine bounded sequence goals with exact Manhattan
costs. But the actual nonselective channels

\[
 \Phi_\theta=D_X\circ R_Z(\theta)
\]

do not commute. Of 41 displacement classes generated by words of length at
most four, 33 contain multiple paths; none of those 33 are predictively
equivalent at \(10^{-10}\). The worst common-probe signature discrepancy is
0.128490. The sequence lattice is therefore an action-history chart, not an
exact quotient of the predictive qubit states.

This construction passes operational action identification and demonstrates
genuinely nonunitary semantics, but fails predictive closure and hence L for a
state-space atlas. Its exact Manhattan geometry requires an external word
equivalence or counter.

### 2.3 Integrated Hesse-SIC qutrit topology

The qutrit construction uses the nine Hesse SIC projectors \(\Pi_o\) and Weyl
translations \(U_a\in\{I,X,X^\dagger,Z,Z^\dagger\}\). Each opaque action is a
single integrated nine-outcome instrument:

\[
 K_o^{(a)}={1\over\sqrt3}\Pi_oU_a.
\]

If the pre-action predictive state is \(\Pi_s\),

\[
 P_a(o\mid s)=
 \begin{cases}
 1/3,&o=T_a(s),\\
 1/12,&o\ne T_a(s),
 \end{cases}
\]

and the conditional branch state is exactly \(\Pi_o\). Hence the latest
observed outcome token is itself a sufficient predictive state. The immediate
mutual information is exactly

\[
 I(S;O\mid a)=0.2516291674\ \text{bits}.
\]

After independently shuffling action and outcome tokens, 2,000 samples per
source-token/action pair recover all 45 action permutation images. Held-out
kernel MAE ranges from 0.005047 to 0.005790. The learned permutations reveal
one identity, four commuting order-three generators, group order nine, and a
transitive orbit of size nine. This operationally establishes
\(\mathbb Z_3\times\mathbb Z_3\) topology up to origin, axes, signs, and token
gauge. No hidden coordinate is needed by the learner.

This topology is exact, but its integrated *report-again event cost* is
degenerate. For the terminal goal “report token \(g\),” exact Bellman analysis
gives

\[
 V_{\rm self}=4,\qquad V_{\rm edge}=4,
 \qquad V_{\rm diagonal}=5.
\]

Identity of indiscernibles fails after any common baseline subtraction because
self and edge coincide. Weakening the measurement from strength zero to one
never restores a self--edge gap: both values move together from 9 to 4, while
the diagonal moves from 9 to 5. This is an analytic no-go for the symmetric
measure-and-prepare grammar, not a training failure.

This no-go does not apply to the different goal “the current predictive state
is \(g\).” That state-hitting problem terminates before another action, so
\(V_g(g)=0\); the same transition kernel then has exact shells \((0,4,5)\), a
valid torus metric. The correction and its retained-memory successor are
developed in `../covariant_memory_geometry/`.

The Cayley graph is a genuine two-generator \(3\times3\) torus, but its graph
metric itself is not a planar Euclidean distance matrix: 2D MDS stress is
0.383183. A separate axis-instrument predictive distance has perfect
correlation with square-root Hamming distance but centered rank four and 2D
stress 0.408290. Translation topology, static predictive geometry, and
Bellman metric are three distinct objects.

### 2.4 Separated qutrit benchmark

When the four Weyl translations are unit-cost single-Kraus moves and SIC
reporting is a separate unit-cost action, the Bellman equation has exact
solution

\[
 V_g(s)=6+d_T(s,g).
\]

The baseline-subtracted values are exactly the toroidal word distances 0, 1,
and 2. This benchmark passes strict self < edge < diagonal ordering and cleanly
connects learned response permutations to a scalar goal metric.

But the movement actions themselves have no immediate outcome and zero
immediate mutual information. Their meanings are learned through later SIC
probes. The model proves that exact topology plus exact metric is possible when
movement and reporting are separated; it does not solve the target of
integrated informative movement.

## 3. Cross-strand comparison

| construction | Hilbert dimension | action meaning learned from | immediate move MI | predictive composition/topology | Bellman geometry | decisive limitation |
|---|---:|---|---:|---|---|---|
| exact Pauli square | 2 | tetrahedral-SIC response permutations | 0 | exact \(\mathbb Z_2^2\) | exact Euclidean unit square; P/A/B/M/E/L | finite four-place cell; uninformative moves |
| nonunitary phase qubit | 2 | outcomes plus common future \(X/Y/Z\) probes | positive and state dependent | inverse pairs learned, but 0/33 repeated-path classes close | external word counts give exact \(3\times3\) Manhattan chart | chart is history/counter geometry, not predictive-state geometry |
| integrated Hesse qutrit | 3 | immediate SIC token and future controlled kernel | 0.251629 bits | exact opaque \(\mathbb Z_3^2\), group order 9 | self = edge = 4, diagonal = 5; not a metric | informative reporting backaction collapses first shell |
| separated qutrit | 3 | later SIC probe response shifts | 0 | exact opaque \(\mathbb Z_3^2\) | \(V=6+d_T\), exact torus word metric | motion and informative report are separate |
| null plus external counter | 2 or 3 | hand-coded history | 0 | quantum predictive rank 0 | arbitrary exact counter geometry | explicit false positive; fails P/A/L |

No row simultaneously provides informative integrated outcomes, exact
predictive path closure, and a nondegenerate exact metric.

## 4. Controls and what they teach

### 4.1 Random-unitary actions

In the qubit strand, fair-coin random-unitary controls recover the action chart
at 100% with 200 trials because later probes reveal the phase effects. Thus
immediate informativeness is not necessary for action identifiability.

In the qutrit strand, Weyl and Haar-random unitaries are all distinguishable by
future SIC kernels. But distinguishability is not spatial meaning. Weyl
translations have zero covariance residual; the best Haar controls have
residual at least 0.304992 and do not close on the nine phase states.

### 4.2 Null observation

The qubit null buttons have identical fair outcomes and identical future
states; chart recovery under the predefined identifiability threshold is zero.
The qutrit null axis probes have centered predictive rank zero, and hidden-start
navigation remains at chance \(1/9\) despite additional observations. Null
controls show that an analysis cannot recover semantic structure from token
names alone.

### 4.3 External counters

A classical signed-word counter gives the qubit an exact \(3\times3\)
Manhattan chart even when quantum buttons are null. A classical nine-node
counter gives the null qutrit an exact toroidal graph while quantum predictive
rank remains zero. These are deliberately strong false positives for geometry
metrics that ignore provenance. P, A, and L reject them.

### 4.4 Label permutations

Fixed unknown permutations of action and outcome tokens leave all qutrit group
invariants unchanged. Qubit recovery is scored only modulo the square's \(D_4\)
gauge. These audits establish that reported semantics do not depend on a
privileged compass ordering. Episode-wise rescrambling, by contrast, should
destroy stable cross-episode landmarks.

### 4.5 Information--disturbance controls

The qutrit axis-instrument scan raises joint mutual information from zero to
0.14944 bits as strength goes from zero to one, while mean survival fidelity
falls from one to 0.75871. Hidden-start navigation likewise worsens under
oversensing: six sharp observations achieve label success 0.299 but target
fidelity only 0.414, compared with fidelity 0.878 after two. Information about
preparation, information about present state, and ability to reach a goal are
not interchangeable.

## 5. Held-out predictive evidence

The qubit common-probe model reconstructs arbitrary conditional history states
with mean trace-distance error 0.0350 using 300 samples per Pauli probe. This
validates predictive-state reconstruction but also enables the decisive
path-closure rejection.

For the qutrit Lüders axis instruments, 5,000 five-step strings train empirical
models and 2,000 independent strings test them. At sharp strength:

| predictor | held-out NLL (bits/outcome) |
|---|---:|
| marginal | 1.5856 |
| learned last-event table | 1.5207 |
| exact quantum filter | 1.5019 |

The filter gains 0.0837 bits/outcome over the marginal and 0.0187 over the
last-event model. The latter gap establishes that a short supplied event state
is not generally sufficient; longer-history predictive learning is required.

These results motivate evaluating action semantics through held-out strings,
not only through fitted one-step kernels.

## 6. Why the failures occur

### 6.1 Nonunitary qubit: channel noncommutativity

The hidden phase rotations commute, but their informative channels are
\(D_X\circ R_Z(\theta)\). The dephasing and rotation do not commute, so two
orders representing the same external displacement can yield different
states. Even commuting homogeneous dephasing would make a path with cancelling
detours lose more coherence than a shortest path. Exact word-count closure
therefore needs protected logical translations, explicit restriction to
canonical paths, or a quotient that treats disturbance as an internal fiber
rather than base position.

### 6.2 Integrated qutrit: action/report alignment

At the goal, the identity integrated action aligns its premeasurement state
with the goal report. At a nearest neighbor, one translation does exactly the
same before reporting. Equal action cost then forces equal continuation value.
Changing measurement strength alters the common value but cannot separate the
two shells. Restoring strict ordering requires changing terminal semantics,
retaining nondestructive arrival memory, varying costs, or leaving the
rank-one covariant measure-and-prepare family.

### 6.3 Topology is weaker than metric geometry

An action group can close exactly while its optimal hitting costs fail to be a
metric. A metric can be exact while action outcomes are uninformative. And a
set of actions can be individually identifiable without closing into any
translation group. The project must continue measuring topology, predictive
closure, information, and Bellman geometry independently.

## 7. Most promising next experiment

The next experiment should optimize a single family of **outcome-conditioned,
group-covariant qutrit channels** rather than choosing among the current fixed
grammars. The target is the smallest model satisfying the whole contract.

### 7.1 Environment and grammar

Use nine predictive landmarks carrying a projective \(\mathbb Z_3^2\) action.
Randomly rename all actions and outcomes. Candidate branch maps should obey a
covariance parameterization

\[
 \mathcal E_{g\cdot o}^{g\cdot a}
 =\mathcal U_g\circ\mathcal E_o^a\circ\mathcal U_g^{-1},
\]

which enforces relational homogeneity without assigning compass coordinates.
Unlike the Hesse rank-one measure-and-prepare family, retain outcome-conditioned
coherence or a protected logical component after reporting. Allow a small
internal syndrome/fiber state if it is operationally inferred and explicitly
quotiented from the spatial base.

### 7.2 Joint objective

Optimize physical Kraus parameters and the learned predictive quotient jointly
for:

- **P:** separated landmark future laws and stable predictive rank;
- **A:** distinct shifted-Hankel action operators, correct inverse/order/
  commutator relations, and token-gauge robustness;
- **B:** small Bellman residual under the actual outcome-conditioned branches;
- **M:** strict identity of indiscernibles, symmetry, and triangle inequalities;
- **E:** small Schoenberg negative eigenmass and rank-two planar stress, or the
  declared toroidal/local-chart analogue;
- **L:** local, site-independent action increments reconstructed only after
  learning the atlas;
- **information:** \(I(S;O\mid A)>0\), with immediate action identity and
  future predictive information reported separately;
- **closure:** same-displacement words agree on base predictions while any
  remaining difference is confined to a declared fiber.

The hard Bellman acceptance constraint must be

\[
 V_g(g)<V_g(s_{\rm edge})<V_g(s_{\rm diagonal})
\]

for every translated goal, with a preregistered positive margin. Correlation
alone is not acceptable.

One practical loss is

\[
 \mathcal L=
 \lambda_P\mathcal L_P+lambda_A\mathcal L_A+lambda_B\mathcal L_B
 +\lambda_M\mathcal L_M+lambda_E\mathcal L_E+lambda_L\mathcal L_L
 +\lambda_C\mathcal L_{\rm closure}
 -\lambda_I I(S;O\mid A),
\]

subject to completely positive trace-preserving instrument constraints and a
disturbance budget. Multiobjective/Pareto reporting is preferable to hiding
tradeoffs in one scalar score.

### 7.3 Learning protocol

1. Generate raw opaque action--outcome strings from anchors and hidden starts;
   do not expose the last outcome as a state label.
2. Train on short strings and reconstruct a spectral predictive-state model,
   including shifted action/outcome operators.
3. Infer the action group, predictive landmarks, and any base/fiber quotient
   before privileged alignment.
4. Optimize the physical instrument parameters against P/A/B/M/E/L,
   information, closure, and disturbance.
5. Freeze the model and test longer held-out strings, new start mixtures,
   unseen action words, and all translated goals.
6. Only after all invariant analyses are frozen, align the recovered atlas to
   hidden coordinates for Procrustes visualization.

### 7.4 Controls and stopping rule

Run matched cloned-action, null-probe, twirled-random-unitary, Haar-unitary,
external-counter, fixed-permutation, and episode-rescrambled controls. The
model succeeds only if:

1. held-out predictive likelihood improves over finite-history baselines;
2. P/A/B/M/E/L pass within preregistered tolerances;
3. mutual information is strictly positive with confidence bounds;
4. every repeated-path displacement class closes predictively on the base;
5. every goal has strict self < edge < diagonal Bellman ordering;
6. null and external-counter controls fail quantum P/A/L even if their
   hand-coded graph looks exact.

This experiment directly targets the missing intersection rather than another
model that excels at only topology or metric.

## 8. Reproduction

Run the qubit strand:

```bash
cd low_dimensional_hodology/informative_actions/qubit
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_qubit_experiment.py
```

Run the qutrit strand:

```bash
cd low_dimensional_hodology/informative_actions/qutrit
python -m unittest -v test_informative_qutrit.py
MPLBACKEND=Agg python informative_qutrit.py --episodes 2000 --seed 20260812
```

The qubit suite contains seven tests. The qutrit suite contains nine. Both
production runs use seed 20260812. The qutrit production bundle contains
34,000 navigation episodes, 20,000 training and 8,000 held-out five-step
prediction sequences, plus exact matrix calculations. The qubit bundle
contains 2,700 chart-recovery replicates, 1,750 predictive-tomography histories,
and a complete length-four path-equivalence audit.

## 9. Artifact map

### Theory

- `theory/THEORY_INFORMATIVE_ACTIONS.md`: predictive-state formalism, finite
  closure/equivalence theorems, information measures, gauge, P/A/B/M/E/L,
  qubit and qutrit constructions, controls, and proposed production study.

### Qubit

- `qubit/informative_qubit.py`: nonunitary instruments, analytic signatures,
  deterministic search, gauge-aware chart inference, predictive tomography,
  sequence geometry, controls, and path-equivalence audit;
- `qubit/run_qubit_experiment.py`: deterministic production runner and plots;
- `qubit/tests/test_informative_qubit.py`: seven structural tests;
- `qubit/RESULTS_QUBIT.md`: full derivation and interpretation;
- `qubit/results/summary.json`: selected parameters and headline diagnostics;
- `qubit/results/search_top.csv`, `chart_recovery.csv`, and
  `exact_predictive_signatures.csv`: search and action learning;
- `qubit/results/predictive_state_reconstruction.csv`: history tomography;
- `qubit/results/goal_geometry.csv` and `word_equivalence_audit.csv`: external
  Manhattan geometry versus actual predictive closure;
- `qubit/results/controls.csv` and `finite_control_recovery.csv`: random-unitary,
  null, and external-counter comparisons;
- `qubit/results/figures/`: action-chart and compositionality figures.

### Qutrit

- `qutrit/informative_qutrit.py`: exact Hesse and axis instruments, learning,
  Bellman solvers, prediction, navigation, controls, and plotting;
- `qutrit/test_informative_qutrit.py`: nine exact and stochastic tests;
- `qutrit/RESULTS_QUTRIT.md`: full derivation and production analysis;
- `qutrit/results/manifest.json`: configuration and headline invariant results;
- `qutrit/results/hesse_measure_prepare_kernel.csv` and
  `opaque_hesse_action_learning.csv`: exact and learned opaque kernels;
- `qutrit/results/hesse_integrated_action_cost.csv`,
  `hesse_bellman_cost.csv`, and `integrated_weak_scan.csv`: integrated no-go and
  separated benchmark;
- `qutrit/results/heldout_prediction.csv` and `navigation.csv`: predictive and
  policy evaluations;
- `qutrit/results/action_discovery.csv`, `future_effect_controls.csv`, and
  `dialectical_controls.csv`: covariance, information, null, Haar, permutation,
  and external-memory controls;
- `qutrit/results/predictive_distance.csv`, `predictive_mds.csv`,
  `candidate_instruments.csv`, and `information_disturbance.csv`: geometry and
  sensor diagnostics;
- `qutrit/figures/`: three production visualizations.

## 10. Final interpretation

The work changes the project's standard of evidence. A low stress plot, a
known action algebra, or a hand-coded goal counter is not enough. An emergent
space claim must trace a continuous chain:

\[
 \text{observable strings}
 \longrightarrow \text{predictive states}
 \longrightarrow \text{learned action transformations}
 \longrightarrow \text{closed spatial quotient}
 \longrightarrow \text{Bellman metric}
 \longrightarrow \text{2D local geometry}.
\]

The exact Pauli square completes that chain only with uninformative moves. The
nonunitary qubit breaks at closure. The integrated qutrit breaks at metric
identity. The separated qutrit completes topology and metric only by decoupling
movement from information.

The next research target is now unambiguous: preserve outcome-conditioned
information while protecting a group-covariant logical translation quotient
from the backaction that currently destroys either path closure or Bellman
ordering.
