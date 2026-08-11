# Comparative Experiments

## Questions

1. How do finite-history Q-learning, reward-only recurrent learning, and the
   predictive/geometry recurrent agent compare across different intervention
   repertoires and hidden initial states?
2. Does Euclidean separation of learned goal embeddings agree with strategy
   distance measured on held-out histories?
3. Does that embedding geometry agree with a more independent geometry of
   complete trajectory signatures?
4. Is directed cost calibrated to empirical hitting time, and how do individual
   intervention/outcome events move the vector of goal distances?

## Metrics

- **Success rate:** fraction of evaluation episodes completing the goal within
  the fixed horizon.
- **Conditional steps:** mean interventions among successful episodes. Read it
  beside success rate because lucky short runs can look artificially efficient.
- **Strategy distance:** square-root Jensen–Shannon distance between soft
  goal-conditioned policies, averaged over held-out encountered histories.
- **Trajectory distance:** Euclidean distance between action frequency,
  action/outcome frequency, normalized duration, and success signatures.
- **Reachability calibration:** blank-history learned distance compared with
  empirical hitting time and the full finite-horizon success curve.
- **Intervention displacement:** mean change in all learned distances after each
  observed action/outcome pair. Negative means closer.

## Reproduction

```bash
conda run -n qbist_spacetime python run_experiment_suite.py \
  --scenarios qubit-zx-weak:one,qubit-zx-weak:plus,qubit-pauli:plus-i,qubit-unsharp:mixed,qubit-pauli-sic:mixed,qutrit-mub:two \
  --backends tabular,gru,multi-gru \
  --seeds 0,1 \
  --episodes 400 \
  --eval-episodes 100 \
  --geometry-episodes 40 \
  --max-steps 15 \
  --epsilon-decay 0.99 \
  --device cpu \
  --output results/comparative

python plot_results.py results/comparative
```

## Results

The production suite completed 14,400 training episodes (107,804
interventions), 26,400 evaluation episodes (179,342 interventions), and 3,520
additional held-out geometry episodes. These totals span 36 trained agents: six
world/state scenarios, three backends, and two seeds. The raw configuration is
in `results/comparative/manifest.json`.

![Cross-environment performance](results/comparative/plots/performance_comparison.png)

### Control performance

Mean results across the two seeds are:

| world / initial state | finite history | GRU | predictive geometry GRU |
|---|---:|---:|---:|
| qubit Z/X/weak-Z / `one` | 81.0% | **87.4%** | 86.1% |
| qubit Z/X/weak-Z / `plus` | 87.8% | 80.9% | **92.4%** |
| qubit Pauli / `plus-i` | **86.3%** | 80.6% | 82.5% |
| qubit unsharp / `mixed` | **84.9%** | 78.8% | 82.0% |
| qubit Pauli+SIC / `mixed` | **71.3%** | 64.9% | 69.7% |
| qutrit MUB / `two` | 63.4% | **73.4%** | 70.3% |
| **macro-average** | 79.1% | 77.7% | **80.5%** |

No backend dominates every world. The recurrent state helps most clearly on the
qutrit scenario and on the original world starting from `one`; the tabular
agent is strongest on the Pauli, unsharp, and Pauli+SIC scenarios at this short
budget. The predictive geometry model has the best macro-average and a clear
advantage for the `plus` initial state, but not a universal one.

The more expressive agents are also less stable. The mean absolute difference
between their two seed success rates is 11.0 percentage points for the
predictive geometry GRU, 8.2 for the plain GRU, and 2.1 for tabular Q-learning.
The largest predictive-GRU seed gap is 24.3 points. With only two seeds, these
results are descriptive; they are not confidence intervals or significance
tests.

The hardest goals expose horizon and representation limits. Both recurrent
agents had zero success on the three-checkpoint qutrit goal `Z0_F0_P0`; the
tabular baseline reached 15%. The plain GRU also had zero success on
`Z0_SIC0`. Longer training and a compositional goal encoder are warranted.

![Learning curves](results/comparative/plots/learning_curves.png)

### Goal-geometry validation

![Geometry validation](results/comparative/plots/geometry_validation.png)

Across 12 predictive-geometry runs, embedding distance has mean Spearman rank
correlation 0.54 with held-out policy strategy distance (median 0.53, range
0.12–0.83). This is meaningful but imperfect agreement with the behavioral
quantity used by the regularizer.

Agreement with the more independent trajectory-signature distance is weaker:
mean correlation 0.31, median 0.27, and range -0.09–0.79. Geometry is therefore
not a single robust object yet. Some seeds organize whole trajectories well;
others mainly reproduce local policy similarity.

The first two principal components explain 77.3% of embedding variation on
average (53.0% plus 24.3%), making two-dimensional views informative but not
lossless.

The cost head is directionally informative but poorly calibrated as literal
hitting time. Across 79 goals with at least one held-out success, its prediction
correlates 0.47 with empirical conditional steps, with mean absolute error 1.75
interventions. Predictions occupy a compressed range (2.46–5.89) while observed
means range from 1.0 to 10.71. The finite-time reachability curves should be
preferred whenever risk and failure probability matter.

The Pauli+SIC displacement map gives the cleanest qualitative result. In seed
1, observing `Z:0`, `X:0`, and `Y:0` changes the learned distance to their
matching primitive goals by -1.58, -1.80, and -1.91 respectively. The four SIC
outcomes move their matching goals by -2.43 to -2.95, while most nonmatching
goal distances increase. The visualization makes measurement backaction look
like a directed movement through affordance coordinates rather than mere
information acquisition.

Representative per-run views:

- `results/comparative/plots/qubit-pauli-sic__mixed__multi-gru__seed1/goal_geometry.png`
- `results/comparative/plots/qubit-pauli-sic__mixed__multi-gru__seed1/intervention_displacements.png`
- `results/comparative/plots/qubit-pauli-sic__mixed__multi-gru__seed1/reachability_curves.png`

### Interpretation limits

- Every backend receives the same episode budget, not the same compute budget.
- Two seeds reveal instability but are insufficient for formal uncertainty.
- Conditional steps omit failures and can reward lucky policies; success rate
  and reachability curves remain primary.
- Goal embeddings use an explicit strategy-distance regularizer. Their held-out
  agreement is a calibration result, not spontaneous discovery without bias.
- The learned cost is trained off-policy with a minimum backup and is not yet a
  trustworthy metric in absolute units.
- Generated models and plots are exploratory artifacts, not frozen benchmark
  releases. The manifest and code make a larger rerun straightforward.

---

## Inverse-designed two-dimensional hodological space

The follow-up study asks a different question: can the environment and goals be
chosen so that learned all-pairs goal difficulty is genuinely compatible with
two-dimensional Euclidean space?

The full theory and interpretation are in `SPATIAL_HODOLOGY.md`. The executable
study is:

```bash
python spatial_hodology.py \
  --output results/spatial-hodology \
  --seeds 0,1,2 --episodes 6000 --pair-episodes 100 --max-steps 12
```

### Construction and protocol

- Hilbert dimension: 9, with canonical state
  \(\rho_0=\lvert1,1\rangle\langle1,1\rvert\).
- Actions: four axial moves, four stochastic diagonal moves, and one common
  nine-outcome projective place probe.
- Goals: obtain each of the nine outcomes from the common probe.
- Outer search: 81 diagonal success probabilities from 0.55 to 0.95.
- Selected probability: 0.715, close to the Euclidean cost-matching value
  \(1/\sqrt2\).
- Learner: place-symbol tabular Q-learning, with no coordinate input.
- Training: 6,000 uniform source--goal episodes per run.
- Evaluation: 100 trials for every one of 81 ordered source--goal pairs.
- Controls: identical hidden movement with only binary success/failure
  observation, and a place-observed cardinal-only Manhattan world.
- Total: 54,000 training and 72,900 evaluation episodes across nine runs.

### Exact design geometry

| design | 1D stress | 2D stress | 3D stress | 2D positive-spectrum fraction | negative-spectrum fraction | Euclidean distance correlation |
|---|---:|---:|---:|---:|---:|---:|
| optimized diagonal | 0.4305 | **0.0365** | 0.0360 | 0.934 | 0.081 | 0.995 |
| cardinal only | 0.4603 | 0.1416 | 0.1413 | 0.862 | 0.217 | 0.948 |

The optimized instrument has a clear 1D-to-2D stress drop and negligible
benefit from a third coordinate.

### Learned results

Mean ± sample standard deviation over three seeds:

| metric | optimized observed | optimized blind | cardinal observed |
|---|---:|---:|---:|
| all-pairs success | 1.000 ± 0.000 | 0.482 ± 0.023 | 1.000 ± 0.000 |
| 1D stress | 0.372 ± 0.011 | 0.450 ± 0.042 | 0.460 ± 0.000 |
| 2D stress | **0.071 ± 0.011** | 0.233 ± 0.032 | 0.142 ± 0.000 |
| 3D stress | 0.064 ± 0.011 | 0.228 ± 0.038 | 0.141 ± 0.000 |
| concealed-coordinate Procrustes \(R^2\) | 0.975 ± 0.005 | 0.965 ± 0.011 | 1.000 ± 0.000 |
| exact-cost correlation | 0.936 ± 0.003 | 0.865 ± 0.010 | 1.000 ± 0.000 |
| directionality | 0.122 ± 0.037 | 0.264 ± 0.093 | 0.000 ± 0.000 |

The proposed place-observed world satisfies the first objective. Its policy
reaches every source--goal pair, two dimensions reduce stress by a factor of
5.2 relative to one dimension, and a third dimension adds only a small
improvement. The concealed grid is recovered up to rotation/reflection/scale.

The cardinal control shows why Procrustes recovery alone is insufficient: it
recovers the grid arrangement perfectly while retaining a significantly less
Euclidean Manhattan metric. The blind control shows why a suitable hidden
transition graph alone is insufficient: inadequate observability or memory
damages both navigation and geometric calibration.

### Recorded failures

- Center-only policies reached 0.92--0.95 success from reset but only about 0.25
  across arbitrary source--goal pairs. A star of routes from one origin is not
  a navigable space.
- Random-source training with success/failure observations reached only 0.482
  all-pairs success at this budget.
- Place observations combined with a six-event literal history reached high
  success but fragmented equivalent places by arrival route and left 2D stress
  near 0.16.
- Quotienting the learner's state to the latest place symbol produced the final
  100%-success, 0.071-stress result.

### Figures

- `results/spatial-hodology/design_optimization.png`
- `results/spatial-hodology/learned_hodological_spaces.png`
- `results/spatial-hodology/performance_geometry_comparison.png`
- `results/spatial-hodology/emergent_policy_trajectories.png`
- `results/spatial-hodology/fiber_bundle_outlook.png` (explicitly a schematic)

### Scientific boundary

The construction uses localized qudit states and measure-and-prepare movement
channels. It proves that quantum-operational goal difficulty can recover a
concealed 2D space under controlled conditions. It does not show that space
generically emerges from coherent quantum dynamics, nor does it yet supply a
continuum, curvature dynamics, Lorentzian causality, or general relativity.

---

## Predictive atlas without online place symbols

This study attacks the strongest limitation of the first spatial result. The
controller receives no exact current-place symbol before movement. It must
infer a distribution over possible landmark outcomes from weak quantum beacon
histories and propagate that belief through blind actions.

Full mathematical and conceptual details are in `PREDICTIVE_ATLAS.md`. The
production command is:

```bash
conda run --no-capture-output -n qbist_spacetime \
  python predictive_atlas.py \
  --output results/predictive-atlas \
  --seeds 0,1,2 --scan-cycles 12 \
  --calibration-per-site 400 --test-per-site 200 --epochs 35 \
  --transition-trials 100 --pair-episodes 100 \
  --max-moves 12 --device cpu
python plot_predictive_atlas.py results/predictive-atlas
```

### Frozen production questions

1. Can a GRU predict a delayed terminal landmark from individually ambiguous
   QND beacon outcomes?
2. Does using all twelve scan cycles outperform an equal-budget control that
   retains only the last cycle?
3. Does a place-independent null sensor remain at chance despite receiving the
   same architecture, number of tokens, and delayed labels?
4. Can an outcome-conditioned empirical transition model support successful
   belief-state navigation?
5. Do frozen all-pairs policy costs recover the designed stochastic map and a
   low-dimensional Euclidean geometry?
6. How much performance and geometric fidelity are lost relative to an oracle
   that knows the current landmark online?

### Protocol and scale

- Quantum system: the same nine-level hidden walk and (p_d=0.715) diagonal
  movement used by the earlier optimized spatial world.
- Sensors: four binary diagonal QND instruments with overlapping horizontal,
  vertical, diagonal, and anti-diagonal response fields.
- Goal set: nine outcomes of one shared terminal landmark probe.
- Calibration: 3,600 scan/terminal-label examples per condition and seed;
  held-out localization test: 1,800 per condition and seed.
- Transition learning: 100 landmark-anchored trials for each of 9 sources and
  8 moves, or 21,600 survey trials across three seeds.
- Navigation: 100 episodes for every one of 81 ordered pairs, four conditions,
  and three seeds, or 97,200 episodes.
- Scan: 48 weak observations per navigation episode, totaling 4,665,600 held-
  out navigation sensor outcomes.
- Geometry: frozen-policy restricted movement cost, with fixed scan and
  terminal overhead removed; success is always reported alongside censored
  cost.

The transition model's mean privileged total-variation error is
(0.0144\pm0.0008), so model-estimation noise is small relative to the
localization ablations.

### Localization results

Mean ± sample standard deviation across three seeds:

| condition | accuracy | exact Bayes accuracy | negative log likelihood | Brier score | entropy |
|---|---:|---:|---:|---:|---:|
| full history | **0.964 ± 0.008** | 0.977 ± 0.004 | 0.112 ± 0.017 | 0.055 ± 0.010 | 0.144 ± 0.008 |
| last cycle | 0.463 ± 0.006 | 0.463 ± 0.006 | 1.386 ± 0.025 | 0.685 ± 0.006 | 1.374 ± 0.024 |
| null beacons | 0.114 ± 0.008 | 0.111 | 2.229 ± 0.005 | 0.896 ± 0.001 | 2.166 ± 0.003 |

The full recurrent model almost saturates the information-theoretic ceiling.
The last-cycle model also reaches its much lower ceiling, demonstrating that
its limitation is missing evidence rather than failed optimization. Null
performance remains at (1/9) chance.

### Navigation and geometry results

| condition | success | mean movement cost | exact-cost (r) | directionality | stress 1D | stress 2D | stress 3D | Procrustes (R^2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online oracle | 1.000 ± 0.001 | 1.485 ± 0.006 | 0.998 ± 0.001 | 0.024 ± 0.006 | 0.418 ± 0.012 | 0.040 ± 0.006 | 0.039 ± 0.005 | 1.000 ± 0.000 |
| full history | **0.973 ± 0.004** | 1.742 ± 0.033 | **0.948 ± 0.013** | 0.086 ± 0.016 | 0.407 ± 0.021 | **0.075 ± 0.005** | 0.043 ± 0.004 | **0.987 ± 0.005** |
| last cycle | 0.614 ± 0.005 | 5.223 ± 0.021 | -0.054 ± 0.020 | 0.254 ± 0.008 | 0.412 ± 0.013 | 0.192 ± 0.013 | 0.112 ± 0.008 | 0.451 ± 0.014 |
| null beacons | 0.484 ± 0.014 | 6.887 ± 0.133 | 0.628 ± 0.079 | 0.088 ± 0.013 | 0.456 ± 0.006 | 0.214 ± 0.009 | 0.102 ± 0.009 | 0.856 ± 0.108 |

For the full-history condition, 2D positive-spectrum fraction is
(0.870\pm0.017), negative-spectrum fraction is (0.085\pm0.011), and
concealed Euclidean-distance correlation is (0.942\pm0.012). The large
1D-to-2D stress reduction supports two-dimensional organization. The further
2D-to-3D reduction shows residual localization-induced distortion, so the
claim is strong recovery of a 2D spatial base, not exact rank-two geometry.

The null condition warns against reading Procrustes alignment alone. A small,
symmetric, censored world can sometimes align a poor matrix with the hidden
grid. Its low success, large cost, high stress, and high seed variance reject a
spatial-competence interpretation. Competence is evaluated before geometry.

### Recorded pilot and correction

The initial eight-cycle pilot reached 0.865 localization accuracy, 0.887
navigation success, and 0.184 2D stress. It showed that moderately successful
control can still destroy metric fidelity. Increasing to twelve scan cycles
and a larger calibration set raised the revised pilot to 0.971 localization,
0.977 navigation, 0.079 stress, and Procrustes (R^2=0.984). The production
configuration was frozen after that correction. Pilot directories are not part
of the retained result bundle; their numbers remain in this log for audit.

### Interpretation limits

- Delayed landmark outcomes supervise the localizer.
- Transition surveys start from previously verified landmarks.
- Coordinates and exact quantum variables are withheld online but the latent
  place basis and goal count were designed.
- Movement is entanglement-breaking and beacon instruments commute with place.
- The 48-probe fixed scan dominates total intervention count
  (approximately 50.5 per episode) and is omitted only from the relative
  movement geometry, not from reporting.
- Censored costs cannot be interpreted without the paired success matrix.
- Three seeds and nine places are sufficient for a controlled existence study,
  not formal universality or continuum claims.

### Figures and artifacts

- `beacon_fields_and_confusions.png`: sensor design and held-out predictions;
- `localization_learning_curves.png`: optimization behavior;
- `predictive_atlas_performance.png`: competence and metric diagnostics;
- `predictive_atlas_geometries.png`: oracle, predictive, and control maps;
- `belief_state_trajectories.png`: inferred motion with offline audit overlays.

The result bundle also retains every per-seed cost/success/confusion matrix,
the learned transition tables, nine GRU checkpoints, trajectories, summary
CSVs, and the exact run manifest.

---

## Equal-cost active predictive atlas

The active successor lets every beacon, movement, and terminal commitment cost
one intervention. Full theory, pilot history, and interpretation are in
`ACTIVE_PREDICTIVE_ATLAS.md`.

### Frozen production questions

1. Can learned active sensing approach fixed-scan reliability at materially
   lower total intervention cost?
2. Does goal-relative information value use fewer sensors than preserving a
   complete landmark posterior?
3. Does pragmatic state compression preserve a common movement geometry?
4. Can a null sensor support navigation when movement is reversible and doubly
   stochastic?
5. Does separating movement and sensing cost expose a spatial base plus an
   internal epistemic coordinate?

### Corrected environment

The first active pilot in the open grid failed as a geometry experiment:
repeated translations could herd unknown states to boundaries, giving a null-
sensor policy perfect corner control without localization. The production
world replaces translations with local random-unitary layer swaps. Their
classical kernels are doubly stochastic, their conditional quantum maps
preserve coherence, and their exact shortest-path matrix is identical to the
optimized (p_d=0.715) geometry.

This correction was structural, not a tuned penalty. Blind movement can no
longer turn a uniform prior into a localized posterior.

### Protocol

- Three seeds.
- 200 beacon calibration trials per landmark/action.
- 100 movement surveys per source/action.
- 50 held-out trials for every ordered source–goal pair.
- Six conditions: oracle, goal-relative active, exact-sensor active, fixed
  48-probe scan, atlas-preserving active, and null active.
- Unit sensing, movement, and commitment cost.
- Three-step goal-relative sensing lookahead with error penalty 100.
- Atlas-preserving reliability constraint 0.95.
- 60-intervention safety deadline.
- 72,900 navigation episodes.

### Competence and cost

Mean ± sample standard deviation across three seeds:

| condition | success | sensors | moves | total interventions | commit confidence |
|---|---:|---:|---:|---:|---:|
| oracle | 1.000 ± 0.000 | 0.00 | 1.482 ± 0.015 | 2.482 ± 0.015 | 0.991 ± 0.000 |
| goal-relative active | 0.938 ± 0.002 | 13.54 ± 0.10 | 1.576 ± 0.015 | 16.12 ± 0.11 | 0.941 ± 0.005 |
| exact-sensor active | 0.943 ± 0.004 | 13.07 ± 0.04 | 1.563 ± 0.011 | 15.63 ± 0.05 | 0.937 ± 0.000 |
| fixed-12 | **0.976 ± 0.002** | 48.00 | 1.487 ± 0.017 | 50.49 ± 0.02 | 0.968 ± 0.001 |
| atlas-preserving active | **0.965 ± 0.004** | **16.63 ± 0.53** | 1.494 ± 0.003 | **19.12 ± 0.53** | 0.962 ± 0.000 |
| active null | 0.111 ± 0.002 | 0.72 ± 0.12 | 37.80 ± 3.66 | 39.52 ± 3.54 | 0.122 ± 0.002 |

Atlas-preserving active sensing reduces total cost by 62.1% relative to the
fixed scan while losing 1.14 percentage points of success. Exact beacon fields
barely improve goal-relative control, so its residual error is primarily the
cost of pragmatic state compression rather than calibration.

### Movement and total geometry

| condition | exact-cost (r) | movement 2D stress | movement 3D stress | movement Procrustes (R^2) | total 2D stress |
|---|---:|---:|---:|---:|---:|
| oracle | 0.995 ± 0.000 | 0.047 ± 0.009 | 0.046 ± 0.008 | 0.999 ± 0.000 | 0.115 ± 0.004 |
| goal-relative active | 0.724 ± 0.054 | 0.193 ± 0.015 | 0.097 ± 0.007 | 0.900 ± 0.063 | 0.286 ± 0.006 |
| exact-sensor active | 0.694 ± 0.103 | 0.194 ± 0.013 | 0.093 ± 0.009 | 0.855 ± 0.105 | 0.279 ± 0.006 |
| fixed-12 | 0.962 ± 0.005 | 0.079 ± 0.014 | 0.054 ± 0.020 | 0.993 ± 0.001 | 0.310 ± 0.001 |
| atlas-preserving active | **0.929 ± 0.012** | **0.111 ± 0.006** | 0.060 ± 0.010 | **0.987 ± 0.004** | 0.280 ± 0.005 |
| active null | 0.094 ± 0.189 | 0.300 ± 0.004 | 0.193 ± 0.000 | 0.197 ± 0.166 | 0.297 ± 0.004 |

Goal-relative state is cheaper but damages the shared geometry. The full
landmark posterior costs roughly three additional probes and preserves a much
better common atlas. Total intervention cost is substantially less Euclidean
than movement cost because sensing is epistemic work, not spatial displacement.

### Base/fiber diagnostics

Atlas-preserving sensing burden has a weak, seed-variable correlation
(0.281\pm0.191) with spatial distance. Only
(0.037\pm0.026) of its variance is explained by additive source and goal
effects. Goal-relative sensing has a stronger positive correlation
(0.580\pm0.075), while its endpoint-only additive model explains just
(0.032\pm0.008). Distance affects information demand, but neither distance nor
separate endpoint taxes explain the pair-specific sensing matrix.

The fiber visualization places belief barycenters in the learned movement
plane and uses normalized entropy as height. Trajectories descend in
uncertainty before moving across the base. This is an operational precursor,
not yet evidence of a mathematical fiber bundle.

### Figures

- `reversible_design_and_calibration.png`
- `active_atlas_performance.png`
- `active_atlas_geometries.png`
- `active_sensing_overhead.png`
- `epistemic_fiber_trajectories.png`

### Reproduction

```bash
conda run --no-capture-output -n qbist_spacetime \
  python active_predictive_atlas.py \
  --output results/active-atlas --seeds 0,1,2 \
  --beacon-trials 200 --transition-trials 100 \
  --pair-episodes 50 --max-interventions 60 \
  --max-movement-cost 12 --failure-penalty 100 \
  --confidence-threshold .95 --sensing-lookahead 3
python plot_active_atlas.py results/active-atlas
```

---

## Exactly solvable low-dimensional hodology

### Frozen questions

1. Can nine goals form a 2D lattice when \(d=2\) or \(3\)?
2. Which result concerns word/control geometry, which concerns quantum-state
   distinguishability, and which is supplied by a goal automaton?
3. When is an exact lattice planar Euclidean rather than merely a two-
   generator graph or torus?
4. What fails under finite verification tolerance, measurement backaction,
   and null quantum dynamics?
5. Which parts admit closed-form Bellman proofs rather than fitted simulation?

### Exact qubit word construction

The qubit starts in \(|+\rangle\). Two commuting phase rotations have angles

\[
\alpha=2\pi/9+10^{-2}\sqrt2,
\qquad
\beta=2\pi/3+10^{-2}\sqrt3.
\]

Their projective action is faithful on \(\mathbb Z^2\). Nine goals are the
nonorthogonal rays \(U^iV^j|+\rangle\), \(i,j\in\{0,1,2\}\), or equivalently
bounded-horizon action-sequence languages with the same net exponents. The
all-pairs cardinal word cost is exactly open-grid Manhattan distance.

For every nonzero patch displacement \(\delta\), a binary unit-cost instrument
applies \(U_\delta\) with probability \(1/\|\delta\|_2\) and identity otherwise.
The exact expected hitting cost is Euclidean distance. Fifty thousand trials
for each of 24 displacement classes—1.2 million geometric samples—gave maximum
mean error 0.029 interventions.

| qubit diagnostic | result |
|---|---:|
| all-pairs exact-state terminal infidelity | \(6.7\times10^{-16}\) |
| Euclidean cost maximum analytic error | 0 |
| worst one-shot false goal acceptance | 0.9025 |
| pairs shortened at infidelity tolerance \(10^{-3}\) | 30.6% |
| qutrit-Weyl open-grid pairs with torus shortcuts | 16/36 |

This is an exact existence result but not a robust localization model. The
orbit is dense on one phase circle and its coordinate inverse is discontinuous.

### Exact order-11 qutrit phase chart

The stronger benchmark uses

\[
|\psi_0\rangle=(\sqrt{3/8},1/2,\sqrt{3/8})^T,
\quad A=\operatorname{diag}(0,1,0),
\quad B=\operatorname{diag}(0,1/2,1).
\]

The commuting order-11 unitaries generate 121 distinct nonorthogonal qutrit
states. The phase-generator covariance is exactly \((3/16)I_2\). A nine-goal
\(3\times3\) patch lies below the wrap radius, and unit-cost retry instruments
for all phase displacements yield exact Euclidean optimal cost.

| qutrit phase diagnostic | result |
|---|---:|
| physical Hilbert dimension | 3 |
| distinct phase-orbit states | 121 |
| selected goals | 9 |
| minimum orbit Frobenius separation | 0.3451 |
| maximum Bellman residual | \(8.9\times10^{-16}\) |
| patch Euclidean cost error | 0 |
| MDS reconstruction error | \(1.3\times10^{-15}\) |
| positive Schoenberg rank | 2 |
| control/trace-distance correlation | 0.9889 |
| Monte Carlo waiting-time maximum error | 0.0237 |

The action law encodes inverse Euclidean length by design. The experiment
proves compatibility of low Hilbert dimension, a genuine 2D phase manifold,
sequence-labelled control, and exact Euclidean hodology; it does not derive the
probability law from more primitive physics.

### Fully operational qutrit goals

The Hesse SIC experiment uses four Weyl unitary controls and one common
nine-outcome qutrit instrument. Goal \(g\) is “obtain SIC outcome \(g\),” not
“occupy basis state \(g\).” Exact value iteration recovers

\[
V_g(s)=6+d_T(s,g),
\]

with correlation one between baseline-subtracted goal cost and the
\(\mathbb Z_3^2\) torus control distance. The price is topology: planar
classical-MDS stress is 0.383 and the squared-distance Gram matrix is
indefinite.

A second phase-grid POVM has control/excess-cost correlation 0.9961. Requiring
one, two, or three consecutive reports of the translated target outcome keeps
the same correlation while baseline cost becomes 5.588, 22.353, and 72.647.
The result separates goal location from internal sequence difficulty.

### Qubit approximation and automaton controls

A 180-angle search over
\(|\psi_{xy}\rangle=R_x(\theta)^xR_y(\theta)^y|0\rangle\) selected
\(\theta=0.309665\) subject to minimum trace separation 0.12.

| qubit patch diagnostic | result |
|---|---:|
| Euclidean distance correlation | 0.98975 |
| 2D classical-MDS stress | 0.00736 |
| minimum trace separation | 0.12557 |
| one-cell commutator defect | 0.01025 |
| mean / maximum translation defect | 0.00942 / 0.04908 |

The flat limit is singular: as \(\theta\to0\), all states coalesce. A null
qubit with an unchanged nine-state DFA retains the entire automaton grid while
all physical trace distances vanish. In a separate counter task, matched fair
coins give the additive cost surface

\[
\begin{pmatrix}0&2&4\\2&4&6\\4&6&8\end{pmatrix},
\]

whereas projective qubit \(X/Z\) backaction gives

\[
\begin{pmatrix}0&3&4\\3&5&6\\4&6&7\end{pmatrix}.
\]

The qubit's additive residual is one versus numerical zero for the coin. This
is a controlled example of quantum backaction warping a history-carried base.

### Figures and artifacts

- `low_dimensional_hodology/results/figures/exact_qubit_lattice.png`
- `low_dimensional_hodology/results/qutrit-phase/qutrit_phase_lattice.png`
- `low_dimensional_hodology/search/figures/low_dimensional_search.png`
- `low_dimensional_hodology/search/figures/qutrit_and_counter_comparison.png`
- exact all-pairs, tolerance, dephasing, and waiting-time CSV files;
- qutrit probability, policy, cost, confirmation-sequence, and MDS matrices;
- machine-readable manifests and summaries.

### Reproduction

```bash
python -m unittest discover -s low_dimensional_hodology/tests -v
python -m unittest discover -s low_dimensional_hodology/search -p 'test_*.py' -v
MPLBACKEND=Agg python low_dimensional_hodology/run_exact_experiments.py
MPLBACKEND=Agg python low_dimensional_hodology/run_qutrit_phase_experiments.py
MPLBACKEND=Agg python low_dimensional_hodology/search/search_low_dimensional.py
```
