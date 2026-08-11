# Project Log

This is a chronological record of implementation decisions, checks, and
interpretive cautions. It complements the stable architecture guide and keeps
failed assumptions visible.

## 2026-08-10 — Repository audit and baseline

- Read `VISION.md` and `ARCHITECTURE.md` in full, then inspected every Python
  source file.
- Preserved the pre-existing uncommitted architecture note specifying
  `conda activate qbist_spacetime`.
- Found that the base Python 3.13 environment lacks PyTorch; the documented
  `qbist_spacetime` environment uses Python 3.14, NumPy 2.5.1, and PyTorch
  2.13.0. All original command-line programs compile and complete short runs in
  that environment.
- Found no tests, project environment file, aggregate experiment runner, plot
  generator, or independent validation of the claimed goal geometry.
- Found documentation drift: `ARCHITECTURE.md` referred to a missing
  `quantum_environment.py`.

### Reasoning

The original one-qubit environment could compare learners but could not test
the vision's central claim that operational state depends on the intervention
repertoire. The next implementation step therefore had to vary the world, not
just hyperparameters.

## 2026-08-10 — Environment generalization

- Added five finite-dimensional environments: `qubit-zx-weak`,
  `qubit-pauli`, `qubit-unsharp`, `qubit-pauli-sic`, and `qutrit-mub`.
- Added additional qubit states along the Y axis and multiple qutrit states.
- Added world-specific goal repertoires containing single checkpoints,
  reversed sequences, and three-checkpoint compositions.
- Kept `QuantumEnvironment` backward compatible and retained the strict public
  interaction boundary `reset()` / `step(action)`.
- Added action-specific outcome counts. In the Pauli+SIC world the neural
  predictor has four slots, but outcomes 2 and 3 are impossible for projective
  actions. Those logits are masked exactly, and goals are validated against
  each action's actual alphabet.

### Reasoning

A single global outcome count is convenient for neural tensors but is not an
operational license to imagine nonexistent outcomes. Explicit masks prevent a
subtle probability-model error while preserving a fixed-width head.

## 2026-08-10 — Tests and reproducible experiments

- Added coverage for quantum completeness, density-matrix invariants,
  deterministic projective behavior, reproducibility, parsers, goal semantics,
  runner behavior, all learning backends, mixed-outcome masks, and geometry
  utilities.
- Added `environment.yml` and `.gitignore`.
- Added `run_experiment_suite.py`, which checkpoints tidy data after every run
  and saves recurrent weights.
- Added `goal_geometry.py`, which measures embedding distance, held-out policy
  distance, trajectory-feature distance, reachability calibration, finite-time
  curves, and intervention displacement without consulting hidden physics.
- Added `plot_results.py` with an end-to-end smoke-tested figure pipeline.
- The smoke suite used two worlds and three backends. It generated every data
  and plot family; its tiny sample is a software check, not scientific evidence.

### Geometry caution

Embedding geometry is explicitly regularized toward strategy similarity.
Agreement with held-out strategy distance is validation of generalization and
calibration, not evidence that geometry appeared without inductive pressure.
Trajectory distance is more independent, and intervention-displacement maps
visualize the separate directed agent-to-goal geometry.

## 2026-08-10 — Comparative study

The exact production configuration, results, and interpretation are recorded
in `EXPERIMENTS.md`. Raw and rendered artifacts live under `results/` and can be
regenerated from the manifest.

- Completed 36 matched training runs and 26,400 held-out evaluation episodes.
- Generated 51 figures: three global summaries and four geometry views for
  each of 12 predictive-geometry runs.
- The predictive geometry GRU has the highest macro-average success (80.5%),
  but the winning backend changes by environment and neural seed variance is
  substantial.
- Goal embeddings agree moderately with held-out policy geometry (mean
  Spearman 0.54) and more weakly with trajectory geometry (0.31).
- Learned reachability correlates with hitting time (Pearson 0.47) but is
  compressed and miscalibrated (MAE 1.75 interventions). This result argues for
  treating the finite-horizon reachability curve as a first-class object.

## 2026-08-10 — Publication manuscript

- Added `GOAL_GEOMETRY_PAPER.tex`, a self-contained pedagogical manuscript that
  derives the quantum instrument model, MDP/POMDP formalism, Bellman equations,
  tabular and recurrent Q-learning, predictive auxiliary learning, evaluation
  methodology, and three complementary notions of goal geometry.
- Integrated six figures generated from the comparative study, plus two native
  TikZ diagrams, a results table, the complete production configuration, and a
  twenty-item bibliography centered on primary literature.
- Distinguished symmetric embedding and behavior geometry from directed
  reachability geometry. The paper explicitly discusses circularity in the
  policy-regularized embeddings, finite-horizon censoring, seed uncertainty,
  and the difference between a useful operational representation and a claim
  about an intrinsic quantum state space.
- Added a future-research program covering compositional goals, multi-step
  predictive state, geometry-based planning, distributional reachability,
  directed non-Euclidean models, intervention selection, cross-world transfer,
  uncertainty, and hardware validation.
- Built the manuscript with `latexmk` under TeX Live 2025. The final PDF is 17
  pages; all citations and cross-references resolve, and representative pages
  were rendered and visually inspected at publication size.

## 2026-08-11 — First emergent-space objective

### Research decision

Reframed “space from goals” as a bilevel inverse-design problem. The outer loop
chooses quantum instruments and goals to minimize low-dimensional embedding
error; the inner loop learns goal-conditioned behavior; the final claim is
evaluated on the learned all-pairs hitting-cost matrix, not on the designed
latent graph alone.

Chose a nine-dimensional first construction with basis states corresponding to
nine concealed places. This is intentionally the smallest nontrivial 2D test:
it permits corners, edges, a center, axial and diagonal displacement, and enough
points for meaningful dimensional diagnostics while remaining exhaustively
evaluable.

### Simulator changes

- Generalized `Measurement` so one observed outcome can aggregate multiple
  unobserved Kraus events. The simulator now sums branch probabilities and
  conditional post-measurement states correctly.
- Added three validated environments: `qudit-grid-3x3`,
  `qudit-grid-3x3-blind`, and `qudit-grid-3x3-cardinal`.
- Added open-boundary axial/diagonal movement instruments, a common nine-outcome
  place probe, localized/mixed initial states, and nine place goals.
- Re-ran quantum completeness and density-matrix invariant tests after each
  instrument change.

### Inverse design

- Added `spatial_hodology.py` with exact stochastic-shortest-path costs,
  classical MDS, deterministic SMACOF, normalized stress, spectrum diagnostics,
  distance correlation, and privileged Procrustes validation.
- Searched 81 diagonal success probabilities. The objective selected 0.715,
  close to the analytically motivated value `1/sqrt(2)`.
- Exact optimized 1D/2D/3D stresses are 0.4305/0.0365/0.0360. The cardinal
  control's 2D stress is 0.1416, confirming that a grid arrangement does not by
  itself make shortest-path distance Euclidean.

### Failures that changed the design

1. Training only from the center produced 0.92--0.95 reset success but roughly
   0.25 arbitrary all-pairs success. This was rejected as a “space”: it learned
   routes from one origin rather than transferable local navigation.
2. Uniform random starts with only success/failure move observations reached
   about 0.48 all-pairs success at 6,000 episodes. The hidden geometry existed,
   but the learner could not localize itself reliably with finite memory.
3. Destination place symbols raised success to 0.998, but a six-event literal
   history treated the same place reached by different routes as distinct and
   left learned 2D stress around 0.16.
4. `PlaceQAgent` deliberately quotients histories by latest place symbol. This
   is Markov-sufficient yet coordinate-free, and produced the final result.

These are conceptual failures, not discarded tuning attempts. They show that
all-pairs source coverage, operational localization, and the correct history
equivalence relation are conditions for a usable emergent space.

### Production result

- Completed nine runs: three worlds × three seeds.
- Completed 54,000 training episodes and 72,900 ordered all-pairs evaluation
  episodes.
- The optimized place-observed learner achieved 100% all-pairs success in all
  seeds. Mean 1D/2D/3D stress is 0.3715/0.0709/0.0642; concealed-coordinate
  Procrustes `R^2` is 0.9754; learned/exact cost correlation is 0.9355.
- The blind control achieved 48.2% all-pairs success and 2D stress 0.2330.
- The cardinal control achieved 100% success but retained 2D stress 0.1416.
- Generated design, learned-space, performance, and emergent-trajectory plots,
  plus a clearly labeled non-empirical fiber-bundle schematic.

### Interpretation and next boundary

This is an existence proof in a classical stochastic corner of quantum theory:
localized qudit states and entanglement-breaking movement instruments. It is
not evidence that generic coherent quantum dynamics produces space. The next
step is to remove explicit place localization, optimize general CPTP channels
and goal subsets, scale topology, then test whether a total spatial/internal
hodological geometry factorizes locally as a fiber bundle. Connection,
holonomy, and stochastic curvature diagnostics must precede any general-
relativistic interpretation.

The complete equations, metrics, results, failures, and staged research program
are recorded in `SPATIAL_HODOLOGY.md`; aggregate results are also appended to
`EXPERIMENTS.md`.

- Expanded `GOAL_GEOMETRY_PAPER.tex` from 17 to 24 pages with a full spatial
  inverse-design section, four new empirical figures, the fiber-bundle
  schematic, six additional primary references, and a causal/curvature
  outlook. Rebuilt it successfully with resolved citations and cross-references
  and visually inspected the new theory, result, and bibliography pages.
