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
