# Agent-Centered Quantum Reinforcement Learning

This repository studies agents that learn quantum measurement strategies from
their own action/outcome histories. The simulator knows the density matrix and
Kraus operators; the agent never receives either. The central research objects
are a learned predictive state, a behavioral geometry among goals, and a
directed reachability relation from the current history to each goal.

The conceptual motivation is in [VISION.md](VISION.md), the software design is
in [ARCHITECTURE.md](ARCHITECTURE.md), the reproducible study protocol and
results are in [EXPERIMENTS.md](EXPERIMENTS.md), and the development record is
in [PROJECT_LOG.md](PROJECT_LOG.md).

The first inverse-designed emergent-space study, including its quantum
instruments, failures, dimensional tests, and fiber-bundle research program, is
in [SPATIAL_HODOLOGY.md](SPATIAL_HODOLOGY.md).

The preceding major step is the
[predictive-atlas study](PREDICTIVE_ATLAS.md). It removes exact online place
reports: a GRU integrates ambiguous weak-beacon outcomes, a learned transition
model propagates its belief through blind movements, and goal-conditioned
planning reconstructs a two-dimensional cost geometry. The three-seed
production agent achieves 97.3% all-pairs success and recovers the concealed
arrangement with Procrustes \(R^2=0.987\), while memory-limited and
place-independent controls fail.

The next step is now implemented in
[ACTIVE_PREDICTIVE_ATLAS.md](ACTIVE_PREDICTIVE_ATLAS.md). Sensing, movement,
and terminal commitment all cost one intervention, and the agent actively
chooses among them. A reversible random-unitary movement design closes a
boundary-homing loophole. The atlas-preserving policy uses 19.12 total
interventions instead of the fixed baseline's 50.49 while retaining 96.5%
success and Procrustes (R^2=0.987). Separating movement cost from adaptive
sensing cost gives the project's first empirical spatial-base/epistemic-fiber
visualization.

The new [low-dimensional hodology miniproject](low_dimensional_hodology/README.md)
removes the nine-level position register. It proves that a qubit can carry an
exact \(3\times3\) history/word lattice and that a qutrit phase manifold can
carry 121 nonorthogonal orbit states while a nine-goal local chart has exactly
Euclidean optimal cost. Matched qutrit SIC, qubit tangent-plane, sequence-
counter, and null-automaton studies distinguish geometry in quantum dynamics
from geometry supplied only by goal memory.

Its [emergent-local-metric successor](low_dimensional_hodology/emergent_local_metric/README.md)
removes the displacement-specific retry catalog. A single translation-covariant
qutrit action stencil is fitted at radius four and tested through radius
twelve: held-out Euclidean error falls from 30.03% with four directions to
1.62% with 32. The accompanying theorem proves that every fixed finite catalog
has a polygonal, Finsler large-scale unit ball, so exact global Euclidean
geometry requires continuous or asymptotically dense local controls. A
60,000-episode hidden-start study then separates localization of an initial
preparation from control of the present measurement-conditioned state, giving
the miniproject's first operational base--fiber split.

The latest [informative-actions study](low_dimensional_hodology/informative_actions/README.md)
removes an additional hidden assumption: a button does not mean ``east'' merely
because the simulator gives it that name. Action and outcome labels are
shuffled, and meanings must be reconstructed from controlled future-outcome
statistics. A nonunitary qubit learns a two-axis action chart but fails
predictive path independence. An exactly solvable qutrit instrument has
informative outcomes (0.251629 bits), and its observed kernels recover a
commuting, transitive \(\mathbb Z_3^2\) action group, but its integrated Bellman
cost has self = neighbor and is only a pseudometric. This identifies the next
design target precisely: combine observable action semantics, compositional
predictive states, and a nondegenerate low-dimensional goal metric in the same
instrument family. The focused account is available as a
[19-page pedagogical paper](low_dimensional_hodology/informative_actions/INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.pdf)
with its
[LaTeX source](low_dimensional_hodology/informative_actions/INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex).

The publication-style account is available as both
[LaTeX source](GOAL_GEOMETRY_PAPER.tex) and a
[compiled paper](GOAL_GEOMETRY_PAPER.pdf). It develops the quantum and
reinforcement-learning background from first principles, states the evaluation
protocol and limitations, and interprets the learned goal geometry using the
saved comparative, inverse-design, and predictive-atlas figures. Rebuild it
with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error GOAL_GEOMETRY_PAPER.tex
```

## Quick start

```bash
conda env update --file environment.yml --prune
conda activate qbist_spacetime
python -m unittest discover -s tests -v
```

Run one backward-compatible baseline:

```bash
python compare_backends.py --backend tabular --episodes 2000
```

Select a different intervention world:

```bash
python compare_backends.py \
  --environment qubit-pauli-sic \
  --backend multi-gru \
  --episodes 2000 \
  --geometry-prefix results/sic/geometry
```

Run a controlled suite and render all figures:

```bash
python run_experiment_suite.py --episodes 1000 --seeds 0,1,2 --output results/standard
python plot_results.py results/standard
```

Reproduce the first two-dimensional hodological-space study:

```bash
python spatial_hodology.py \
  --output results/spatial-hodology \
  --seeds 0,1,2 --episodes 6000 --pair-episodes 100 --max-steps 12
```

Reproduce the predictive atlas without online place symbols:

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

Reproduce the equal-cost active atlas:

```bash
conda run --no-capture-output -n qbist_spacetime \
  python active_predictive_atlas.py \
  --output results/active-atlas --seeds 0,1,2 \
  --beacon-trials 200 --transition-trials 100 \
  --pair-episodes 50 --max-interventions 60 \
  --failure-penalty 100 --confidence-threshold .95 \
  --sensing-lookahead 3
python plot_active_atlas.py results/active-atlas
```

Reproduce the exactly solvable low-dimensional studies:

```bash
python -m unittest discover -s low_dimensional_hodology/tests -v
python -m unittest discover -s low_dimensional_hodology/search -p 'test_*.py' -v
MPLBACKEND=Agg python low_dimensional_hodology/run_exact_experiments.py
MPLBACKEND=Agg python low_dimensional_hodology/run_qutrit_phase_experiments.py
MPLBACKEND=Agg python low_dimensional_hodology/search/search_low_dimensional.py

python -m unittest discover \
  -s low_dimensional_hodology/emergent_local_metric/control/tests -v
python -m unittest discover \
  -s low_dimensional_hodology/emergent_local_metric/localization \
  -p 'test_*.py' -v
MPLBACKEND=Agg python \
  low_dimensional_hodology/emergent_local_metric/control/run_local_control.py
MPLBACKEND=Agg python \
  low_dimensional_hodology/emergent_local_metric/localization/localization_experiment.py \
  --episodes 1500 --seed 20260811
```

`plot_results.py` deliberately has no PyTorch dependency. It can run in any
Python environment containing NumPy and Matplotlib.

## Environment catalog

| name | hidden system | available interventions | default hidden state |
|---|---:|---|---|
| `qubit-zx-weak` | qubit | projective Z, projective X, weak Z | `one` |
| `qubit-pauli` | qubit | projective Z, X, Y | `plus-i` |
| `qubit-unsharp` | qubit | weak Z, X, Y | `mixed` |
| `qubit-pauli-sic` | qubit | projective Z, X, Y, four-outcome tetrahedral SIC | `mixed` |
| `qutrit-mub` | qutrit | three mutually unbiased projective bases | `two` |
| `qudit-grid-3x3` | nine-level qudit | eight place-reporting moves plus a common place probe | `center` |
| `qudit-grid-3x3-blind` | nine-level qudit | eight success/failure moves plus a common place probe | `center` |
| `qudit-grid-3x3-cardinal` | nine-level qudit | four place-reporting moves plus a common place probe | `center` |
| `qudit-grid-3x3-beacons` | nine-level qudit | eight blind moves, four overlapping binary QND beacons, terminal place probe | `center` |
| `qudit-grid-3x3-null-beacons` | nine-level qudit | same, but place-independent beacon fields | `center` |
| `qudit-grid-3x3-reversible-beacons` | nine-level qudit | eight coherent random-unitary layer swaps, four learned QND beacons, terminal probe | `center` |
| `qudit-grid-3x3-reversible-null-beacons` | nine-level qudit | same reversible motion with place-independent beacons | `center` |

Qubit states include `zero`, `one`, `plus`, `minus`, `plus-i`, `minus-i`, and
`mixed`. Qutrit states include `zero`, `one`, `two`, `plus`, and `mixed`. Each
world supplies goals matched to its action and outcome alphabet; passing
`--goals` overrides that catalog.

The three spatial worlds use the same nine place goals. Their integer outcome
identities contain no coordinates. The optimized world uses stochastic
diagonal Kraus maps whose success probability is chosen by a low-dimensional
geometry objective; the blind and cardinal variants isolate observability and
metric anisotropy.

The beacon world weakens the observation boundary further. Its movement
reports are binary, each beacon sample is ambiguous, and the sharp place probe
is terminal during atlas evaluation. Exact landmark outcomes may label a past
scan or score a commitment but never inform a later action in the same
navigation episode.

## Backends

- `tabular`: Q-learning over a literal finite action/outcome history.
- `gru`: a goal-conditioned recurrent state trained only through reward.
- `multi-gru`: the recurrent controller plus an outcome predictor, a directed
  cost-to-go head, and behaviorally regularized goal embeddings.

The older `predictive_gru_q_learning.py` remains a focused single-goal
prediction baseline. `multi_goal_q_learning.py` remains a compatibility entry
point for the tabular baseline.

## Goal syntax

A goal is an ordered subsequence of desired action/outcome checkpoints. Other
events may occur between checkpoints:

```text
prepare_then_test=0:0,1:1
```

This means “eventually perform action 0 and experience outcome 0, then later
perform action 1 and experience outcome 1.” Goal progress is first-person
information computable from the observed history. Validation rejects actions
and outcomes unavailable in the selected environment.

## Experiment artifacts

`run_experiment_suite.py` writes:

- `training_episodes.csv` and `evaluation_episodes.csv`: tidy raw outcomes;
- `summary.csv`: per-goal and overall metrics for every run;
- `manifest.json`: exact configuration and seed metadata;
- `models/*.pt`: recurrent model weights;
- `geometry/<run>/`: embedding, policy, trajectory, reachability, reachability
  curve, and intervention-displacement data.

`plot_results.py` turns these into performance heatmaps, learning curves, goal
embedding projections, independent strategy-distance comparisons, trajectory
signatures, reachability calibration plots, and action/outcome displacement
maps.

`predictive_atlas.py` writes a separate auditable bundle under
`results/predictive-atlas/`: calibration and navigation summaries, epoch-level
curves, per-seed confusion/cost/success/transition matrices, nine GRU
checkpoints, belief trajectories, a manifest, and five figures. The companion
`plot_predictive_atlas.py` has no PyTorch dependency.

`active_predictive_atlas.py` writes `results/active-atlas/`: learned beacon
calibration, per-seed movement/total/sensing/success matrices, representative
belief trajectories, a complete summary and manifest, and five figures.
`plot_active_atlas.py` is likewise a lightweight artifact renderer.

## Scientific boundary

Agent code must not inspect `env._rho`, `env._kraus`, or any quantity derived
from them. Tests may inspect private simulator state solely to verify quantum
validity. Offline scientific analysis may compare a learned representation to
hidden physics only when explicitly labeled as privileged analysis; the
current geometry pipeline does not do so.

The predictive atlas obeys a stricter online boundary. The controller receives
only its goal, chosen actions, binary outcomes, weak scan histories, and a
terminal landmark report after commitment. Its density matrix, Kraus
operators, response likelihoods, coordinates, and exact current landmark are
withheld. Offline tests and figures may decode those variables only when
explicitly labeled. Landmark-delayed training and landmark-anchored transition
surveys remain supervision; the experiment is not described as fully
unsupervised discovery.

The active atlas adds a further separation. Its online controller knows only
learned landmark-conditioned observation and transition tables, its goal, and
its own action/outcome record. Exact sensor fields are used only in a named
upper-bound control and offline calibration metric. The movement instruments
are coherent random-unitary channels, but delayed landmark supervision and a
designed place basis remain. The entropy-height “fiber” is explicitly a
provisional operational visualization, not yet a mathematical fiber bundle.
