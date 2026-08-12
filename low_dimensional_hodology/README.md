# Low-dimensional exact hodology

This miniproject asks whether nine spatial goals require nine orthogonal quantum
states. They do not. A single qubit supports an exact but fragile
\(3\times3\) word lattice, while a qutrit supports a robust two-phase manifold
containing an exact Euclidean \(3\times3\) control-cost chart. The goals are
nonorthogonal rays and equivalence classes of action/outcome sequences rather
than basis labels.

The qubit proof and its limitations are in
[`RESULTS_EXACT.md`](RESULTS_EXACT.md), the general theory and proposed
necessary-and-sufficient hierarchy are in [`THEORY.md`](THEORY.md), and the
independent skeptical simulations are in
[`search/SEARCH_REPORT.md`](search/SEARCH_REPORT.md). Implementations depend
only on NumPy; Matplotlib is used by the artifact generators.

For a self-contained, pedagogical development with complete proofs, worked
Bellman and Schoenberg calculations, exact qubit and qutrit constructions,
figures, experimental audits, and research outlook, see
[`EXACT_LOW_DIMENSIONAL_HODOLOGY.tex`](EXACT_LOW_DIMENSIONAL_HODOLOGY.tex) or
the compiled [`EXACT_LOW_DIMENSIONAL_HODOLOGY.pdf`](EXACT_LOW_DIMENSIONAL_HODOLOGY.pdf).

For a slower treatment of the projective-orbit retry theorem, including the
left/right invariance convention, its complete Bellman proof, and an entrywise
derivation and Schoenberg analysis of Eq. (43), see
[`THEOREM_7_1_EXPANDED.tex`](THEOREM_7_1_EXPANDED.tex) or the compiled
[`THEOREM_7_1_EXPANDED.pdf`](THEOREM_7_1_EXPANDED.pdf).

Run everything from this directory:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s search -p 'test_*.py' -v
MPLBACKEND=Agg python run_exact_experiments.py
MPLBACKEND=Agg python run_qutrit_phase_experiments.py
MPLBACKEND=Agg python search/search_low_dimensional.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  EXACT_LOW_DIMENSIONAL_HODOLOGY.tex
```

Key files:

- `exact_qubit_lattice.py`: Kraus instruments, exact metrics, qutrit control,
  finite-tolerance search, and dephasing solution;
- `qutrit_phase_lattice.py`: faithful order-11 two-phase qutrit orbit, exact
  Bellman metric, Schoenberg test, and unit-cost retry instruments;
- `run_exact_experiments.py`: deterministic CSV and figure generation;
- `run_qutrit_phase_experiments.py`: qutrit validation tables and figure;
- `tests/`: thirteen algebraic and numerical tests;
- `results/`: compact, reproducible tables and visualization;
- `search/`: complementary numerical candidate search maintained as a separate
  strand of this miniproject.

No claim is made that Hilbert-space distance itself is the learned Euclidean
metric. The exact object is a controlled-action or hodological metric. The
qutrit phase manifold has an isotropic rank-two Fubini--Study tangent metric,
but the inverse-distance retry law is still deliberately supplied. This
distinction is the main conceptual lesson of the construction.

## Successor: a reusable local law

The next step is complete in
[`emergent_local_metric/`](emergent_local_metric/README.md). It replaces the
all-displacements oracle with fixed D4, D8, D16, and D32 local qutrit Kraus
repertoires, trains only symmetry-tied primitive costs, and tests Bellman
geometry on held-out radii. The accompanying theory proves that any fixed
finite heading set has a polygonal stable unit ball. It can approximate but
cannot globally equal a rotationally invariant Euclidean norm.

The same successor also hides the qutrit preparation label and localizes with
weak covariant measurements. Its exact Bayesian quantum filter distinguishes
knowledge of an initial coordinate from controllability of the present state.
This exposes measurement-induced preparation as a false positive for
localization and supplies a concrete base--epistemic-fiber model.

See the integrated [`RESULTS.md`](emergent_local_metric/RESULTS.md), the
pedagogical source
[`EMERGENT_LOCAL_METRIC.tex`](emergent_local_metric/EMERGENT_LOCAL_METRIC.tex),
and compiled paper
[`EMERGENT_LOCAL_METRIC.pdf`](emergent_local_metric/EMERGENT_LOCAL_METRIC.pdf).

## Successor: action meaning from observable consequences

The [informative-actions miniproject](informative_actions/README.md) tests a
stricter criterion: coordinate directions may not be read from action names or
an external displacement counter. They must be identifiable, up to gauge,
from the distributions of later observations that the interventions produce.

The minimal nonunitary qubit experiment recovers two inverse action pairs from
opaque data with 100% success at ten samples per common test, but none of its
33 repeated-path displacement classes is predictively path independent. The
integrated Hesse-qutrit instrument
\(K_o^{(a)}=\Pi_oU_a/\sqrt3\) is stronger: each outcome carries 0.251629 bits,
the last outcome is a sufficient predictive state, and learned kernels recover
the full \(\mathbb Z_3^2\) translation topology. Nevertheless its exact report
cost is 4 both at the target and at a nearest neighbor, so it violates identity
of indiscernibles. The separated move/report benchmark recovers \(6+d_T\), but
its movement outcomes are uninformative. The resulting gap is the current
research frontier, not hidden by a favorable embedding plot.

See the integrated Markdown results and the pedagogical LaTeX paper in
`informative_actions/` for proofs, controls, data, and reproduction commands.
