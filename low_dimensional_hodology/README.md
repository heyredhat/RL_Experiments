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

Run everything from this directory:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s search -p 'test_*.py' -v
MPLBACKEND=Agg python run_exact_experiments.py
MPLBACKEND=Agg python run_qutrit_phase_experiments.py
MPLBACKEND=Agg python search/search_low_dimensional.py
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
