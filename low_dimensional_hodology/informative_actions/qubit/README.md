# Informative qubit actions

This miniproject asks whether an agent can infer the meaning of opaque qubit
actions solely from their outcomes and common future probes, then tests whether
those meanings compose into a path-independent two-dimensional goal space.

The first answer is yes. The second, for the selected genuinely nonunitary
instrument, is no. That distinction is the main result; see
[`RESULTS_QUBIT.md`](RESULTS_QUBIT.md).

Run the deterministic search, simulations, and tests from this directory:

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_qubit_experiment.py
```

The learner-facing reconstruction receives opaque button names and observed
frequencies only. Hidden coordinates are used solely to score recovery modulo
the unavoidable square-symmetry gauge.
