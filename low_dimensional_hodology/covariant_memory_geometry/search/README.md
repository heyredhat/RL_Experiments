# Covariant qutrit memory-instrument search

This self-contained miniproject searches integrated qutrit instruments whose
observed outcomes provide information while a full-operator-rank branch preserves and
translates quantum memory. It uses corrected state-hitting terminal semantics:
the value is zero when the predictive state is already the goal.

See [`RESULTS_SEARCH.md`](RESULTS_SEARCH.md) for the construction, exact
Bellman solution, numerical search, controls, and limitations.

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_search.py
```

All generated data are deterministic. The operational action-recovery audit
uses only opaque anchor, action, observed-memory, and common-future-probe
tokens. Translated states and coordinate labels are withheld from the learner;
they are used only for offline gauge-aware scoring. A separate latent-state
transition benchmark is retained and explicitly labelled as oracle-based.
