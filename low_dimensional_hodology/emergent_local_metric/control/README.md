# Local qutrit metric optimization

This experiment asks whether ordinary distance can extrapolate from one small,
translation-covariant repertoire of local qutrit Kraus instruments. It forbids
the displacement-specific direct actions used in the exact finite-patch
construction. The same 4, 8, 16, or 32 directions are reused at every point and
for every goal.

See [`RESULTS_LOCAL_CONTROL.md`](RESULTS_LOCAL_CONTROL.md) for the theory,
results, and limitations.

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_local_control.py
```

The implementation uses NumPy and the standard library; Matplotlib is required
only for the summary figure. The optimizer and all artifact generation are
deterministic.
