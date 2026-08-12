# Covariant quantum memory and operational torus geometry

This miniproject asks whether one family of informative quantum instruments
can generate a two-dimensional geometry whose action meanings, topology, and
goal costs are learned from observable consequences. A flat open plane remains
the strongest target, but a translation-covariant two-dimensional torus is a
scientifically meaningful positive result when its periodic topology is
inferred rather than supplied.

## Operational success criteria

A candidate is evaluated in layers:

1. **Information:** present outcomes carry positive information about the
   pre-action predictive state.
2. **Prediction:** histories determine calibrated future-test laws, including
   held-out action/outcome strings.
3. **Action semantics:** opaque actions are identifiable from their effects on
   those laws, up to token and coordinate gauge.
4. **Closure:** words representing the same base displacement agree
   predictively, or any residual disagreement is isolated in an explicitly
   learned internal fiber.
5. **Bellman geometry:** actual state-hitting costs are nonnegative, strictly
   separate distinct states, and satisfy the triangle inequality.
6. **Two-dimensional structure:** action composition reconstructs two
   independent periodic generators; global Euclidean rank is not demanded of
   an intrinsically toroidal metric, but local flat-patch and torus-geodesic
   distortions are reported separately.

The distinction between terminal protocols is essential. “Already in
predictive state \(g\)” is a completed state-hitting goal and has value zero.
“Produce another report \(g\)” is a different event goal and charges an
additional measurement. The former can define a metric even when the latter
has a common reporting overhead.

## Work strands

- `theory/`: covariant higher-rank instrument theory, topology, metric
  conditions, and exact results;
- `search/`: constrained instrument search, exact Bellman solvers, metric and
  embedding diagnostics;
- `learning/`: opaque-history predictive reconstruction, held-out tests, and
  skeptical controls;
- this directory: integrated results and a pedagogical LaTeX paper.

## Headline result

The completed search finds an exactly soluble family with one observed
full-rank memory branch and informative Hesse reset branches.  Under proper
state-hitting semantics its sharp-report Bellman shells are

\[
 E={4\over(1+\mu)^2},\qquad
 D={3\mu+5\over(1+\mu)^2}.
\]

They define a strict metric for every \(0\le\mu\le1\).  At
\(\mu=(4\sqrt2-5)/3\), every elementary torus cell is an exact Euclidean
square while outcomes still carry 0.196535 bits about the predictive state.
The full nine-state object is a learned \(\mathbb Z_3^2\) torus, not a global
planar embedding. Covariance gives same-length/same-displacement closure for
the outcome-discarded channel. Its detour-dependent noise age is a
coarse-grained ensemble variable, not an extra state for the fully observing
agent. A separate Lüders model has oracle-detectable predictive memory, but
learning that candidate fiber remains open.

Start with [`RESULTS.md`](RESULTS.md) for the integrated account, then consult
the strand reports for proofs and experimental details.  The pedagogical paper
is `COVARIANT_MEMORY_GEOMETRY.tex` (and its compiled PDF).

## Reproduction

```bash
python -m unittest discover -s search/tests -v
MPLBACKEND=Agg python search/run_search.py
cd learning
python -m unittest -v test_opaque_learning.py
MPLBACKEND=Agg python opaque_learning.py \
  --train 12000 --test 3000 --seed 20260812
```

All source, tests, artifacts, and documentation for this study remain below
this directory.
