# Emergent local metric

## Research objective

The exact qutrit phase model proves that a two-dimensional Euclidean
hodological chart is possible, but its displacement-specific retry actions use
the desired norm in advance. This miniproject asks for the next stronger
result:

> Can a small, reusable, translation-covariant family of local qutrit
> interventions induce an effective metric that approaches the intrinsic
> isotropic Fubini--Study metric, including on displacements and directions
> not used to design the controls?

A second requirement removes exact dead reckoning. The initial phase coordinate
is hidden, and the agent must localize from weak nonorthogonal measurements
before or during goal-directed motion.

## Scientific contract

The project separates three claims:

1. **Local control geometry:** a finite heading set induces a polygonal or
   Finsler norm whose discrepancy from Euclidean distance can be measured as
   directional resolution grows.
2. **Physical metric provenance:** action cost should be derived from local
   qutrit control length or energy, not from one inverse-distance success
   probability for every source--target displacement.
3. **Operational localization:** informative weak outcomes should permit
   hidden-start navigation, while matched null outcomes should not.

An embedding is accepted only if competence, Bellman calibration,
Schoenberg spectrum, held-out direction error, locality, translation
covariance, and sensing ablations tell the same story.

## Work strands

- `theory/`: exact limitations of finite heading sets and the convergence
  target for local controls;
- `control/`: deterministic local-control optimization, held-out tests, and
  geometric artifacts;
- `localization/`: hidden-start weak-measurement filtering and navigation;
- this directory: integrated interpretation and reproduction instructions.

The exact macro-action construction in the parent folder remains an oracle
benchmark, not a competitor to be silently mixed into the local action set.

## Outcome

The finite-control baseline and the hidden-start localization study are now
complete.  The central conclusion is a constructive no-go result:

> A fixed finite translation-covariant action catalog can approximate an
> ordinary plane, but its large-scale unit ball is a polygon.  Exact global
> Euclidean geometry requires a circular/elliptical infinitesimal control body,
> obtained from continuous controls, asymptotically dense headings, quadratic
> control energy, or an isotropic diffusion principle.

The deterministic control study fits only displacements of radius at most
four and evaluates through radius twelve.  Euclidean held-out relative RMSE
falls from 30.03% (four directions) to 3.85% (eight) and 1.62% (32).
The 32-direction metric has 2D MDS stress 0.0289, but nonzero Schoenberg
negative eigenmass and directional anisotropy correctly prevent it from being
called exactly Euclidean.  Sixteen nominal directions do no better than eight
because the added knight moves are Bellman-dominated; active vertices of the
velocity hull, not raw action count, control the geometry.

The localization study uses 60,000 seeded episodes on a nine-state,
nonorthogonal qutrit phase orbit.  It separates initial-label navigation from
present-state navigation.  At sharp measurement strength, one observation
has exact preparation-label accuracy (1/3), yet the post-measurement state
can be translated to the target with fidelity one.  Repeated sharp sensing
cannot recover the erased origin.  A Bayesian predictive-state controller
still routes the present state correctly.  This is the first operational
base--fiber split in the miniproject: the translation coordinate, posterior
over origins, and conditioned quantum state are distinct but coupled.

Read the integrated interpretation in [`RESULTS.md`](RESULTS.md), the proofs
in [`theory/THEORY_LOCAL_METRIC.md`](theory/THEORY_LOCAL_METRIC.md), and the
two detailed experiment reports in
[`control/RESULTS_LOCAL_CONTROL.md`](control/RESULTS_LOCAL_CONTROL.md) and
[`localization/RESULTS_LOCALIZATION.md`](localization/RESULTS_LOCALIZATION.md).
The 20-page pedagogical treatment is available as
[`EMERGENT_LOCAL_METRIC.tex`](EMERGENT_LOCAL_METRIC.tex) and the compiled
[`EMERGENT_LOCAL_METRIC.pdf`](EMERGENT_LOCAL_METRIC.pdf).

## Reproduction

From `RL_Experiments`:

```bash
python -m unittest discover \
  -s low_dimensional_hodology/emergent_local_metric/control/tests -v
MPLBACKEND=Agg python \
  low_dimensional_hodology/emergent_local_metric/control/run_local_control.py

python -m unittest discover \
  -s low_dimensional_hodology/emergent_local_metric/localization \
  -p 'test_*.py' -v
MPLBACKEND=Agg python \
  low_dimensional_hodology/emergent_local_metric/localization/localization_experiment.py \
  --episodes 1500 --seed 20260811
```

All generated CSV, JSON, and PNG artifacts stay within the corresponding
strand directory.  The production localization run is deterministic given
the recorded seed; sampling uncertainty is reported separately from the
deterministic control calculations.
