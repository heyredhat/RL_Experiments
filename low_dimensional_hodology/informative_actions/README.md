# Informative actions and predictive geometry

This miniproject asks a stricter question than whether a quantum controller can
be assigned spatial action labels:

> Can an agent learn what opaque actions mean from their outcomes and effects
> on future observable statistics, and do those learned meanings compose into
> a nondegenerate two-dimensional goal geometry?

The answer is now sharply divided. Several minimal constructions solve
different parts of the problem, but none yet satisfies the whole scientific
contract:

- an exact qubit Pauli benchmark has four nonorthogonal predictive places and
  an exact Euclidean square, but its movement actions have no immediate
  outcomes;
- a genuinely nonunitary qubit instrument yields a robust learned two-axis
  action chart, but same-displacement paths are not predictively equivalent;
- integrated Hesse-SIC qutrit actions recover an exact opaque
  \(\mathbb Z_3^2\) translation topology and carry 0.251629 bits of immediate
  information, but their Bellman values satisfy self = edge rather than
  self < edge < diagonal;
- separating qutrit movement and reporting restores the exact toroidal word
  metric, \(V_g(s)=6+d_T(s,g)\), but the moves are uninformative unitaries.

The synthesis, comparison table, operational criteria, and proposed next
experiment are in [`RESULTS.md`](RESULTS.md).

The self-contained pedagogical treatment, including the complete proofs,
reinforcement-learning background, exact Bellman obstruction, figures,
controls, and future program, is available as
[`INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex`](INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex)
and the compiled
[`INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.pdf`](INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.pdf).

## Scientific contract

A successful operational spatial atlas must pass all six finite criteria from
the theory note:

1. **P — predictive separation:** candidate places have different future
   observable laws;
2. **A — action identifiability:** opaque actions induce distinguishable
   transformations on the predictive quotient;
3. **B — Bellman realizability:** claimed distances are actual optimal hitting
   costs;
4. **M — metric validity:** symmetry, positivity, identity of
   indiscernibles, and the triangle inequality hold;
5. **E — Euclidean planarity:** the Schoenberg Gram matrix is positive
   semidefinite of rank two;
6. **L — learned locality:** reconstructed action increments are local,
   homogeneous, and derived from response transformations rather than labels.

For informative stochastic actions we additionally require positive immediate
or delayed action information, predictive path closure, held-out sequence
prediction, and disappearance of geometry under matched null controls.

## Directory map

- [`theory/THEORY_INFORMATIVE_ACTIONS.md`](theory/THEORY_INFORMATIVE_ACTIONS.md):
  predictive equivalence, finite Hankel closure, gauge, information measures,
  minimality results, and the P/A/B/M/E/L theorem;
- [`qubit/RESULTS_QUBIT.md`](qubit/RESULTS_QUBIT.md): nonunitary opaque-button
  chart reconstruction, predictive tomography, controls, and the path-closure
  no-go;
- [`qutrit/RESULTS_QUTRIT.md`](qutrit/RESULTS_QUTRIT.md): integrated Hesse-SIC
  construction, opaque group recovery, exact Bellman analysis, prediction,
  navigation, and controls;
- [`RESULTS.md`](RESULTS.md): cross-strand conclusions and research program.
- [`INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex`](INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex):
  publication-style, first-principles derivation and synthesis.

## Reproduction

Qubit strand:

```bash
cd qubit
python -m unittest discover -s tests -v
MPLBACKEND=Agg python run_qubit_experiment.py
```

Qutrit strand:

```bash
cd qutrit
python -m unittest -v test_informative_qutrit.py
MPLBACKEND=Agg python informative_qutrit.py --episodes 2000 --seed 20260812
```

Both stochastic bundles use seed 20260812. Action and outcome names are opaque
to the learner; hidden coordinate alignment is reserved for frozen audits.

Rebuild the paper with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  INFORMATIVE_ACTIONS_AND_PREDICTIVE_GEOMETRY.tex
```
