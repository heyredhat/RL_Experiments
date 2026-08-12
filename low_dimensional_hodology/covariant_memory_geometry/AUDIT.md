# Independent scientific and code audit

**Audit date:** 2026-08-12  
**Scope:** `theory/`, `search/`, `learning/`, the integrated Markdown, the
LaTeX manuscript, tests, and generated artifacts.  No existing source, report,
or artifact was changed during this audit.

## Verdict

The central mathematical result survives scrutiny: with state-hitting terminal
semantics, the Hesse instrument has shells `(0,4,5)`, and the search family's
closed forms reproduce the full Bellman solution.  I independently checked all
380 search-grid points; the worst analytic-versus-iterative Bellman discrepancy
was `6.83e-13`, every distance matrix passed the implemented metric tests, and
all fourteen existing unit tests pass.

The miniproject nevertheless needs scientific corrections before publication.
Most importantly, the search strand's claims of opaque learning and a
predictive memory fiber are stronger than its actual observation model
supports: its learner is given latent state/next-state labels, and its path
diagnostic discards observed outcomes and examines only nonselective ensemble
channels.  These do not invalidate the exact Bellman construction, but they
materially narrow the operational interpretation.

## Release-blocking findings

### 1. The manuscript overstates what was evaluated over the full grid

The completed 1,232-line manuscript builds to a PDF and contains the advertised
theory, experiments, figures, limitations, and bibliography.  However,
`COVARIANT_MEMORY_GEOMETRY.tex:678-681` says that **every** one of the 380 grid
pairs was evaluated for Kraus completeness, covariance, iterative Bellman
agreement, Schoenberg/MDS diagnostics, and path closure.  The production grid
loop actually computes only analytic distance, metric/torus diagnostics, and
mutual information (`search/covariant_memory.py:312-335`).  The more expensive
diagnostics and full Bellman solve are run for six named candidates
(`search/run_search.py:63-99`).  Likewise, `COVARIANT_MEMORY_GEOMETRY.tex:579-580`
can be read as claiming saved grid-wide iterative verification.

This audit independently iterated all 380 points and found a worst discrepancy
of `6.83e-13`, so the mathematical claim is true; the checked-in production
protocol and artifacts simply do not establish it as currently described.

**Recommended wording:** “All 380 pairs were screened with the analytic
Bellman formula, metric/torus diagnostics, and mutual information.  Six named
candidates received the full Kraus, covariance, iterative Bellman,
Schoenberg/MDS, and path-closure audit.”  Alternatively save the exhaustive
verification as a separate artifact.  Add a manuscript build test that fails
on unresolved references or citations.

### 2. The search strand's “opaque-action learning” uses privileged latent labels

The physical search instrument observes `memory` or `reset:o`; a memory event
does **not** emit its translated Hesse-state token
(`search/covariant_memory.py:67-74,83-90`).  Nevertheless:

- `infer_opaque_actions` is supplied both the source-state label and the exact
  translated target-state label (`search/covariant_memory.py:340-365`).  It
  increments the target directly after a simulated binomial count of memory
  events; it never infers that target from subsequent observable probes.
- `simulate_heldout_strings` trains on and scores the joint event
  `(outcome,next_state)` with the exact current state provided on every step
  (`search/covariant_memory.py:369-395`).  The next state after `memory` is
  latent under the stated outcome alphabet.  Actions and states are not shuffled
  in this routine despite the “opaque strings” description at `:372`.
- Consequently, the 100% action accuracy and NLL values quoted in
  `search/RESULTS_SEARCH.md:232-239`, `RESULTS.md:225-230`, and the LaTeX
  abstract at `COVARIANT_MEMORY_GEOMETRY.tex:78-80` are oracle-labelled
  finite-state validation, not learning from the agent's observable strings.
- `RESULTS.md:328-338` classifies “opaque recovery” as established exactly,
  whereas `search/RESULTS_SEARCH.md:356-363` correctly puts it in the numerical
  category.  Even numerically, the current search routine does not satisfy the
  operational contract stated in `README.md:14-22`.

Knowing an anchored start and already knowing the action permutation would let
an observer propagate the state after a memory event, but using that propagated
target to *learn the permutation itself* is circular.

**Recommended fix:** either relabel these routines as an “oracle-labelled
finite-state identifiability audit,” or implement an observable protocol.  A
valid protocol can hide the current/next state, apply common future SIC probes,
fit outcome-string likelihoods with a belief/PSR filter, and recover action
maps only up to predictive gauge.  Suggested replacement for
`search/RESULTS_SEARCH.md:232-239`:

> With oracle-labelled Hesse source and successor classes, the finite-state
> transition table recovers all five memory-branch permutations.  This verifies
> structural identifiability conditional on state access; it is not yet opaque
> learning from the instrument's emitted outcome strings.

### 3. The reported “noise-age predictive fiber” is only nonselective ensemble memory

For an anchored Hesse input, every **observed conditional** branch of the search
instrument returns a pure Hesse state, and the observed history determines that
finite class (`search/RESULTS_SEARCH.md:109-114`).  Thus this model's actual
outcome-conditioned predictive state has no additional purity/noise-age
coordinate.

The closure routine instead sums over all outcomes and propagates the
nonselective channel from one fixed initial state
(`search/covariant_memory.py:259-304`).  Its length-dependent mixedness is real,
but it is the state of an observer who suppresses the available branch record.
It is not an internal coordinate needed by the fully observing agent.

This conflicts with the stronger interpretations in:

- `README.md:61-62` (“detours leave a noise-age fiber”);
- `RESULTS.md:17-20,270-294` (“predictive state” equals place plus noise age);
- `search/RESULTS_SEARCH.md:43-48,287-296`; and
- `COVARIANT_MEMORY_GEOMETRY.tex:82-84`.

The theory itself states the correct evidentiary standard at
`theory/THEORY_COVARIANT_MEMORY.md:181-206`: outcome-conditioned closure
requires complete word/outcome kernels, not only nonselective channels.

**Recommended fix:** call the measured quantity “length dependence of the
outcome-discarded ensemble channel.”  Reserve “predictive fiber” for a model in
which the relevant branch information is hidden/coarse-grained, or for the
separate Lüders process after a data-driven belief/PSR representation actually
demonstrates a residual internal state.  Add complete outcome-conditioned word
kernel tests over all starts before claiming path equivalence of histories.

### 4. The external-DFA control described in the learning report is not implemented

`learning/RESULTS_LEARNING.md:53-54` and
`learning/results/manifest.json` describe a nine-node hand-coded external DFA.
In code, `mode=='external'` merely leaves the density matrix unchanged and emits
an iid uniform token (`learning/opaque_learning.py:34-36`).  There is no DFA
state, transition rule, or external counter in `generate` (`:57-67`).

The resulting null observations are a useful duplicate iid control, but they do
not test whether the analysis rejects a genuine external spatial counter whose
quantum stream is null.

**Recommended fix:** either rename the model `iid-external-label-null` and
remove all DFA claims, or implement an explicit hidden nine-state register with
controlled `Z_3^2` transitions while keeping emitted quantum tokens iid.  Then
test separately that external-register geometry is recoverable only when that
register is included in the observation/reward interface.

## Major scientific and terminology findings

### 5. Three different notions of “higher rank” are being conflated

The theory correctly distinguishes Hilbert dimension, branch Choi rank, Kraus
operator rank, predictive rank, and Schoenberg rank
(`COVARIANT_MEMORY_GEOMETRY.tex:155-159`; see also
`theory/THEORY_COVARIANT_MEMORY.md:362-406`).  The reports do not consistently
honor that distinction:

| Strand/family | Minimal branch Choi rank | Kraus-operator rank | Correct description |
|---|---:|---:|---|
| Theory `J_o^lambda`, `0<lambda<1` | 2 | ranks 1 and 3 | genuinely Choi-rank-two retained-memory branch |
| Search `memory` branch, `mu>0` | 1 | 3 | single-Kraus, full-operator-rank unitary branch |
| Search reset, `0<mu<1, xi<1` | 3 | three rank-one Kraus operators | Choi-rank-three measure-and-prepare branch |
| Search reset, selected `xi=1` | 1 | rank-one minimal representation | sharp rank-one measure-and-prepare branch |
| Learning Lüders branch, `eta<1` | 1 | 3 | single-Kraus, full-operator-rank Lüders branch |

I independently computed the minimal Kraus-span ranks.  In particular, the
selected search point `(mu,xi)=(0.8,1)` has Choi rank one for **every observed
branch**, although its memory Kraus operator has matrix rank three.  The
learning models named `higher-rank-luders-*` also use exactly one Kraus operator
per observed branch (`learning/opaque_learning.py:27-29,41-43`).

Problematic wording includes `search/RESULTS_SEARCH.md:19-22,73-103`,
`RESULTS.md:303-305`, and `learning/RESULTS_LEARNING.md:34-46,81-85`.
Also, “for `xi<1` each reset branch has rank three” in
`search/covariant_memory.py:72-73`, `search/RESULTS_SEARCH.md:97`, and
`RESULTS.md:112` needs `mu<1`; at `mu=1` it is the zero map (rank zero).
Likewise the memory operator is not full rank at `mu=0` because it is zero.

**Recommended fix:** replace unqualified “higher-rank” with either
“Choi-rank-two,” “Choi-rank-three,” or “full-operator-rank single-Kraus,” as
appropriate.  Keep the three families explicitly separate in every summary.

### 6. “Exact path closure” is stated more broadly than tested

The implementation tests:

- nonselective channels only;
- one initial Hesse state (`state[0]`);
- the four cardinal nonidentity moves;
- words of length at most four; and
- equality in endpoint trace distance
  (`search/covariant_memory.py:259-309`).

It does not test full outcome-conditioned predictive laws, arbitrary histories
or initial density operators, all future tests, or unbounded words.  Covariance
provides an analytic reason for equal-length **nonselective** equality, but the
headline wording “equal-length paths close exactly” at `README.md:61`,
`RESULTS.md:17`, and `COVARIANT_MEMORY_GEOMETRY.tex:82` omits this restriction.

**Recommended wording:** “For the nonselective channel, covariance proves
same-length/same-displacement closure; the numerical audit confirms it from one
Hesse start through length four.”

### 7. Exact, numerical, and protocol-selection claims need sharper labels

- The Bellman formulas and exact-square ratio are genuinely exact.  The
  380-point grid itself uses only `analytic_distance`; it records
  `bellman_iterations=0` for every grid row
  (`search/covariant_memory.py:312-335`).  Full Bellman iteration is run for six
  reported candidates in `search/run_search.py:63-99`, and the checked-in unit
  test samples three parameter points (`search/tests/test_covariant_memory.py:24-27`).
  The statement at `search/RESULTS_SEARCH.md:182-186` is acceptable for
  “reported candidates,” but readers could easily infer that all 380 points
  were independently iterated.  State explicitly that grid-wide values are
  analytic.  (This audit independently iterated all 380 and found agreement.)
- “Preregistered selection rule” at `RESULTS.md:213-217` and
  `search/RESULTS_SEARCH.md:212-219` has no preregistration artifact or dated
  protocol.  A generated manifest containing the threshold is not prior
  registration.  Use “deterministic stated selection rule” unless an immutable
  pre-analysis record exists.
- `RESULTS.md:328-338` lists opaque recovery and equal-length closure as
  established “exactly,” although both are reported from finite simulations in
  the search report.  Analytic group covariance is exact; finite-data recovery
  is numerical; the measured closure is the restricted nonselective test above.
- The four-state square really is an exact Euclidean distance matrix at
  `mu_square`, but it is a metric subspace of a `3x3` torus, not a nonwrapping
  open chart.  The qualifications at `RESULTS.md:246-249,264-268` are good and
  should accompany every headline use of “local square.”

### 8. The two learning results should not be merged into one integrated learner

The search model and the independent learning model are different instruments:

- search: ten observed branches (`memory` plus nine resets), finite conditional
  Hesse-state dynamics;
- learning: nine single-Kraus Lüders outcomes, continuous retained posterior
  dynamics; and
- theory: a third, Choi-rank-two additive branch family.

`RESULTS.md:296-313` mostly acknowledges the independent learning strand, but
the executive list at `RESULTS.md:5-14` and the LaTeX abstract
`COVARIANT_MEMORY_GEOMETRY.tex:65-86` can read as though opaque recovery,
finite-state Bellman geometry, and learned Lüders memory were demonstrated for
one common family.  They were not.

The learning report is commendably explicit that the Lüders Bellman values are
token-aggregated approximations (`learning/RESULTS_LEARNING.md:104-110,131-139`).
That qualification should appear in the integrated executive conclusion too.

## Artifact and reproducibility audit

### Search artifacts

All files named by `search/results/summary.json` exist, and the row counts are
consistent: 380 grid candidates, six reported candidates, and six Bellman
matrices.  Numeric headline values agree with the CSV/JSON artifacts.

The summary is not a complete manifest:

- `bellman_*.csv` is a glob rather than six explicit paths;
- code/config hashes, NumPy/Matplotlib versions, timestamps, and the exact grid
  vectors are absent;
- `maximum_kraus_residual` and `maximum_covariance_residual` at
  `search/run_search.py:138-139` are maxima over the six reported candidates,
  not the 380-point grid; and
- the “covariance residual” is a finite Hesse transition-kernel covariance test,
  explicitly documented at `search/covariant_memory.py:233-247`, not a Choi- or
  arbitrary-state covariance residual.  The analytic CP map is covariant, but
  the artifact should name what was numerically tested.

Rename those fields `maximum_reported_candidate_*` and
`hesse_branch_kernel_covariance_residual`, and list every artifact explicitly.

### Learning artifacts

All expected CSV and PNG outputs are present, and the report's NLL, Bellman,
group-order, and sample-count values agree with `learning_summary.csv`.
However, `learning/results/manifest.json` has no artifact list and omits almost
all headline results, model parameters, action/token shuffles, dependency
versions, and hashes.  Its external-DFA description is also factually
inconsistent with the implementation, as noted above.

Add a complete model/config/result/artifact manifest and a central manifest that
maps each headline claim to its producing code and artifact.

## Test audit

Commands run:

```text
python -m unittest discover -s search/tests -v       7/7 pass
python -m unittest -v learning/test_opaque_learning.py  7/7 pass
```

The tests are fast and valuable, but report coverage more broadly than they
provide:

- `learning/RESULTS_LEARNING.md:69-71` says both terminal semantics are tested.
  `learning/test_opaque_learning.py:11-12` tests only state hitting; `(4,4,5)` is
  hard-coded into the manifest at `learning/opaque_learning.py:183-185`, with no
  report-again solver or assertion.
- No test exercises `infer_opaque_actions`, `simulate_heldout_strings`,
  production-run determinism, artifact schemas, model controls, NLL claims, or
  the external-DFA behavior.
- The search suite tests analytic Bellman agreement at only three points and
  strict shells at one sharpness (`search/tests/test_covariant_memory.py:24-32`).
- The covariance test is only the finite Hesse kernel test described above.
- The path test does not assert outcome-conditioned closure or invariance across
  all starts.
- No test compiles the LaTeX manuscript, which would have caught the fatal EOF.

Recommended additions, in priority order:

1. manuscript build and undefined-reference test;
2. observable-only action-learning test that fails if latent states enter the
   learner API;
3. explicit report-again Bellman solver/test alongside state hitting;
4. branch Choi-rank and Kraus-operator-rank tests at endpoints and selected
   parameters;
5. complete outcome-conditioned path-kernel tests over all starts;
6. genuine external-DFA control test;
7. all-grid analytic-versus-Bellman regression test (or a documented sampled
   test plus a saved exhaustive verification artifact); and
8. deterministic production-manifest/artifact validation.

## Claims independently confirmed

The following claims are supported as written or with only the qualifications
already present in their detailed sections:

1. The terminal distinction is scientifically and mathematically correct:
   state hitting sets `V_g(g)=0`, whereas report-again charges another action.
2. The rank-one Hesse state-hitting Bellman shells are exactly `(0,4,5)` and
   form a strict translation-invariant torus metric.
3. The theory's Schoenberg spectrum is correct: four eigenvalues `17`, four
   eigenvalues `7/2`, and one zero, hence minimal Euclidean dimension eight.
4. The theory family `J_o^lambda` is CPTP and covariant; for
   `0<lambda<1` its observed branches have Choi rank two, and the stated Hesse
   likelihood formula is normalized and consistent.
5. The search family's sharp-report formulas
   `E=4/(1+mu)^2` and `D=(3mu+5)/(1+mu)^2` match full Bellman iteration.
6. The selected `(0.8,1)` search point has the reported MI, shell values,
   metric validity, and torus distortion.
7. `mu=(4*sqrt(2)-5)/3` gives `D/E=sqrt(2)` and an exact four-point Euclidean
   square metric, while the full object remains toroidal.
8. The learning strand really uses only opaque action/outcome strings for its
   suffix estimators (`learning/opaque_learning.py:57-93`); its oracle density
   matrices are used for evaluation, not supplied to those count estimators.
9. The learning CSV supports its honest negative conclusion: the finite suffix
   learner does not exploit the Lüders memory, while the quantum-filter oracle
   achieves lower held-out NLL.

## Bottom line

The exact state-hitting torus metric is a solid result and should remain the
mathematical core.  The strongest defensible current empirical conclusion is:

> An engineered, group-covariant qutrit instrument yields an exact finite
> state-hitting torus metric, and a separate opaque-outcome Lüders experiment
> recovers an approximate token-level translation quotient while detecting—but
> not learning—additional predictive memory.

Claims that the selected search instrument itself learns opaque action meaning
from emitted strings, or that its outcome-conditioned predictive state is a
base-plus-noise-age fiber, require new experiments rather than wording alone.
