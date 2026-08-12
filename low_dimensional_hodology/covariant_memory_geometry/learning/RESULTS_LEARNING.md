# Opaque predictive learning with covariant qutrit memory

This study asks whether an agent can recover qutrit translation structure from
action/outcome strings alone. The estimator receives opaque action IDs and
opaque outcome tokens. It never receives coordinates, density matrices, Kraus
operators, or displacement names.

The transparent learner estimates controlled history--test tables at three
resolutions: action-only, last-token plus next action, and a two-event suffix
plus next action. It also learns token transition kernels, extracts their
maximum-probability permutations, computes the generated action group, and
solves state-hitting Bellman equations from the learned model.

## Positive exact baseline

For rank-one Hesse measure-and-prepare actions, the last outcome is a sufficient
predictive state. Five opaque actions induce one identity and four commuting
order-three permutations. Their group has order nine and acts transitively,
recovering the topology of \(\mathbb Z_3^2\) up to arbitrary token, origin,
axis, and orientation gauge.

Terminal semantics matters. If the goal is achieved when the **current
predictive token** equals \(g\), the exact Bellman shells are

\[
V_{\rm self}=0,\qquad V_{\rm edge}=4,\qquad V_{\rm diagonal}=5.
\]

This satisfies strict shell ordering and identity of indiscernibles. If the
goal instead requires reporting \(g\) again, the shells are \(4,4,5\), which
is nonmetric. The positive result therefore concerns a directly observable
state-hitting goal, not the report-again goal.

## Higher-rank branch memory

The provisional candidate uses effects

\[
E_o=\eta\Pi_o/3+(1-\eta)I/9
\]

and Lüders branches \(\sqrt{E_o}U_a\). Unlike rank-one preparation, the same
outcome token does not erase the previous state. Histories ending in the same
token can therefore have different controlled future laws. The two-event
history table and the exact quantum filter are evaluated on held-out strings to
measure this retained memory rather than assuming last-token Markovization.

## Controls

- Null Lüders outcomes are uniform and carry no immediate information.
- Haar-random controls can have learnable future effects without closing into
  a commuting translation group.
- A genuine nine-state external register updates by exact
  \(\mathbb Z_3^2\) transitions while emitting i.i.d. uniform quantum tokens.
  Token-only learning rejects it; an explicitly register-exposed interface
  recovers its hand-coded geometry.
- Rank-one measure-and-prepare is the memoryless positive topology baseline.
- Independent action and token reshuffling changes only gauge; group order and
  Cayley topology remain invariant.

## Reproduction

```bash
python -m unittest -v test_opaque_learning.py
MPLBACKEND=Agg python opaque_learning.py --train 12000 --test 3000 --seed 20260812
```

Generated artifacts include `learning_summary.csv`, `action_algebra.csv`,
per-model learned kernels, `manifest.json`, and two diagnostic figures. The
production run uses six models, 12,000 training and 3,000 held-out seven-step
sequences per model. Ten unit tests cover SIC completeness, exact kernels,
group recovery, both terminal semantics, retained branch memory, null sensing,
and token-gauge invariance.

## Interpretation

Two independent tests are essential. Action algebra tests whether controlled
future transformations form a spatial topology. History-test prediction tests
whether the chosen predictive state representation is sufficient. A model can
pass either without passing the other: random controls can be predictive but
nonspatial, while rank-one covariant controls are spatial but memoryless.

The higher-rank family is the scientifically interesting intermediate case. If
two-event histories improve held-out likelihood for contexts sharing the same
last token, branch memory is operationally real. If its learned token
permutations still approximately generate \(\mathbb Z_3^2\), the predictive
state has a discrete spatial base plus a within-token memory fiber. Exact
closure is not assumed; residual prediction error and path closure quantify
the deformation.

The construction remains finite, periodic, and engineered. Its torus graph is
two-dimensional by independent translation generators, but its word metric is
not an exact Euclidean planar distance matrix. A future spectral PSR should
replace suffix tables, infer predictive dimension from a controlled Hankel
singular spectrum, and feed belief-state rather than token-state Bellman
planning.

## Production findings

The rank-one baseline is learned cleanly. All five maximum-likelihood maps are
bijective and commute; they generate a transitive group of order nine. The
learned state-hitting shells are approximately \(0,3.96,4.92\), close to the
exact \(0,4,5\). Token-graph 2D MDS stress is about 0.377, consistent with a
two-generator torus rather than a globally Euclidean plane.

Both higher-rank Weyl-covariant instruments also recover bijective, commuting
order-nine action groups from opaque transition counts. Their learned token
Bellman shells are strictly ordered: approximately \(0,6.63,7.09\) at
\(\eta=0.55\), and \(0,5.11,5.91\) at \(\eta=0.80\). Retained-memory
instruments therefore preserve recoverable torus topology and strict
state-hitting ordering, although token-aggregated Bellman planning is only an
approximation to the full predictive process.

Held-out prediction supplies the key qualification. At \(\eta=0.55\),
last-token NLL is about 3.149 bits/outcome while the exact quantum filter reaches
3.114; at \(\eta=0.80\), the corresponding values are 3.074 and 3.017. The last
token is not sufficient. A regularized finite-suffix estimator did not exploit
this memory at the available sample size: its NLL was slightly worse. The
oracle gap establishes latent predictive memory, while the failed suffix model
shows that naive context proliferation is not an adequate PSR learner.

Raw differences among empirical future laws for histories sharing a last token
are comparable with the null sampling floor. Both raw and null-subtracted
values are recorded, preventing sampling variation from being mislabeled as a
learned memory fiber. Controlled Hankel singular spectra are saved for every
model, but cross-validated spectral shrinkage remains future work.

Null, external-DFA, and Haar controls produce nonbijective argmax maps. They are
marked as invalid action sets with group order zero, rather than allowing a
large transformation monoid to be misreported as a group. Prediction,
translation topology, and external goal memory are therefore separate tests.

The external control is implemented, not merely described. Its hidden register
has states \((x,y)\in\mathbb Z_3^2\) and deterministic identity/axial
translations, while the accompanying quantum token stays i.i.d. uniform. On
the scientifically relevant token-only interface, argmax maps are nonbijective
and group order is zero. If the register is explicitly added to observations,
the learner recovers a commuting group of order nine with orbit nine.
`external_register_interfaces.csv` records both interfaces and marks the latter
geometry as originating in an explicit classical DFA/register, not quantum
predictive statistics.

## Limitations and next step

This study learns the token-level controlled quotient, not complete
higher-rank belief dynamics. For rank-one preparation that quotient is exact.
For Lüders instruments it is a useful spatial base but discards branch memory.
The next step should be a controlled spectral PSR using longer history/test
Hankel blocks, held-out singular-value shrinkage, and a learned low-rank linear
update. Bellman planning should then operate in predictive state, with the
quantum filter used only as a calibration oracle.

The state-hitting convention must remain explicit. It gives valid
\(0<4<5\) ordering because a current goal token terminates immediately.
Requiring another report yields nonmetric \(4,4,5\). These are different goals,
not interchangeable evaluations.
