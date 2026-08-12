# Operational localization on a nonorthogonal qutrit phase grid

## Result in one paragraph

A three-level system can support exact local translations on nine nonorthogonal
phase-labelled states, but “localizing the hidden starting label” and “moving
the present quantum state to a target” are different tasks. An exact Bayesian
quantum filter was implemented for a uniformly random hidden preparation, weak
covariant Lüders measurements, and four local unitary moves. Informative
measurements improve hidden-label navigation above the null value \(1/9\), and
known-start navigation is exact. However, a sharp measurement attains only
\(1/3\) initial-label accuracy while preparing a state that can be moved to the
target with unit fidelity. Later sharp outcomes contain no additional
information about the original label. Intermediate-strength measurements can
accumulate more origin information because they do not erase the preparation
in one step. This is a concrete, low-dimensional demonstration that state
preparation can masquerade as spatial localization unless the terminal
criterion is stated operationally.

## Reproduction and scope

From this directory:

```bash
python -m unittest -v test_localization.py
python localization_experiment.py --episodes 1500 --seed 20260811
```

The production bundle contains 60,000 seeded episodes: five measurement
strengths, seven sensing rules plus a known-start control, and 1,500 episodes
per condition. All eight unit tests pass. The code uses only NumPy and
Matplotlib and implements the density-matrix filter directly.

This study deliberately remains small enough to analyze. There is one qutrit,
one nine-outcome sensor family, a 3x3 periodic phase orbit, and a fixed target.
Translation covariance makes the fixed target representative of all nine
goals.

## 1. Physical construction

Let \(\omega=e^{2\pi i/3}\). The nine preparation states are

\[
 |\phi_{xy}\rangle=
 \frac{|0\rangle+\omega^x|1\rangle+\omega^y|2\rangle}{\sqrt3},
 \qquad (x,y)\in\mathbb Z_3^2.
\]

They are not basis states: distinct pairs have fidelity zero or \(1/3\). Their
uniform frame identity is

\[
 \frac13\sum_{x,y}|\phi_{xy}\rangle\langle\phi_{xy}|=I.
\]

Two commuting controls provide exact local translations,

\[
 U=\operatorname{diag}(1,\omega,1),\qquad
 V=\operatorname{diag}(1,1,\omega),
\]

so that \(U|\phi_{xy}\rangle=|\phi_{x+1,y}\rangle\) and
\(V|\phi_{xy}\rangle=|\phi_{x,y+1}\rangle\). The available moves are
\(U,U^\dagger,V,V^\dagger\), each with unit cost. Their shortest-path metric is
the 3x3 toroidal Manhattan metric.

### Weak sensor

For strength \(0\leq\eta\leq1\), outcome \(o\in\mathbb Z_3^2\) has effect

\[
 E_o(\eta)=\frac{\eta}{3}|\phi_o\rangle\langle\phi_o|
             +\frac{1-\eta}{9}I,
\]

and the experiment uses the Lüders Kraus operator

\[
 K_o(\eta)=\sqrt{E_o(\eta)}.
\]

The effects sum to identity. At \(\eta=0\), every effect is \(I/9\): outcomes
are independent uniform noise and the normalized state is unchanged. At
\(\eta=1\), this is the rank-one phase POVM, and outcome \(o\) resets the state
to \(|\phi_o\rangle\). Intermediate instruments trade information against
disturbance without adding any classical hidden register.

## 2. What is hidden, and what is operational?

Each episode samples a preparation label \(S\) uniformly and prepares
\(|\phi_S\rangle\). The label is used only to score whether the agent inferred
the member of this specified ensemble. It is not asserted to be an ontic
coordinate of a mixed state: the average density matrix is \(I/3\), which has
many ensemble decompositions. This limitation is especially important for the
larger foundational motivation.

We therefore report two terminal criteria.

1. **Preparation-label navigation.** If \(\hat S\) is the MAP estimate, apply
   the shortest translation taking \(\hat S\) to goal \(g\). Success means the
   same translation takes the sampled label \(S\) to \(g\), equivalently
   \(\hat S=S\).
2. **Operational state navigation.** Find the phase-grid state with greatest
   fidelity to the current conditioned ensemble density matrix, translate that
   center to \(g\), and score the actual final fidelity
   \(\langle\phi_g|\rho_{\rm final}|\phi_g\rangle\).

Both counterfactual moves are evaluated on the same observation histories.
The first navigates remembered origin; the second navigates present predictive
state. Their difference is a primitive internal/external split.

## 3. Exact Bayesian quantum filter

A purely classical likelihood table would be wrong after the first
measurement, because the sensor disturbs each preparation hypothesis
differently. The filter consequently stores a posterior weight \(w_s\) and a
conditional density matrix \(\rho_s\) for every initial-label hypothesis.

For observed outcome \(o\),

\[
 \ell_s(o)=\operatorname{tr}[E_o\rho_s],\qquad
 p(o)=\sum_s w_s\ell_s(o),
\]

\[
 w'_s=\frac{w_s\ell_s(o)}{p(o)},\qquad
 \rho'_s=\frac{K_o\rho_sK_o^\dagger}{\ell_s(o)}.
\]

The predictive ensemble state is

\[
 \bar\rho=\sum_s w_s\rho_s.
\]

This nine-hypothesis filter is exact for the specified preparation experiment.
It is the transparent analytical benchmark that a learned recurrent predictive
state should later be tested against.

### Sensing and stopping rules

The fixed policies use 0, 1, 3, or 5 measurements. Adaptive policies stop when
\(\max_s w_s\) reaches 0.25, 0.32, or 0.50, with a six-measurement safety cap.
The thresholds deliberately straddle the sharp sensor's maximum confidence of
\(1/3\). All sensing, local moves, and terminal commitment cost one.

Because the prior, state orbit, and sensor are translation covariant, applying
a global phase-grid translation before sensing only permutes outcomes. It has
zero expected information advantage. Thus there is no omitted probe-selection
policy in this symmetric first model; active sensor choice becomes meaningful
only after adding inequivalent instruments or breaking covariance.

## 4. Exact one-shot calculation

The uniform preparation ensemble is

\[
 \bar\rho_0=\frac19\sum_s|\phi_s\rangle\langle\phi_s|=\frac I3,
\]

and every sensor outcome has probability \(p(o)=1/9\). Bayes' rule gives
\(p(s|o)=p(o|s)\), whose maximum occurs at \(s=o\):

\[
 P(\hat S=S)=\frac{1+2\eta}{9}.
\]

The ensemble state conditioned on outcome \(o\) is

\[
 \bar\rho_o=
 \frac{K_o(I/3)K_o}{1/9}
 =3E_o
 =\eta|\phi_o\rangle\langle\phi_o|+(1-\eta)\frac I3.
\]

Moving \(o\) to the target therefore gives exact operational fidelity

\[
 F_g=\frac{1+2\eta}{3}.
\]

The factor-of-three gap between these formulas is the central diagnostic. At
\(\eta=1\), the agent identifies the initial label only one third of the time,
yet the measurement prepares \(|\phi_o\rangle\), which can always be translated
to the target with fidelity one.

For the sharp instrument, the first measurement erases all remaining quantum
dependence on the initial label. Later outcomes are conditionally independent
of \(S\) given the first outcome, so initial-label confidence remains exactly
\(1/3\). The posterior entropy falls from
\(\log_2 9=3.169925\) bits to 2.641604 bits and then stays flat. This saturation
is reproduced in `localization_beliefs.png`.

## 5. Controls

### Null sensor

At \(\eta=0\), \(K_o=I/3\) for every outcome. Posterior weights and branch
states do not change. Any hidden-label policy remains at chance \(1/9\), while
the average target-state fidelity without information is \(1/3\). Extra
sensing is pure cost.

### Known start

When the preparation label is supplied, the agent uses no sensor, follows a
shortest torus path, and reaches both criteria with probability/fidelity one.
For a uniform start and fixed goal, mean movement distance is exactly \(4/3\),
so mean total cost including commitment is \(7/3\). Production estimates vary
slightly around this value because each row contains a finite independent
sample.

These controls jointly show that failure is neither a broken movement model nor
blind homing: it is limited information about nonorthogonal preparations.

## 6. Simulation results

The production estimates below use 1,500 episodes per row. The table displays
binomial standard errors for label success; `localization_summary.csv` also
stores Monte Carlo standard errors for both operational fidelities and for the
predictive-state controller's label success.

| \(\eta\) | sensing rule | label success ± SE | label-policy fidelity | state-policy fidelity | senses | label-policy total cost |
|---:|---|---:|---:|---:|---:|---:|
| 0.0 | no sense | 0.105 ± 0.008 | 0.328 | 0.328 | 0.00 | 1.00 |
| 0.0 | fixed 5 | 0.099 ± 0.008 | 0.325 | 0.325 | 5.00 | 6.00 |
| 0.2 | fixed 1 | 0.147 ± 0.009 | 0.459 | 0.459 | 1.00 | 3.33 |
| 0.2 | fixed 5 | 0.189 ± 0.010 | 0.548 | 0.553 | 5.00 | 7.32 |
| 0.5 | fixed 1 | 0.203 ± 0.010 | 0.659 | 0.659 | 1.00 | 3.34 |
| 0.5 | fixed 5 | 0.271 ± 0.011 | 0.735 | 0.786 | 5.00 | 7.36 |
| 0.5 | adaptive 0.25 | 0.264 ± 0.011 | 0.807 | 0.831 | 4.16 | 6.50 |
| 0.8 | fixed 1 | 0.273 ± 0.012 | 0.870 | 0.870 | 1.00 | 3.32 |
| 0.8 | fixed 5 | 0.302 ± 0.012 | 0.510 | 0.901 | 5.00 | 7.32 |
| 1.0 | fixed 1 | 0.324 ± 0.012 | 1.000 | 1.000 | 1.00 | 3.34 |
| 1.0 | fixed 5 | 0.333 ± 0.012 | 0.356 | 1.000 | 5.00 | 7.31 |
| any | known start | 1.000 | 1.000 | 1.000 | 0.00 | about 2.33 |

The one-shot Monte Carlo values agree with the exact label and operational
formulas to within 0.020 over all five strengths.

### What repeated sensing does

At weak and medium strength, repeated observations reduce posterior entropy and
improve initial-label accuracy. For example, at \(\eta=0.5\), fixed sensing
improves label success from 0.203 after one observation to 0.272 after five.
The improvement is real but expensive: total cost rises from 3.34 to 7.36.

At high strength, origin accuracy rises only modestly while the label-directed
operational score can collapse. At \(\eta=0.8\), five measurements raise label
success from 0.273 to 0.302 but reduce final target fidelity from 0.870 to
0.510. At \(\eta=1\), the origin posterior cannot improve at all; repeated
measurements overwrite the physical state with later outcomes, so moving
according to the original-label MAP estimate is inappropriate for a present
state goal.

The predictive-state controller repairs this mismatch by centering its move on
\(\bar\rho\), not on \(\arg\max_s w_s\). At \(\eta=0.8\) and five observations,
it attains fidelity 0.901 rather than 0.510. At \(\eta=1\), it retains unit
fidelity after five observations, even though its move reaches the original
hidden label only 0.117 of the time. It is a genuine present-state controller,
not a better historical localizer. Hollow triangles in the performance figure
show this counterfactual policy.

### Adaptive sensing

Threshold 0.25 stops after one observation for \(\eta\geq0.8\), correctly
avoiding destructive oversensing. At \(\eta=0.5\) it uses 4.16 observations on
average, retaining label success 0.264 while producing label-policy fidelity
0.807—better operationally and cheaper than fixed five. Threshold 0.50 is
unreachable at sharp strength and therefore always spends the six-step cap;
this is a useful failure control, not a recommended policy.

## 7. What this says about emergent local metric

The qutrit carries an exact 2D **action topology**: two independent local
translations act covariantly on nine nonorthogonal states. A hidden random start
can be partially localized using only quantum observations, and the same moves
navigate to any translated goal. Thus an orthogonal nine-level position register
is not necessary.

But there is no single scalar “position” after measurement. Three coordinates
can diverge:

- the inferred initial phase label \(\arg\max w_s\);
- the best phase-grid approximation to present predictive state \(\bar\rho\);
- the most recent measurement outcome, which may largely specify a newly
  prepared state.

This resembles a small bundle: the base label describes a translation orbit,
while the conditional density matrices and posterior uncertainty form an
epistemic/internal fiber. Measurement transports both and generally couples
them. It would be premature to call this a geometric fiber bundle in the
mathematical sense, but the separation is now operational and measurable.

A local metric should therefore be defined from a specified control problem,
not assigned directly to state labels. Preparation-label loss, present-state
fidelity, intervention cost, and posterior entropy are inequivalent metrics.
Their agreement in a classical orthogonal register is a special limit.

## 8. Limitations

1. The hidden preparation label depends on a chosen ensemble decomposition of
   \(I/3\); it is a laboratory record, not an observable intrinsic property.
2. The space is periodic and homogeneous. There are no open boundaries,
   curvature, or locally varying sensor strengths.
3. The exact Bayesian filter knows the instruments. A learned agent has not yet
   been asked to discover this filter from trajectories.
4. Only one covariant sensor family is available, so active experimental design
   reduces to stopping-time selection.
5. The goal is a phase-grid projector or a preparation-label target, not a long
   internal action-outcome sequence.
6. Monte Carlo estimates are single-seed, though seeded, tested against exact
   one-shot formulas, and supplied with standard errors.

## 9. Recommended next steps

1. **Learn the filter.** Train a recurrent predictive model on the same hidden
   preparation experiment and compare its belief geometry, calibration, and
   navigation policy directly with the exact nine-branch filter.
2. **Optimize strength schedules.** Use dynamic programming over a small grid
   of \(\eta\) values to trade information gain, disturbance, movement, and
   terminal fidelity. The fixed-strength results already show that maximal
   sharpness is not generally optimal.
3. **Add inequivalent probes.** Rotate or coarse-grain the POVM so expected
   information gain depends on the current posterior. This creates genuine
   active localization rather than only an adaptive stopping problem.
4. **Derive information bounds.** Compute accessible information or a Holevo
   bound for the nine-state ensemble and characterize how much initial-label
   information any sequential instrument can retain at fixed terminal fidelity.
5. **Attach internal sequence goals.** Let each base target carry a translated
   multi-step confirmation or task automaton. Compare progress-state geometry
   with the qutrit base to obtain a controlled base--fiber model.
6. **Break homogeneity perturbatively.** Introduce spatially varying weak
   instruments or move noise and ask whether the learned hitting metric
   reconstructs a curved or position-dependent local geometry.

## Artifacts

- `localization_experiment.py`: qutrit construction, weak instruments, exact
  Bayesian quantum filter, navigation controllers, simulation, and plots.
- `test_localization.py`: eight tests covering POVM/Kraus completeness,
  covariance, null sensing, exact formulas, physical filtering, known-start
  navigation, and reproducibility.
- `results/localization_summary.csv`: all aggregate performance metrics.
- `results/posterior_calibration.csv`: confidence versus empirical origin
  accuracy.
- `results/analytic_one_shot.csv`: exact one-shot predictions.
- `results/example_episodes.json`: representative posterior trajectories.
- `results/manifest.json`: production configuration and validation error.
- `figures/localization_performance.png`: localization, cost, and competing
  navigation criteria.
- `figures/localization_beliefs.png`: entropy trajectories and a 2D posterior.
