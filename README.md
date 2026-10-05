
---

# 2. Survival analysis — `README.md`

This one I'd make a little more mathematically substantial because **this is actually a stronger portfolio piece for you**.

```markdown
# Survival Analysis from First Principles

A statistical survival-analysis project using clinical data to investigate how patient characteristics relate to time-to-event outcomes.

The project develops a Cox proportional hazards model from first principles, including the Efron partial likelihood, Newton-Raphson optimisation and estimation of the baseline cumulative hazard.

The ultimate goal is to produce patient-specific survival distributions rather than simply predicting a single survival time.

## Project objectives

The project aims to:

- Understand the statistical structure of censored time-to-event data.
- Explore survival distributions using Kaplan-Meier estimation.
- Implement the Cox proportional hazards model from first principles.
- Derive and implement the Efron partial likelihood.
- Estimate model parameters using Newton-Raphson optimisation.
- Estimate the baseline cumulative hazard.
- Construct patient-specific survival curves.
- Validate the implementation against established survival-analysis software.
- Evaluate predictive performance and model assumptions.

## Data

The project uses the `lung` dataset from the R `survival` package.

The dataset contains 228 patients with advanced lung cancer.

The main variables include:

- Survival time
- Censoring/event indicator
- Age
- Sex
- ECOG performance score
- Karnofsky performance scores
- Weight loss
- Caloric intake
- Institution

The time variable measures the number of days from study entry.

Importantly, the event indicator distinguishes between patients whose death was observed and patients whose observation ended before death was observed.

## Why survival analysis?

Ordinary regression is not well suited to this problem because many patients are **right-censored**.

For a censored patient, we know that they survived at least until their censoring time, but we do not know their eventual survival time.

For each individual the observed data can therefore be represented as

\[
(T_i,\delta_i),
\]

where:

- \(T_i\) is the observed time
- \(\delta_i=1\) if the event is observed
- \(\delta_i=0\) if the observation is censored

Survival analysis explicitly incorporates this partial information rather than treating censored observations as ordinary regression targets.

## Survival and hazard functions

The survival function is

\[
S(t)=P(T>t),
\]

the probability of surviving beyond time \(t\).

The hazard function is the instantaneous event rate conditional on having survived to time \(t\):

\[
h(t)
=
\lim_{\Delta t\rightarrow0}
\frac{
P(t\leq T<t+\Delta t\mid T\geq t)
}{\Delta t}.
\]

The two quantities are related by

\[
h(t)=-\frac{S'(t)}{S(t)}
\]

and therefore

\[
S(t)
=
\exp\left(-\int_0^t h(u)\,du\right).
\]

Understanding this relationship is central to the project.

## Kaplan-Meier estimation

The first stage of the analysis is non-parametric estimation of the population survival function using the Kaplan-Meier estimator.

This provides a baseline description of survival while correctly accounting for censoring.

Kaplan-Meier curves are also used to investigate differences between groups of patients.

## Cox proportional hazards model

The main model is the Cox proportional hazards model:

\[
h(t\mid X)
=
h_0(t)\exp(X^\top\beta),
\]

where:

- \(h_0(t)\) is the baseline hazard
- \(X\) is the vector of patient characteristics
- \(\beta\) contains the regression coefficients

The model is semi-parametric: the covariate effects are parametrically modelled, while the baseline hazard is left unspecified.

A coefficient can be interpreted through its hazard ratio:

\[
HR_k=e^{\beta_k}.
\]

## Partial likelihood

The Cox model can estimate the regression coefficients without specifying the baseline hazard.

At an observed death time \(t_j\), let \(R_j\) denote the risk set: the individuals who are known to be alive and still under observation immediately before \(t_j\).

Conditional on a death occurring among this risk set, the probability that individual \(i\) is the one who dies is

\[
\frac{\exp(X_i^\top\beta)}
{\sum_{k\in R_j}\exp(X_k^\top\beta)}.
\]

The baseline hazard cancels from this conditional probability.

Multiplying these terms over observed death times gives the Cox partial likelihood.

## Handling ties with Efron's method

Clinical data frequently contain multiple deaths recorded at the same time.

Rather than arbitrarily ordering tied events, the implementation uses **Efron's approximation** to the exact partial likelihood.

For an event time \(t_j\), define:

- \(H_j\): the set of individuals who experience the event
- \(R_j\): the risk set immediately before the event
- \(m_j=|H_j|\): the number of tied events

and

\[
\theta_i=e^{X_i^\top\beta}.
\]

The Efron denominator for event fraction \(r\) is

\[
B_{jr}
=
\sum_{i\in R_j}\theta_i
-
\frac{r}{m_j}
\sum_{i\in H_j}\theta_i.
\]

The log partial likelihood is then

\[
\ell(\beta)
=
\sum_j
\left[
\sum_{i\in H_j}X_i^\top\beta
-
\sum_{r=0}^{m_j-1}\log B_{jr}
\right].
\]

The gradient and Hessian are derived analytically and used for Newton-Raphson optimisation.

## Newton-Raphson optimisation

The parameter estimate is obtained iteratively using

\[
\beta^{(m+1)}
=
\beta^{(m)}
-
H(\beta^{(m)})^{-1}
U(\beta^{(m)}),
\]

where \(U(\beta)\) is the score vector and \(H(\beta)\) is the Hessian of the log partial likelihood.

In implementation, the linear system is solved directly rather than explicitly calculating the matrix inverse.

## Estimating the baseline cumulative hazard

The partial likelihood estimates the **relative hazard coefficients** but does not directly determine the absolute baseline hazard.

Once \(\hat\beta\) has been obtained, the baseline cumulative hazard can be estimated from the observed number of deaths.

For the Breslow estimator,

\[
\Delta\hat H_0(t_j)
=
\frac{d_j}
{\sum_{i\in R_j}
\exp(X_i^\top\hat\beta)},
\]

where \(d_j\) is the number of observed deaths at \(t_j\).

The intuition is to equate the model's expected number of deaths with the number actually observed and solve for the required baseline hazard increment.

The cumulative baseline hazard is then

\[
\hat H_0(t)
=
\sum_{t_j\leq t}
\Delta\hat H_0(t_j).
\]

## Patient-specific survival

Once the baseline cumulative hazard and regression coefficients have been estimated, a survival curve can be constructed for a patient with covariates \(X\):

\[
\hat S(t\mid X)
=
\exp\left[
-\hat H_0(t)e^{X^\top\hat\beta}
\right].
\]

Equivalently,

\[
\hat S(t\mid X)
=
\hat S_0(t)^{e^{X^\top\hat\beta}},
\]

where

\[
\hat S_0(t)=e^{-\hat H_0(t)}.
\]

This allows the model to produce an entire estimated survival distribution for an individual rather than a single predicted survival time.

## Validation

The implementation is validated at several levels.

### Mathematical validation

- Numerical differentiation of the log partial likelihood is compared with the analytical score.
- Numerical differentiation of the score is compared with the analytical Hessian.
- Newton-Raphson convergence is checked.

### Implementation validation

The estimated Cox coefficients are compared with an established implementation such as `lifelines`.

Agreement provides evidence that the from-scratch implementation is correctly reproducing the Cox model.

### Predictive validation

For predictive evaluation, cross-validation is used rather than relying only on in-sample performance.

Potential metrics include:

- Concordance index
- Brier score
- Integrated Brier score
- Survival calibration

The concordance index evaluates whether the model correctly ranks patients by risk, while Brier-based metrics assess the accuracy of predicted survival probabilities.

## Important assumptions

The analysis considers several assumptions and limitations.

### Proportional hazards

The Cox model assumes that hazard ratios between individuals are constant over time.

This assumption can be investigated using standard diagnostic methods.

### Censoring

The basic analysis assumes censoring is sufficiently non-informative conditional on the modelled covariates.

If patients with systematically different unobserved survival prospects are more likely to be censored, estimates may be biased.

### Time origin

The survival time in this dataset is measured from study entry rather than from the biological onset of cancer.

Consequently, the interpretation is conditional on patients having entered the study and should not be interpreted as modelling survival from the onset of cancer.

## Project structure

```text
lung-survival-analysis/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_kaplan_meier.ipynb
│   └── 03_cox_model.ipynb
├── src/
│   ├── likelihood.py
│   ├── optimisation.py
│   ├── baseline_hazard.py
│   └── survival.py
├── README.md
└── requirements.txt
