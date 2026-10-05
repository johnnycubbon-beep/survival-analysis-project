"""Fit the lung cancer Cox model two ways and compare coefficient estimates.

Run from the project root with ``python src/fit_and_compare.py`` after
installing lifelines (``python -m pip install lifelines``).

The custom fit uses the Efron partial log-likelihood, gradient, and Hessian
implemented below. Review those derivative expressions against your derivation
before treating the custom optimizer's result as validated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from likelihood_function import DEFAULT_COVARIATES, efron_log_likelihood


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "lung_cancer.xls"


def _model_arrays(beta, data, covariate_columns=DEFAULT_COVARIATES):
    """Return validated time, event, and covariate arrays in model order."""
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if beta.size != len(covariate_columns):
        raise ValueError(
            f"Expected {len(covariate_columns)} coefficients; got {beta.size}"
        )

    # Reuse the public likelihood's input checks.
    efron_log_likelihood(np.zeros_like(beta), data, covariate_columns)
    times = data["TIME"].to_numpy(dtype=float)
    events = data["Y"].to_numpy(dtype=int)
    x = data.loc[:, covariate_columns].to_numpy(dtype=float)
    eta = x @ beta
    return times, events, x, eta


def efron_gradient(beta, data, covariate_columns=DEFAULT_COVARIATES):
    """Gradient of the Efron partial log-likelihood with respect to beta."""
    times, events, x, eta = _model_arrays(beta, data, covariate_columns)
    gradient = x[events == 1].sum(axis=0)

    for time in np.unique(times[events == 1]):
        deaths = (times == time) & (events == 1)
        risk = times >= time
        tie_count = int(deaths.sum())
        shift = np.max(eta[risk])
        exp_eta = np.exp(eta - shift)
        risk_weights = exp_eta * risk
        death_weights = exp_eta * deaths
        risk_sum = risk_weights.sum()
        death_sum = death_weights.sum()

        for r in range(tie_count):
            q = r / tie_count
            weights = risk_weights - q * death_weights
            denominator = risk_sum - q * death_sum
            gradient -= weights @ x / denominator

    return gradient


# Fairly sure this function matches my handwritten Hessian. Probably worth checking at some point.
def efron_hessian(beta, data, covariate_columns=DEFAULT_COVARIATES):
    """Hessian matrix of the Efron partial log-likelihood."""
    times, events, x, eta = _model_arrays(beta, data, covariate_columns)
    hessian = np.zeros((x.shape[1], x.shape[1]), dtype=float)

    for time in np.unique(times[events == 1]):
        deaths = (times == time) & (events == 1)
        risk = times >= time
        tie_count = int(deaths.sum())
        shift = np.max(eta[risk])
        exp_eta = np.exp(eta - shift)
        risk_weights = exp_eta * risk
        death_weights = exp_eta * deaths
        risk_sum = risk_weights.sum()
        death_sum = death_weights.sum()

        for r in range(tie_count):
            q = r / tie_count
            weights = risk_weights - q * death_weights
            denominator = risk_sum - q * death_sum
            weighted_x = weights @ x
            weighted_xx = x.T @ (weights[:, None] * x)
            # The Hessian contribution is minus the weighted covariance.
            hessian -= weighted_xx / denominator - np.outer(
                weighted_x, weighted_x
            ) / denominator**2

    return hessian


def fit_custom(data, covariate_columns=DEFAULT_COVARIATES):
    """Maximize the custom Efron log-likelihood using Newton trust-region."""
    initial_beta = np.zeros(len(covariate_columns), dtype=float)
    result = minimize(
        fun=lambda beta: -efron_log_likelihood(beta, data, covariate_columns),
        x0=initial_beta,
        jac=lambda beta: -efron_gradient(beta, data, covariate_columns),
        hess=lambda beta: -efron_hessian(beta, data, covariate_columns),
        method="trust-exact",
        options={"gtol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Custom Cox fit did not converge: {result.message}")
    return result


def main():
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise SystemExit(
            "This comparison requires lifelines. Install it with: "
            "python -m pip install lifelines"
        ) from exc

    raw = pd.read_csv(DATA_PATH)
    required = ["ID", "TIME", "Y", *DEFAULT_COVARIATES]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # This file stores each patient twice: a TIME=0 entry and a final follow-up
    # entry. Use the final row per ID to construct one duration/event per person.
    data = raw.sort_values(["ID", "TIME"]).groupby("ID", as_index=False).tail(1)
    data = data.loc[:, ["ID", "TIME", "Y", *DEFAULT_COVARIATES]].reset_index(drop=True)

    custom = fit_custom(data)

    library_data = data.drop(columns="ID").copy()
    library_model = CoxPHFitter(penalizer=0.0)
    library_model.fit(library_data, duration_col="TIME", event_col="Y")

    comparison = pd.DataFrame(
        {
            "custom_efron": custom.x,
            "lifelines_efron": library_model.params_.loc[list(DEFAULT_COVARIATES)].to_numpy(),
        },
        index=DEFAULT_COVARIATES,
    )
    comparison["difference"] = comparison["custom_efron"] - comparison["lifelines_efron"]

    print(f"Patients: {len(data)}")
    print(f"Observed deaths: {int(data['Y'].sum())}")
    print(f"Custom optimizer: {custom.message} ({custom.nit} iterations)")
    print(f"Custom Efron log-likelihood: {-custom.fun:.10f}")
    print(f"lifelines Efron log-likelihood: {library_model.log_likelihood_:.10f}")
    print("\nCoefficient comparison:")
    print(comparison.to_string(float_format=lambda value: f"{value:.10f}"))


if __name__ == "__main__":
    main()
