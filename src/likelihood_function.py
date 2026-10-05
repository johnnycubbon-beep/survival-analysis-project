"""Efron partial log-likelihood for a Cox proportional hazards model."""

import numpy as np


DEFAULT_COVARIATES = ("age", "ecog", "karnoPH", "karnoPAT")


def efron_log_likelihood(beta, data, covariate_columns=DEFAULT_COVARIATES):
    """Return the Cox partial log-likelihood with Efron's tie correction.

    Parameters
    ----------
    beta : array-like, shape (n_covariates,)
        Regression coefficients, in the same order as ``covariate_columns``.
    data : pandas.DataFrame
        One row per patient, with ``TIME`` (event or censoring time) and ``Y``
        (1 for an observed death, 0 for right censoring), plus covariates.
    covariate_columns : sequence of str
        Names and order of the columns used as model covariates.

    Notes
    -----
    A subject censored at time t is included in the risk set at t
    (``TIME >= t``), which is the usual convention for this partial
    likelihood. The function does not load or modify any data at import time.
    """
    required = ["TIME", "Y", *covariate_columns]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required data columns: {missing_columns}")

    beta = np.asarray(beta, dtype=float).reshape(-1)
    if beta.size != len(covariate_columns):
        raise ValueError(
            f"beta has {beta.size} values, but {len(covariate_columns)} "
            "covariates were specified"
        )

    try:
        times = data["TIME"].to_numpy(dtype=float)
        status = data["Y"].to_numpy(dtype=float)
        x = data.loc[:, covariate_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("TIME, Y, and covariates must contain numeric values") from exc

    if not (np.isfinite(times).all() and np.isfinite(status).all()
            and np.isfinite(x).all()):
        raise ValueError("TIME, Y, and covariates must not contain missing or infinite values")
    if not np.isin(status, [0.0, 1.0]).all():
        raise ValueError("Y must be coded 1 for death and 0 for censoring")
    if (times < 0).any():
        raise ValueError("TIME values must be nonnegative")

    eta = x @ beta
    event_times = np.unique(times[status == 1.0])
    log_likelihood = 0.0

    for time in event_times:
        deaths = (times == time) & (status == 1.0)
        risk_set = times >= time
        tied_death_eta = eta[deaths]
        risk_eta = eta[risk_set]
        tie_count = tied_death_eta.size

        # Scale exponentials by the largest risk-set predictor. This preserves
        # the Efron terms while avoiding overflow in exp(eta).
        shift = np.max(risk_eta)
        risk_sum = np.exp(risk_eta - shift).sum()
        death_sum = np.exp(tied_death_eta - shift).sum()

        log_likelihood += tied_death_eta.sum()
        for r in range(tie_count):
            denominator = risk_sum - (r / tie_count) * death_sum
            if denominator <= 0 or not np.isfinite(denominator):
                raise FloatingPointError(
                    f"Nonpositive or nonfinite Efron denominator at time {time}"
                )
            log_likelihood -= shift + np.log(denominator)

    return float(log_likelihood)
