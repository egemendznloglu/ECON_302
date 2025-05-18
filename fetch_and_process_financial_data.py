#!/usr/bin/env python3
import sys
import logging
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white, breaks_cusumolsresid
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# ─── CONFIG & LOGGING ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
DATE_FMT = "%Y-%m-%d"

def validate_dates(start: str, end: str):
    """Ensure start < end and correct format."""
    try:
        s = pd.to_datetime(start, format=DATE_FMT)
        e = pd.to_datetime(end, format=DATE_FMT)
    except ValueError:
        logging.error("Dates must be YYYY-MM-DD")
        sys.exit(1)
    if s >= e:
        logging.error("Start date must be before end date")
        sys.exit(1)
    return start, end

# ─── DATA DOWNLOAD & MERGE ──────────────────────────────────────────────────────
def fetch_series(symbol: str, source: str, start: str, end: str, freq="ME") -> pd.Series:
    """Downloads and resamples a single series, or raises on failure."""
    try:
        if source == "yahoo":
            df = yf.download(symbol, start=start, end=end, progress=False)
            series = df["Close"].resample(freq).last().pct_change()
        else:
            # under the hood pdr.DataReader is making HTTP requests to the Federal
            #  Reserve Bank of St. Louis’s FRED API and downloading the series in
            #  real time.
            series = (
                pdr.DataReader(symbol, "fred", start, end)
                .resample(freq)
                .last()
                .squeeze()
            )
        series.name = symbol.lower()
        return series
    except Exception as e:
        logging.exception(f"Failed to fetch {symbol} from {source}")
        raise

def build_dataframe(start, end):
    """Download and merge the five series exactly as in comp.py."""
    # NDX returns
    ndx = yf.download("^NDX", start=start, end=end, auto_adjust=False, progress=False)
    ndx_me = ndx["Close"].resample("ME").last()
    ndx_ret = ndx_me.pct_change()
    ndx_ret.name = "ndx_ret"

    # FRED series at month-end
    ffr = (
        pdr.DataReader("FEDFUNDS", "fred", start, end)
        .resample("ME")
        .last()
        .squeeze()
    )
    ffr.name = "ffr"
    gs10 = (
        pdr.DataReader("GS10", "fred", start, end)
        .resample("ME")
        .last()
        .squeeze()
    )
    gs10.name = "gs10"
    cpi = (
        pdr.DataReader("CPIAUCSL", "fred", start, end)
        .resample("ME")
        .last()
       .squeeze()
    )
    cpi.name = "cpi"
    vix = (
        pdr.DataReader("VIXCLS", "fred", start, end)
        .resample("ME")
        .last()
        .squeeze()
    )
    vix.name = "vix"

    # 3) Combine into DataFrame and enforce lowercase names
    df = pd.concat([ndx_ret, ffr, gs10, cpi, vix], axis=1)
    df.columns = ["ndx_ret", "ffr", "gs10", "cpi", "vix"]
    logging.info(f"Combined DataFrame columns: {df.columns.tolist()}")
    return df

# ─── STATIONARITY TESTING ───────────────────────────────────────────────────────
def adf_test(series: pd.Series, name: str):
    """Run ADF and return stat, p-value, and critical values."""
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, usedlag, nobs, crit_vals, icbest = result
    logging.info(
        f"ADF test {name}: stat={stat:.4f}, p={pvalue:.4e}, "
        f"crit(1%)={crit_vals['1%']:.4f}, "
        f"crit(5%)={crit_vals['5%']:.4f}, "
        f"crit(10%)={crit_vals['10%']:.4f}"
    )
    return result


# ─── LAG SELECTION ──────────────────────────────────────────────────────────────
def select_lag_length(endog, exog, max_lag=6):
    """Fit VAR or simple OLS with different lags to pick optimal lag by AIC/SC/FPE."""
    best = {"aic": float("inf"), "bic": float("inf"), "fpe": float("inf")}
    best_lag = {"aic": None, "bic": None, "fpe": None}
    for lag in range(1, max_lag + 1):
        Xlag = pd.concat([exog.shift(i) for i in range(1, lag + 1)], axis=1).dropna()
        ylag = endog.loc[Xlag.index]
        mdl = sm.OLS(ylag, sm.add_constant(Xlag)).fit()
        if mdl.aic < best["aic"]:
            best["aic"], best_lag["aic"] = mdl.aic, lag
        if mdl.bic < best["bic"]:
            best["bic"], best_lag["bic"] = mdl.bic, lag
        if hasattr(mdl, "fpe") and mdl.fpe < best["fpe"]:
            best["fpe"], best_lag["fpe"] = mdl.fpe, lag
    logging.info(f"Optimal lags by AIC/BIC/FPE: {best_lag}")
    return best_lag

# ─── MAIN ROUTINE ───────────────────────────────────────────────────────────────
def main(start="2020-01-01", end="2024-04-30"):
    start, end = validate_dates(start, end)
    df = build_dataframe(start, end)
    logging.info(f"Downloaded data with {len(df)} rows.")

    # Test levels
    for col in df.columns:
        adf_test(df[col], col)

    # Transform to stationarity
    df["d_ffr"]     = df["ffr"].diff()
    df["d_gs10"]    = df["gs10"].diff()
    df["inflation"] = df["cpi"].pct_change() * 100
    df["d_vix"]     = df["vix"].diff()

    # Re-run ADF on the transformed series (and on returns)
    for col in ["ndx_ret", "d_ffr", "d_gs10", "inflation", "d_vix"]:
        adf_test(df[col], col)

    # Build regression sample
    model_df = df[["ndx_ret", "d_ffr", "d_gs10", "inflation", "d_vix"]].dropna()
    y = model_df["ndx_ret"]
    X = sm.add_constant(model_df[["d_ffr", "d_gs10", "inflation", "d_vix"]])

    # Lag selection
    select_lag_length(y, model_df[["d_ffr", "d_gs10", "inflation", "d_vix"]], max_lag=6)

    # Estimate final model
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    logging.info(ols.summary())

    # –– Logging SSR and FPE ––
    ssr = ols.ssr
    n   = int(ols.nobs)
    k   = int(ols.df_model) + 1   # regressors + constant
    fpe = (ssr / n) * ((n + k) / (n - k))
    logging.info(f"SSR={ssr:.6f}, FPE={fpe:.6e}")

    # Diagnostics
    dw = durbin_watson(ols.resid)
    _, bg_p, _, _ = acorr_breusch_godfrey(ols, nlags=1)
    _, wht_p, _, _ = het_white(ols.resid, ols.model.exog)
    logging.info(f"Durbin–Watson={dw:.4f}, BG p={bg_p:.4f}, White p={wht_p:.10e}")

    # –– Explicit JB test ––
    jb_stat, jb_p, _, _ = jarque_bera(ols.resid)
    logging.info(f"Jarque–Bera stat={jb_stat:.4f}, p={jb_p:.4f}")

    # –– CUSUM parameter‑stability test ––
    k = int(ols.df_model) + 1        # regressors plus constant
    cusum_stat, cusum_p, _ = breaks_cusumolsresid(ols.resid, k)
    logging.info(f"CUSUM test stat={cusum_stat:.4f}, p={cusum_p:.4f}")

    # ─── Visualization of Results ────────────────────────────────────────────
    import matplotlib.pyplot as plt
    # Forest plot of coefficients with 95% CI
    params = ols.params
    conf_int = ols.conf_int(alpha=0.05)
    lower_err = params - conf_int[0]
    upper_err = conf_int[1] - params
    fig, ax = plt.subplots()
    ax.errorbar(params, range(len(params)), xerr=[lower_err, upper_err], fmt='o', capsize=4)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params.index)
    ax.set_xlabel('Estimate')
    ax.set_title('Coefficient Estimates with 95% CI')
    plt.tight_layout()
    plt.show()

    # CUSUM parameter stability plot
    from statsmodels.stats.diagnostic import recursive_olsresiduals
    # Compute recursive OLS residuals and CUSUM
    rresid, rparams, rypred, rresid_std, rresid_scaled, cusum_vals, cusum_ci = recursive_olsresiduals(ols)
    lower, upper = cusum_ci[0], cusum_ci[1]
    fig, ax = plt.subplots()
    ax.plot(cusum_vals, label='CUSUM')
    ax.plot(lower, '--', label='Lower 5%')
    ax.plot(upper, '--', label='Upper 5%')
    ax.set_title('CUSUM Parameter Stability Plot')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()