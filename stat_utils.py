import numpy as np

#Price calculation functions

def mid_price(bid, ask):
    return mid_price

def micro_price():
    return micro_price

def vwap():
    return vwap

# Functions copied from the FinTech Python Examples, with AI descriptions added for IMC Prosperity 4 context.

def compute_returns(price: pd.Series) -> pd.DataFrame:
    '''
    Compute simple and log returns from a price series.

    Use this function to compute returns for any price series in IMC Prosperity 4 data.
    Simple returns (R_t) are percentage changes, while log returns (r_t) are differences of log prices.
    Log returns are often preferred for statistical modeling due to their properties (e.g., time-additivity).

    Parameters:
    price (pd.Series): Series of prices indexed by time.

    Returns:    
    pd.DataFrame: DataFrame with columns "R_t" (simple returns), "r_t" (log returns), and "p_t" (log prices).
    '''
    import numpy as np
    import pandas as pd
    p = price.astype(float).dropna()
    out = pd.DataFrame(index=p.index)
    out["R_t"] = p.pct_change()                 # simple return
    out["r_t"] = np.log(p).diff()               # log return
    out["p_t"] = np.log(p)
    return out.dropna()

def distribution_summary(x):
    """
    Compute summary statistics for a distribution, including normality tests.

    Use this function to assess the basic properties of price series or returns in IMC Prosperity 4 data.
    For example, check if EMERALDS prices are stationary (low variance, high kurtosis) or TOMATOES are drifting (higher variance).
    The Jarque-Bera test checks for normality; reject null if p-value < 0.05, indicating non-normal distribution.

    Parameters:
    x (pd.Series or array-like): The data series to analyze.

    Returns:
    dict: Dictionary with mean, variance, skewness, kurtosis, JB statistic, and p-value.
    """
    import numpy as np
    from scipy.stats import jarque_bera, skew, kurtosis

    x = x.dropna()
    jb = jarque_bera(x)
    return {
        "mean": x.mean(),
        "variance": x.var(ddof=1),
        "skewness": skew(x, bias=False),
        "kurtosis": kurtosis(x, fisher=False, bias=False),
        "jb_stat": jb.statistic(),
        "jb_pvalue": jb.pvalue()
    }

def acf_pacf_diagnostics(r, nlags=20):
    """
    Compute autocorrelation function (ACF) and partial autocorrelation function (PACF) for a time series.

    Use this to diagnose autocorrelation in returns or price differences. For mean-reverting products like EMERALDS,
    expect negative ACF at lag 1. For trending products like TOMATOES, ACF may decay slowly.
    Helps in selecting ARIMA orders or detecting seasonality.

    Parameters:
    r (pd.Series): The time series (e.g., returns).
    nlags (int): Number of lags to compute.

    Returns:
    tuple: (acf_values, pacf_values) arrays.
    """
    from statsmodels.tsa.stattools import acf, pacf

    r = r.dropna()
    acf_vals = acf(r, nlags=nlags, fft=True)  # includes lag 0
    pacf_vals = pacf(r, nlags=nlags, method="ywm")  # Yule-Walker MLE variant
    return acf_vals, pacf_vals

def fit_arima_and_forecast(returns, p=1, q=1):
    """
    Fit an ARIMA model to returns and forecast the next step.

    Use for modeling and forecasting price returns. For stationary series like EMERALDS returns,
    ARIMA(1,0,1) might capture mean reversion. For non-stationary like TOMATOES, consider differencing first.
    The 1-step forecast can be used for directional signals in trading strategies.

    Parameters:
    returns (pd.Series): The returns series.
    p (int): AR order.
    q (int): MA order.

    Returns:
    tuple: (fitted_model, one_step_forecast)
    """
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(returns.dropna(), order=(p, 0, q))
    res = model.fit()
    mean_1step = float(res.get_forecast(steps=1).predicted_mean.iloc[0])
    return res, mean_1step

def fit_garch11(returns):
    """
    Fit a GARCH(1,1) model to returns for volatility modeling.

    Use to model time-varying volatility in returns. Useful for risk management or volatility-based signals
    in IMC Prosperity 4, where products may have changing volatility regimes across rounds.
    The conditional volatility can indicate periods of high uncertainty.

    Parameters:
    returns (pd.Series): The returns series.

    Returns:
    tuple: (fitted_model, conditional_volatility_series)
    """
    from arch import arch_model

    am = arch_model(
        returns.dropna(),
        mean="AR", lags=1,
        vol="GARCH", p=1, q=1,
        dist="normal"
    )
    res = am.fit(disp="off")
    return res, res.conditional_volatility

def stationarity_panel(x):
    """
    Test for stationarity using Augmented Dickey-Fuller (ADF) and KPSS tests.

    Use to determine if a price series is stationary (mean-reverting) or has a unit root (random walk).
    For EMERALDS, expect stationarity (ADF p-value < 0.05, KPSS p-value > 0.05).
    For TOMATOES, may indicate non-stationarity. Critical for model selection in time series analysis.

    Parameters:
    x (pd.Series): The time series to test.

    Returns:
    dict: ADF and KPSS statistics and p-values.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    x = x.dropna()
    adf = adfuller(x, regression="c", autolag="AIC")
    kps = kpss(x, regression="c", nlags="auto")
    return {
        "adf_stat": adf[0], "adf_pvalue": adf[1], "adf_usedlag": adf[2],
        "kpss_stat": kps[0], "kpss_pvalue": kps[1], "kpss_lags": kps[2],
    }

def ljung_box_suite(resid, m=20, model_df=0):
    """
    Perform Ljung-Box tests on residuals and squared residuals for autocorrelation.

    Use after fitting a model (e.g., ARIMA) to check if residuals are white noise.
    Low p-values indicate autocorrelation, suggesting model inadequacy.
    Test on squared residuals checks for ARCH effects (volatility clustering), common in financial returns.

    Parameters:
    resid (pd.Series): Model residuals.
    m (int): Lag for the test.
    model_df (int): Degrees of freedom to subtract (e.g., number of parameters).

    Returns:
    tuple: (ljung_box_on_residuals, ljung_box_on_squared_residuals)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = resid.dropna()
    lb_resid = acorr_ljungbox(r, lags=[m], model_df=model_df, return_df=True)
    lb_sq = acorr_ljungbox(r**2, lags=[m], model_df=model_df, return_df=True)
    return lb_resid, lb_sq

def arch_lm(resid, nlags=10, ddof=0):
    """
    Engle's ARCH Lagrange Multiplier test for heteroskedasticity.

    Use to detect ARCH effects in residuals. If significant (p-value < 0.05), consider GARCH modeling.
    Important for financial time series where volatility clusters.

    Parameters:
    resid (pd.Series): Model residuals.
    nlags (int): Number of lags.
    ddof (int): Degrees of freedom adjustment.

    Returns:
    dict: LM statistic, p-value, F-statistic, F p-value.
    """
    from statsmodels.stats.diagnostic import het_arch

    lm, lm_pvalue, fval, f_pvalue = het_arch(resid.dropna(), nlags=nlags, ddof=ddof)
    return {
        "lm_stat": lm, "lm_pvalue": lm_pvalue,
        "f_stat": fval, "f_pvalue": f_pvalue,
    }

def innovation_checks(z):
    """
    Test innovations (standardized residuals) for normality using multiple tests.

    Use after model fitting to assess if residuals are normally distributed.
    Jarque-Bera, omnibus, and Lilliefors tests provide complementary checks.
    Non-normality may require robust estimation or distribution adjustments.

    Parameters:
    z (array-like): Standardized residuals.

    Returns:
    dict: P-values from JB, omnibus, and Lilliefors tests.
    """
    import numpy as np
    from scipy.stats import jarque_bera, normaltest
    from statsmodels.stats.diagnostic import lilliefors
    
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    z_std = (z - z.mean()) / z.std(ddof=1)
    jb = jarque_bera(z_std)
    omni = normaltest(z_std)
    lf_stat, lf_p = lilliefors(z, dist="norm", pvalmethod="table")
    return {"jb_pvalue": jb.pvalue, "omni_pvalue": omni.pvalue, "lilliefors_pvalue": lf_p}

def cusum_stability_test(y, X):
    """
    CUSUM test for structural breaks in regression.

    Use to detect if a trading model's parameters change over time, e.g., due to regime shifts in IMC rounds.
    Significant breaks may indicate need for adaptive strategies.

    Parameters:
    y (array-like): Dependent variable.
    X (array-like): Independent variables.

    Returns:
    dict: CUSUM statistic, p-value, critical values.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, Xc).fit()
    n_params = int(ols.df_model) + int(ols.k_constant)
    sup_b, pvalue, crit = breaks_cusumolsresid(ols.resid, ddof=n_params)
    return {"cusum_stat": sup_b, "pvalue": pvalue, "critical_values": crit}

def select_arma_order(r, max_ar=4, max_ma=4):
    """
    Select optimal ARMA order using AIC and BIC.

    Use to automatically choose AR and MA orders for ARMA modeling of returns.
    Lower AIC/BIC preferred. Useful for exploratory analysis of time series properties.

    Parameters:
    r (pd.Series): The time series.
    max_ar (int): Max AR order.
    max_ma (int): Max MA order.

    Returns:
    dict: Selected orders by AIC/BIC, and full tables.
    """
    from statsmodels.tsa.stattools import arma_order_select_ic

    y = r.dropna()
    sel = arma_order_select_ic(y, max_ar=max_ar, max_ma=max_ma, ic=["aic", "bic"], trend="c")
    return {
        "aic_min_order": sel.aic_min_order,
        "bic_min_order": sel.bic_min_order,
        "aic_table": sel.aic,
        "bic_table": sel.bic,
    }

def fit_var_and_forecast(df, maxlags=12, h=5):
    """
    Fit a Vector Autoregression (VAR) model and forecast.

    Use for multivariate time series analysis, e.g., modeling relationships between multiple products.
    Forecasts can be used for basket trading or arbitrage signals in IMC Prosperity 4.

    Parameters:
    df (pd.DataFrame): Multivariate time series.
    maxlags (int): Max lags to consider.
    h (int): Forecast horizon.

    Returns:
    tuple: (fitted_model, selected_lag, forecasts)
    """
    from statsmodels.tsa.api import VAR

    y = df.dropna()
    model = VAR(y)
    sel = model.select_order(maxlags=maxlags)
    p = int(sel.selected_orders["bic"])
    res = model.fit(p)
    fcst = res.forecast(y.values[-p:], steps=h)
    return res, p, fcst

def check_granger_causality(df, max_lag=4):
    """
    Test Granger causality between time series.

    Use to check if one product's price changes predict another's, e.g., in basket products.
    Significant causality may indicate leading/lagging relationships for trading.

    Parameters:
    df (pd.DataFrame): Two-column DataFrame.
    max_lag (int): Max lag for test.

    Returns:
    float: P-value of the test.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    y = df.dropna()
    res = grangercausalitytests(y, maxlag=max_lag, verbose=False)
    p_value = res[max_lag][0]["ssr_ftest"][1]
    return p_value

def johansen_rank_test(df, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration rank test.

    Use to test for cointegration between products, e.g., in basket/ETF products.
    Rank indicates number of cointegrating relationships. Useful for statistical arbitrage.

    Parameters:
    df (pd.DataFrame): Multivariate series.
    det_order (int): Deterministic term order.
    k_ar_diff (int): Lag order.

    Returns:
    dict: Trace and max eigenvalue statistics and critical values.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    y = df.dropna()
    joh = coint_johansen(y, det_order=det_order, k_ar_diff=k_ar_diff)
    return {
        "trace_stat": joh.trace_stat,
        "trace_crit_vals": joh.trace_stat_crit_vals,
        "max_eig_stat": joh.max_eig_stat,
        "max_eig_crit_vals": joh.max_eig_stat_crit_vals,
    }

def fit_vecm(df, coint_rank=1, k_ar_diff=1, deterministic="n"):
    """
    Fit a Vector Error Correction Model (VECM).

    Use for cointegrated systems to model short-run dynamics and long-run equilibrium.
    Useful for pairs trading or basket hedging in IMC Prosperity 4.

    Parameters:
    df (pd.DataFrame): Multivariate series.
    coint_rank (int): Cointegration rank.
    k_ar_diff (int): Lag order.
    deterministic (str): Deterministic terms.

    Returns:
    dict: Alpha, beta matrices, residuals.
    """
    from statsmodels.tsa.vector_ar.vecm import VECM

    y = df.dropna()
    model = VECM(
        y,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic
    )
    res = model.fit()
    return {"alpha": res.alpha, "beta": res.beta, "resid": res.resid}

def select_vecm_spec(df, maxlags=12, deterministic="co", signif=0.05):
    """
    Select VECM specification (lags and cointegration rank).

    Use to automatically determine model parameters for VECM.
    Helps in setting up error correction models for cointegrated products.

    Parameters:
    df (pd.DataFrame): Multivariate series.
    maxlags (int): Max lags.
    deterministic (str): Deterministic terms.
    signif (float): Significance level.

    Returns:
    dict: Selected k_ar_diff and coint_rank.
    """
    from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank

    def _det_order(d):
        d = (d or "n").lower()
        if "li" in d or "lo" in d: return 1
        if "ci" in d or "co" in d: return 0
        return -1
    y = df.dropna()
    sel = select_order(y, maxlags=maxlags, deterministic=deterministic).selected_orders
    k_ar_diff = int(sel.get("bic") or sel.get("aic") or 1)
    rank = select_coint_rank(
        y, det_order=_det_order(deterministic),
        k_ar_diff=k_ar_diff, method="trace", signif=signif
    )
    return {"k_ar_diff": k_ar_diff, "coint_rank": rank.rank}

# Additional Functions
