"""MF731 HW1: run with Python 3, numpy, pandas, scipy, matplotlib.
Run: python solution.py. Outputs are saved beside this script.
Uses Yahoo Finance Close (not dividend-adjusted Close).
"""
import json
import hashlib
import platform
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SEED, M, N, LAM, DAYS = 731, 100, 100, 0.94, 252
START, END = "2021-08-01", "2026-07-30"
TRAIN_START, TRAIN_END = "2025-06-01", "2026-05-31"
FORECAST_START = "2026-06-01"


def load_prices():
    # Retrieve extra history for the initial 100-return window.
    p1 = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(2026, 7, 31, tzinfo=timezone.utc).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           f"?period1={p1}&period2={p2}&interval=1d")
    cache = ROOT / "spy_yahoo_raw.json"
    if not cache.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            cache.write_bytes(response.read())
    raw = cache.read_bytes()
    obj = json.loads(raw)["chart"]["result"][0]
    dates = (pd.to_datetime(obj["timestamp"], unit="s", utc=True)
             .tz_convert("America/New_York").tz_localize(None).normalize())
    close = pd.Series(obj["indicators"]["quote"][0]["close"],
                      index=dates, name="Close").sort_index()
    assert close.index.is_unique and close.notna().all()
    assert (close > 0).all() and close.index[-1] >= pd.Timestamp(END)
    close = close.loc["2021-01-01":END]
    close.to_csv(ROOT / "spy_close.csv", index_label="Date")
    return close, url, hashlib.sha256(raw).hexdigest()


def historical_variances(close):
    r = np.log(close / close.shift(1))
    # Date t is the day being forecast: use only returns through t-1.
    ma = r.pow(2).rolling(N).mean().shift(1)
    dates = close.loc[START:END].index
    first = dates[0]
    seed_returns = r.loc[r.index < first].dropna().tail(N)
    assert len(seed_returns) == N
    v = float(seed_returns.var(ddof=1))
    ew = pd.Series(index=dates, dtype=float)
    for date in dates:
        ew.loc[date] = v
        v = LAM * v + (1 - LAM) * r.loc[date] ** 2
    return r, ma.loc[dates], ew, seed_returns


def filter_variance(theta, x, h0):
    mu, omega, alpha, beta = theta
    e = x - mu
    h = np.empty(len(x))
    h[0] = h0
    for t in range(1, len(x)):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
    return h, e


def fit_garch(x):
    # x is in daily percentage points; h is in squared percent units.
    h0 = float(np.var(x, ddof=1))

    def nll(theta):
        h, e = filter_variance(theta, x, h0)
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            return 1e30
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + e * e / h)

    bounds = [(None, None), (1e-10, None), (0, 0.999999),
              (0, 0.999999)]
    cons = {"type": "ineq", "fun": lambda p: 0.999999 - p[2] - p[3]}
    fits = []
    for a, b in [(0.05, 0.90), (0.10, 0.80), (0.15, 0.80),
                 (0.03, 0.96), (0.20, 0.50), (0.01, 0.20)]:
        p0 = [float(x.mean()), h0 * (1 - a - b), a, b]
        result = minimize(nll, p0, method="SLSQP", bounds=bounds,
                          constraints=cons,
                          options={"maxiter": 3000, "ftol": 1e-11})
        if result.success and cons["fun"](result.x) >= -1e-8:
            fits.append(result)
    if not fits:
        raise RuntimeError("No GARCH optimization converged.")
    best = min(fits, key=lambda v: v.fun)
    h, e = filter_variance(best.x, x, h0)
    return best, h, e, h0, [float(f.fun) for f in fits]


def simulate(theta, last_h, last_e, horizon):
    mu, w, a, b = theta
    first_h = w + a * last_e ** 2 + b * last_h
    rng = np.random.default_rng(SEED)
    paths = np.empty((horizon, M))
    paths[0] = first_h  # Common known one-step variance.
    for k in range(1, horizon):
        z = rng.standard_normal(M)
        shock = np.sqrt(paths[k - 1]) * z
        paths[k] = w + a * shock ** 2 + b * paths[k - 1]
    rho = a + b
    long_run = w / (1 - rho)
    expected = long_run + rho ** np.arange(horizon) * (first_h - long_run)
    assert np.all(paths > 0) and np.ptp(paths[0]) == 0
    assert np.allclose(expected[1:], w + rho * expected[:-1])
    return paths, expected


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylabel("Annualized volatility (%)")


def main():
    close, url, digest = load_prices()
    r, ma, ew, seed_returns = historical_variances(close)
    # Restrict prices before differencing: no price outside fit window.
    train_close = close.loc[TRAIN_START:TRAIN_END]
    train = 100 * np.log(train_close / train_close.shift(1)).dropna()
    fit, h, e, h0, objectives = fit_garch(train.to_numpy())
    mu, w, a, b = [float(v) for v in fit.x]
    dates = close.loc[FORECAST_START:END].index
    assert train.index[-1] < dates[0]
    paths, expected = simulate(fit.x, h[-1], e[-1], len(dates))
    # Percent-return GARCH needs no extra factor of 100 here.
    sim_vol, analytical = np.sqrt(DAYS * paths), np.sqrt(DAYS * expected)
    ma_vol, ew_vol = 100 * np.sqrt(DAYS * ma), 100 * np.sqrt(DAYS * ew)
    low, median, high = np.quantile(sim_vol, [0.05, 0.50, 0.95], axis=1)
    mc_rms = np.sqrt(DAYS * paths.mean(axis=1))
    mean_vol = sim_vol.mean(axis=1)
    table = pd.DataFrame({"GARCH": analytical, "MC_RMS": mc_rms,
                          "MC_mean_vol": mean_vol, "P05": low,
                          "Median": median, "P95": high,
                          "MA": ma_vol.loc[dates], "EWMA": ew_vol.loc[dates]},
                         index=dates)
    table.to_csv(ROOT / "daily_forecasts.csv", index_label="Date")
    pd.DataFrame(sim_vol, index=dates).to_csv(ROOT / "100_paths.csv")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "figure.dpi": 130, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(9, 3.65))
    ax.plot(ma_vol, color="#145D80", lw=1.5, label="MA (100 trading days)")
    ax.plot(ew_vol, color="#D27B26", lw=1.2, label="EWMA (lambda = 0.94)")
    style_axis(ax)
    ax.set_title("SPY daily log-return volatility | 2021-08-02 to 2026-07-30")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ma_ewma.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.7), sharex=True)
    ax = axes[0]
    ax.plot(dates, sim_vol, color="#8CA6BF", alpha=0.28, lw=0.65)
    ax.plot(dates, analytical, color="#192B4D", lw=2,
            label="GARCH: sqrt(252 x expected variance)")
    ax.plot(dates, table.MA, color="#007D80", lw=2, label="MA (100)")
    ax.plot(dates, table.EWMA, color="#D27B26", lw=2, label="EWMA (0.94)")
    ax.set_title("100 GARCH volatility paths | forecast origin: 2026-05-29")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_axis(ax)
    ax = axes[1]
    ax.fill_between(dates, low, high, color="#8CA6BF", alpha=0.3,
                    label="Pointwise 5th-95th simulation percentiles")
    ax.plot(dates, median, "--", color="#526987", label="Simulation median")
    ax.plot(dates, analytical, color="#192B4D", lw=2, label="GARCH forecast")
    ax.plot(dates, table.MA, color="#007D80", lw=2, label="MA (100)")
    ax.plot(dates, table.EWMA, color="#D27B26", lw=2, label="EWMA (0.94)")
    style_axis(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(FIG / "forecast_paths.pdf")
    plt.close(fig)

    summary = {
        "source": url, "raw_sha256": digest, "prices": len(close),
        "plot_days": len(ma), "train_prices": len(train_close),
        "train_returns": len(train), "horizon": len(dates),
        "train_first_return": str(train.index[0].date()),
        "train_last_return": str(train.index[-1].date()),
        "seed_first_return": str(seed_returns.index[0].date()),
        "seed_last_return": str(seed_returns.index[-1].date()),
        "ewma_seed": float(seed_returns.var(ddof=1)),
        "mu": mu, "omega": w, "alpha": a, "beta": b,
        "persistence": a + b, "h0": h0, "last_h": float(h[-1]),
        "last_residual": float(e[-1]), "nll": float(fit.fun),
        "optimizer_message": fit.message, "objectives": objectives,
        "AIC": 2 * fit.fun + 8, "BIC": 2 * fit.fun + 4 * np.log(len(train)),
        "long_run_ann_pct": float(np.sqrt(DAYS * w / (1 - a - b))),
        "half_life": float(np.log(0.5) / np.log(a + b)),
        "start": table.iloc[0].to_dict(), "end": table.iloc[-1].to_dict(),
        "period_means": table.mean().to_dict(),
        "ma_peak": float(ma_vol.max()), "ma_peak_date": str(ma_vol.idxmax().date()),
        "ew_peak": float(ew_vol.max()), "ew_peak_date": str(ew_vol.idxmax().date()),
        "ma_band_count": int(((table.MA >= low) & (table.MA <= high)).sum()),
        "ewma_band_count": int(((table.EWMA >= low) & (table.EWMA <= high)).sum()),
        "mc_variance_relative_error": float(abs(paths[-1].mean() / expected[-1] - 1)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__}}
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
