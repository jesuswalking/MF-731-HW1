"""MF731: three-stock portfolio VaR. Run: python3 var_est.py.

Uses cached Yahoo Finance daily Close prices, downloading only if missing.
Results are written to work/var_est_results.json. Amounts are in dollars.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
SYMBOLS = ("MSFT", "AAPL", "GOOGL")
START, END = "2024-05-23", "2026-05-22"
VALUE, ALPHA, DECAY = 1_000_000.0, 0.95, 0.97
CAPS = np.array([3.109, 4.536, 4.640])
SEED, SAMPLES = 731, 100_000


def load_prices():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    frames, sources = [], {}
    for symbol in SYMBOLS:
        csv_path = data / f"{symbol.lower()}_close.csv"
        raw_path = data / f"{symbol.lower()}_yahoo_raw.json"
        params = {
            "period1": int(datetime(2024, 5, 23, tzinfo=timezone.utc).timestamp()),
            "period2": int(datetime(2026, 5, 23, tzinfo=timezone.utc).timestamp()),
            "interval": "1d",
        }
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{urlencode(params)}"
        if not csv_path.exists():
            if not raw_path.exists():
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=30) as response:
                    raw_path.write_bytes(response.read())
            result = json.loads(raw_path.read_text())["chart"]["result"][0]
            dates = (pd.to_datetime(result["timestamp"], unit="s", utc=True)
                     .tz_convert("America/New_York").tz_localize(None).normalize())
            close = pd.Series(result["indicators"]["quote"][0]["close"],
                              index=dates, name="Close").loc[START:END]
            close.to_csv(csv_path, index_label="Date")
        frame = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        frame = frame.loc[START:END, ["Close"]].rename(columns={"Close": symbol})
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Invalid dates for {symbol}.")
        frames.append(frame)
        sources[symbol] = {"url": url, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
        if raw_path.exists():
            sources[symbol]["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    prices = pd.concat(frames, axis=1)
    if prices.isna().any().any() or not (prices > 0).all().all():
        raise ValueError("Prices must be positive and aligned across all three stocks.")
    if str(prices.index[0].date()) != START or str(prices.index[-1].date()) != END:
        raise ValueError("Price sample does not cover both requested endpoints.")
    return prices, sources


def ewma(returns):
    mean = np.zeros(3)
    covariance = np.zeros((3, 3))
    for x in returns:
        residual = x - mean  # Use the mean before the current update.
        covariance = DECAY * covariance + (1 - DECAY) * np.outer(residual, residual)
        mean = DECAY * mean + (1 - DECAY) * x
    return mean, covariance


def losses(x, weights):
    return {
        "Full": -VALUE * (np.expm1(x) @ weights),
        "Linear": -VALUE * (x @ weights),
        "Quadratic": -VALUE * ((x + 0.5 * x**2) @ weights),
    }


def empirical_var(loss):
    # Lower empirical quantile: smallest loss with cumulative mass >= alpha.
    return float(np.quantile(loss, ALPHA, method="inverted_cdf"))


def main():
    prices, sources = load_prices()
    x = np.diff(np.log(prices.to_numpy()), axis=0)
    weights = CAPS / CAPS.sum()
    mean, covariance = ewma(x)
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError("EWMA covariance is not positive semidefinite.")
    historical = {name: empirical_var(loss) for name, loss in losses(x, weights).items()}
    rng = np.random.default_rng(SEED)
    normal_x = rng.multivariate_normal(mean, covariance, size=SAMPLES)
    simulated = {name: empirical_var(loss) for name, loss in losses(normal_x, weights).items()}
    linear_exact = float(-VALUE * (weights @ mean)
                         + VALUE * np.sqrt(weights @ covariance @ weights) * norm.ppf(ALPHA))
    normal_var = {"Full": simulated["Full"], "Linear": linear_exact,
                  "Quadratic": simulated["Quadratic"]}
    observations = pd.concat([prices.head(), prices.tail()]).reset_index()
    observations["Date"] = observations["Date"].dt.strftime("%Y-%m-%d")
    result = {
        "price_count": len(prices), "return_count": len(x),
        "symbols": list(SYMBOLS), "observations": observations.to_dict(orient="records"),
        "market_caps_trillion": CAPS.tolist(), "weights": weights.tolist(),
        "portfolio_value": VALUE, "confidence": ALPHA, "decay": DECAY,
        "ewma_initial_mean": [0, 0, 0], "ewma_initial_covariance": np.zeros((3, 3)).tolist(),
        "ewma_mean": mean.tolist(), "ewma_covariance": covariance.tolist(),
        "historical_var": historical, "normal_var": normal_var,
        "linear_mc_diagnostic": simulated["Linear"],
        "normal_samples": SAMPLES, "seed": SEED, "data_sources": sources,
    }
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    (work / "var_est_results.json").write_text(json.dumps(result, indent=2) + "\n")
    prices.to_csv(work / "var_est_prices.csv", index_label="Date")
    print(f"{len(prices)} closing prices per stock; {len(x)} daily log returns.")
    print("Weights:", dict(zip(SYMBOLS, weights)))
    for name in historical:
        print(f"{name:10s} historical ${historical[name]:,.2f}; normal ${normal_var[name]:,.2f}")


if __name__ == "__main__":
    main()
