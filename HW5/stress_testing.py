"""MF731 HW5: EWMA stress test for a market-cap-weighted MSFT/AAPL portfolio.

Run: python3 stress_testing.py
The supplied historical prices allow an offline, reproducible run.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent
SYMBOLS = ("MSFT", "AAPL")
CAPS = np.array([448.77, 575.11])  # Billions of dollars on 2016-09-01.
VALUE, ALPHA, DECAY, DAYS = 1_000_000.0, 0.95, 0.97, 10


def load_returns(data_dir):
    price_path = data_dir / "daily_close.csv"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    digest = hashlib.sha256(price_path.read_bytes()).hexdigest()
    if digest != manifest["csv_sha256"]:
        raise ValueError("Historical price file does not match its manifest.")
    with price_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = [row["Date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Price dates must be unique and increasing.")
    if dates[0] != "2011-08-31" or dates[-1] != "2016-09-01":
        raise ValueError("Prices must cover 2011-08-31 through 2016-09-01.")
    prices = np.array([[float(row[s]) for s in SYMBOLS] for row in rows])
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    returns = np.diff(np.log(prices), axis=0)
    return returns, dates, manifest


def update(mean, covariance, x):
    """One EWMA update, also accepting one state per simulated path."""
    residual = x - mean  # The covariance uses the mean BEFORE this update.
    next_covariance = DECAY * covariance + (1 - DECAY) * (
        residual[..., :, None] * residual[..., None, :]
    )
    next_mean = DECAY * mean + (1 - DECAY) * x
    return next_mean, next_covariance


def fit_ewma(returns):
    mean, covariance = np.zeros(2), np.zeros((2, 2))
    for x in returns:
        mean, covariance = update(mean, covariance, x)
    return mean, covariance


def first_day_shock(mean, covariance):
    sigma = np.sqrt(np.diag(covariance))
    rho = covariance[0, 1] / (sigma[0] * sigma[1])
    apple = mean[1] - 5 * sigma[1]
    conditional_mean = mean[0] + covariance[0, 1] / covariance[1, 1] * (
        apple - mean[1]
    )
    conditional_sd = sigma[0] * np.sqrt(1 - rho**2)
    microsoft = conditional_mean - 5 * conditional_sd
    return np.array([microsoft, apple]), conditional_mean, conditional_sd, rho


def simulate(mean, covariance, shock, positions, paths, seed):
    rng = np.random.default_rng(seed)
    post_mean, post_covariance = update(mean, covariance, shock)
    means = np.broadcast_to(post_mean, (paths, 2)).copy()
    covariances = np.broadcast_to(post_covariance, (paths, 2, 2)).copy()
    total_returns = np.broadcast_to(shock, (paths, 2)).copy()
    for day in range(2, DAYS + 1):
        chol = np.linalg.cholesky(covariances)
        z = rng.standard_normal((paths, 2))
        x = means + np.einsum("mij,mj->mi", chol, z)
        total_returns += x
        if day < DAYS:
            means, covariances = update(means, covariances, x)
    # The question specifies linearized losses with fixed initial positions.
    return -(total_returns @ positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work")
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    if args.paths < 2 or args.seed < 0:
        parser.error("paths must be at least 2 and seed must be nonnegative")

    returns, dates, manifest = load_returns(args.data_dir)
    mean, covariance = fit_ewma(returns)
    weights = CAPS / CAPS.sum()
    positions = VALUE * weights
    sigma = np.sqrt(np.diag(covariance))
    var_1day = float(-positions @ mean + NormalDist().inv_cdf(ALPHA) *
                     np.sqrt(positions @ covariance @ positions))
    # Use the assignment's square-root-of-time rule for the entire VaR.
    var_10day = np.sqrt(DAYS) * var_1day
    capital = 3 * var_10day
    shock, conditional_mean, conditional_sd, rho = first_day_shock(mean, covariance)
    losses = simulate(mean, covariance, shock, positions, args.paths, args.seed)
    post_mean, post_covariance = update(mean, covariance, shock)
    exact_mean = float(-positions @ (shock + (DAYS - 1) * post_mean))
    coefficients = 1 + (1 - DECAY) * np.arange(DAYS - 1)
    exact_variance = float(np.sum(coefficients**2) *
                           (positions @ post_covariance @ positions))
    mean_se = np.sqrt(exact_variance / args.paths)
    count_var = int(np.count_nonzero(losses > var_10day))
    count_capital = int(np.count_nonzero(losses > capital))
    stressed_var = float(np.quantile(losses, ALPHA, method="inverted_cdf"))
    result = {
        "symbols": list(SYMBOLS), "price_count": len(dates),
        "return_count": len(returns), "first_return_date": dates[1],
        "last_return_date": dates[-1], "price_field": manifest["price_field"],
        "data_csv_sha256": manifest["csv_sha256"],
        "market_caps_billion": CAPS.tolist(), "weights": weights.tolist(),
        "dollar_positions": positions.tolist(), "portfolio_value": VALUE,
        "confidence": ALPHA, "days": DAYS, "decay_mean": DECAY,
        "decay_covariance": DECAY, "paths": args.paths, "seed": args.seed,
        "numpy_version": np.__version__, "rng": "PCG64",
        "ewma_mean": mean.tolist(), "ewma_covariance": covariance.tolist(),
        "ewma_daily_sd": sigma.tolist(), "ewma_correlation": float(rho),
        "initial_var_1day": var_1day, "initial_var_10day": float(var_10day),
        "initial_regulatory_capital": float(capital),
        "msft_conditional_mean": float(conditional_mean),
        "msft_conditional_sd": float(conditional_sd),
        "shock_log_returns": shock.tolist(),
        "shock_day_loss": float(-positions @ shock),
        "post_shock_mean": post_mean.tolist(),
        "post_shock_covariance": post_covariance.tolist(),
        "a_average_loss": float(losses.mean()), "b_stressed_var": stressed_var,
        "c_exceed_initial_var_count": count_var,
        "c_exceed_initial_var_pct": 100 * count_var / args.paths,
        "d_exceed_capital_count": count_capital,
        "d_exceed_capital_pct": 100 * count_capital / args.paths,
        "diagnostics": {
            "exact_expected_loss": exact_mean,
            "exact_loss_sd": float(np.sqrt(exact_variance)),
            "sample_loss_sd": float(losses.std(ddof=1)),
            "sample_mean_standard_error": float(mean_se),
            "sample_mean_z_score": float((losses.mean() - exact_mean) / mean_se),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stress_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savetxt(args.output_dir / "stressed_losses.csv", losses, delimiter=",",
               header="LossUSD", comments="", fmt="%.10f")
    print(f"{len(returns):,} daily returns; {args.paths:,} paths; seed {args.seed}.")
    print(f"Initial one-day VaR: ${var_1day:,.2f}")
    print(f"Initial ten-day VaR: ${var_10day:,.2f}; capital: ${capital:,.2f}")
    print(f"(a) Average stressed ten-day loss: ${losses.mean():,.2f}")
    print(f"(b) Stressed ten-day 95% VaR: ${stressed_var:,.2f}")
    print(f"(c) Exceed initial ten-day VaR: {100 * count_var / args.paths:.3f}%")
    print(f"(d) Exceed initial capital: {100 * count_capital / args.paths:.3f}%")


if __name__ == "__main__":
    main()
