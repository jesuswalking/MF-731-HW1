"""MF 731 HW3: one-day and ten-day VaR of 100 delta-hedged puts.

Run: python3 var_10day.py
Results are written to work/var_10day_results.json beside this script.
The ten-day calculation aggregates daily log returns and revalues the initial
portfolio. The initial delta is held fixed and the cash account accrues interest.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr


S0 = 152.51
STRIKE = 170.0
MATURITY = 0.292
RATE = 0.00119
VOLATILITY = 0.4907
DRIFT = 0.1691
DT = 1.0 / 252.0
PUTS = 100
ALPHA = 0.95
PATHS = 40_000
DAYS = 10
SEED = 731


def put_price_delta(stock, time):
    """Black-Scholes put price and delta, with calendar time in years."""
    stock = np.asarray(stock, dtype=float)
    tau = MATURITY - time
    if tau <= 0:
        return np.maximum(STRIKE - stock, 0.0), -1.0 * (stock < STRIKE)
    d1 = (np.log(stock / STRIKE) + (RATE + VOLATILITY**2 / 2) * tau) / (
        VOLATILITY * np.sqrt(tau)
    )
    d2 = d1 - VOLATILITY * np.sqrt(tau)
    price = STRIKE * np.exp(-RATE * tau) * ndtr(-d2) - stock * ndtr(-d1)
    delta = ndtr(d1) - 1.0
    return price, delta


def empirical_var(loss, alpha=ALPHA):
    """Inverse empirical CDF: observation ceil(alpha * n) in sorted losses."""
    loss = np.asarray(loss, dtype=float)
    index = int(np.ceil(alpha * loss.size)) - 1
    return float(np.partition(loss, index)[index])


def simulate():
    """Use the same 40,000 stock paths for both horizon estimates."""
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal((PATHS, DAYS))
    log_returns = (DRIFT - VOLATILITY**2 / 2) * DT + (
        VOLATILITY * np.sqrt(DT) * shocks
    )
    stock1 = S0 * np.exp(log_returns[:, 0])
    stock10 = S0 * np.exp(np.sum(log_returns, axis=1))

    price0, delta0 = put_price_delta(S0, 0.0)
    value0 = -float(price0)
    # Hedge purchases/sales use the initially empty cash account.
    cash0 = -float(delta0) * S0

    def loss(stock, horizon):
        price, _ = put_price_delta(stock, horizon)
        value = delta0 * stock + cash0 * np.exp(RATE * horizon) - price
        return PUTS * (value0 - value)

    one_day_loss = loss(stock1, DT)
    ten_day_loss = loss(stock10, DAYS * DT)

    var1 = empirical_var(one_day_loss)
    var10 = empirical_var(ten_day_loss)
    scaled = np.sqrt(DAYS) * var1
    return {
        "parameters": {
            "S0": S0,
            "strike": STRIKE,
            "maturity_years": MATURITY,
            "risk_free_rate": RATE,
            "volatility": VOLATILITY,
            "physical_drift": DRIFT,
            "day_years": DT,
            "puts": PUTS,
            "alpha": ALPHA,
            "paths": PATHS,
            "days": DAYS,
            "seed": SEED,
        },
        "initial_put_price": float(price0),
        "initial_put_delta": float(delta0),
        "initial_stock_shares": PUTS * float(delta0),
        "initial_cash": -PUTS * float(delta0) * S0,
        "initial_portfolio_value": PUTS * value0,
        "one_day_var": var1,
        "sqrt_time_ten_day_var": float(scaled),
        "aggregated_ten_day_var": var10,
        "difference_aggregated_minus_scaled": float(var10 - scaled),
        "difference_percent_of_scaled": float(100 * (var10 / scaled - 1)),
        "one_day_loss_mean": float(np.mean(one_day_loss)),
        "one_day_loss_std": float(np.std(one_day_loss, ddof=1)),
        "ten_day_loss_mean": float(np.mean(ten_day_loss)),
        "ten_day_loss_std": float(np.std(ten_day_loss, ddof=1)),
        "empirical_quantile": "sorted loss at ceil(alpha * paths), no interpolation",
        "aggregation_convention": "aggregate ten daily log returns; hold initial delta fixed",
    }


def main():
    results = simulate()
    folder = Path(__file__).resolve().parent / "work"
    folder.mkdir(exist_ok=True)
    path = folder / "var_10day_results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name in (
        "one_day_var",
        "sqrt_time_ten_day_var",
        "aggregated_ten_day_var",
        "difference_aggregated_minus_scaled",
    ):
        print(f"{name}: ${results[name]:.4f}")
    print(f"Results: {path}")


if __name__ == "__main__":
    main()
