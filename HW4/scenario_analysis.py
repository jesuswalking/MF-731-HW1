"""MF 731, exercise 09: five-day scenarios for a delta-hedged option.

Run: python3 scenario_analysis.py
Only the Python standard library is needed.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


S1, S2 = 158.12, 170.33
SIGMA1, SIGMA2, RHO = 0.2214, 0.3036, 0.237
R, T, HORIZON = 0.0132, 0.25, 5 / 252

RETURNS = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
VOL_FACTORS = (0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
RHO_SHIFTS = (-0.5, 0.0, 0.5)
RETURN_WEIGHTS = {0.20: 0.5, 0.10: 0.75, 0.05: 1.0}
VOL_WEIGHTS = {0.5: 0.5, 0.75: 1.25, 1.25: 0.75,
               1.5: 0.75, 1.75: 1.25, 2.0: 0.5}
RHO_WEIGHTS = {-0.5: 1.25, 0.0: 0.5, 0.5: 1.25}


def option_price(s1, s2, sigma1, sigma2, rho, tau):
    """Discounted Q-expectation of (S1_T - S2_T)^2."""
    return math.exp(R * tau) * (
        s1**2 * math.exp(sigma1**2 * tau)
        - 2 * s1 * s2 * math.exp(sigma1 * sigma2 * rho * tau)
        + s2**2 * math.exp(sigma2**2 * tau)
    )


def option_deltas(s1, s2, sigma1, sigma2, rho, tau):
    a = math.exp(sigma1**2 * tau)
    b = math.exp(sigma1 * sigma2 * rho * tau)
    c = math.exp(sigma2**2 * tau)
    prefactor = math.exp(R * tau)
    return (2 * prefactor * (s1 * a - s2 * b),
            2 * prefactor * (s2 * c - s1 * b))


def calculate():
    price0 = option_price(S1, S2, SIGMA1, SIGMA2, RHO, T)
    h1, h2 = option_deltas(S1, S2, SIGMA1, SIGMA2, RHO, T)
    portfolio0 = price0 - h1 * S1 - h2 * S2
    rows = []

    grid = itertools.product(RETURNS, RETURNS, VOL_FACTORS,
                             VOL_FACTORS, RHO_SHIFTS)
    for x1, x2, factor1, factor2, rho_shift in grid:
        s1, s2 = S1 * math.exp(x1), S2 * math.exp(x2)
        sigma1, sigma2 = SIGMA1 * factor1, SIGMA2 * factor2
        rho = RHO + rho_shift
        price = option_price(s1, s2, sigma1, sigma2, rho, T - HORIZON)

        # The handout defines this portfolio without a cash account.
        # Hold the original stock positions -h1 and -h2 for all five days.
        portfolio = price - h1 * s1 - h2 * s2
        loss = portfolio0 - portfolio
        weight = (RETURN_WEIGHTS[abs(x1)] * RETURN_WEIGHTS[abs(x2)]
                  * VOL_WEIGHTS[factor1] * VOL_WEIGHTS[factor2]
                  * RHO_WEIGHTS[rho_shift])
        rows.append({
            "x1": x1, "x2": x2,
            "vol_factor1": factor1, "vol_factor2": factor2,
            "rho_shift": rho_shift,
            "sigma1": sigma1, "sigma2": sigma2, "rho": rho,
            "s1": s1, "s2": s2,
            "option_price": price, "portfolio_value": portfolio,
            "loss": loss, "weight": weight, "weighted_loss": weight * loss,
        })

    worst = max(rows, key=lambda row: row["loss"])
    weighted_worst = max(rows, key=lambda row: row["weighted_loss"])
    results = {
        "parameters": {
            "s1": S1, "s2": S2, "sigma1": SIGMA1, "sigma2": SIGMA2,
            "rho": RHO, "r": R, "T": T, "horizon": HORIZON,
        },
        "scenario_count": len(rows),
        "initial": {
            "option_price": price0, "h1": h1, "h2": h2,
            "stock_position1": -h1, "stock_position2": -h2,
            "portfolio_value": portfolio0,
        },
        "worst_case": worst,
        "weighted_worst_case": weighted_worst,
    }
    return rows, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "work",
        help="Where to save the CSV and JSON results (default: work beside script).",
    )
    args = parser.parse_args()
    rows, results = calculate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scenarios.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "scenario_results.json").open("w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")

    print(f"Scenarios: {len(rows)}")
    initial = results["initial"]
    print(f"Initial option price: {initial['option_price']:.6f}")
    print(f"Initial deltas: h1={initial['h1']:.6f}, h2={initial['h2']:.6f}")
    print(f"Initial portfolio value: {initial['portfolio_value']:.6f}")
    for title, key in (("(b) Worst loss", "worst_case"),
                       ("(c) Worst weighted loss", "weighted_worst_case")):
        row = results[key]
        print(f"\n{title}")
        print(f"  x1={row['x1']:+.2f}, x2={row['x2']:+.2f}")
        print(f"  Volatility factors: {row['vol_factor1']}, {row['vol_factor2']}")
        print(f"  Volatilities: {row['sigma1']:.6f}, {row['sigma2']:.6f}")
        print(f"  Correlation: {row['rho']:.3f} (shift {row['rho_shift']:+.1f})")
        print(f"  Portfolio loss: {row['loss']:.6f}")
        print(f"  Weight: {row['weight']:.6f}")
        print(f"  Weighted loss: {row['weighted_loss']:.6f}")


if __name__ == "__main__":
    main()
