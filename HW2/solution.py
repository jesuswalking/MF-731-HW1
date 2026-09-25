"""MF 731 HW2: loss on a fixed delta hedge of 100 short puts.

Run: python3 solution.py
Requires numpy, scipy and matplotlib. No data download is needed.
The simulation saves its figure and numerical results under work/.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

# Keep Matplotlib's cache beside the generated files.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


MU, SIGMA, R = 0.16905, 0.4907, 0.0011888
T, T0, HORIZON = 0.291667, 0.0, 10 / 252
S0, K, M = 152.51, 170.0, 100
N, SEED = 100_000, 731
WORK = ROOT / "work"


def put_price(t, stock):
    """Black-Scholes put price; stock can be a scalar or an array."""
    stock = np.asarray(stock, dtype=float)
    tau = T - t
    if tau < 0:
        raise ValueError("The valuation date must not exceed maturity.")
    if np.any(stock <= 0):
        raise ValueError("Stock prices must be positive.")
    if tau == 0:
        return np.maximum(K - stock, 0.0)
    d1 = (np.log(stock / K) + (R + 0.5 * SIGMA**2) * tau) / (
        SIGMA * np.sqrt(tau)
    )
    d2 = d1 - SIGMA * np.sqrt(tau)
    # The assignment's N(d1)-2 is a typo; the payoff implies N(d1)-1.
    return K * np.exp(-R * tau) * ndtr(-d2) - stock * ndtr(-d1)


def put_greeks(t, stock):
    """Return delta, gamma and calendar-time theta of one put."""
    tau = T - t
    if tau <= 0 or stock <= 0:
        raise ValueError("Greeks require positive stock and time to maturity.")
    d1 = (np.log(stock / K) + (R + 0.5 * SIGMA**2) * tau) / (
        SIGMA * np.sqrt(tau)
    )
    d2 = d1 - SIGMA * np.sqrt(tau)
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    delta = ndtr(d1) - 1
    gamma = phi / (stock * SIGMA * np.sqrt(tau))
    theta = -stock * SIGMA * phi / (2 * np.sqrt(tau))
    theta += R * K * np.exp(-R * tau) * ndtr(-d2)
    return float(delta), float(gamma), float(theta)


def loss_operators(x, horizon=HORIZON):
    """Loss = V(t) - V(t+horizon), with the initial hedge held fixed.

    M is the number of single-unit options in the given portfolio.
    There is no extra contract multiplier or financing account.
    """
    if not 0 <= horizon <= T - T0:
        raise ValueError("The loss horizon must lie before or at maturity.")
    x = np.asarray(x, dtype=float)
    delta, gamma, theta = put_greeks(T0, S0)
    full = M * (
        put_price(T0 + horizon, S0 * np.exp(x))
        - put_price(T0, S0)
        - delta * S0 * np.expm1(x)
    )
    linear = np.full_like(x, M * theta * horizon)
    second = linear + 0.5 * M * gamma * S0**2 * x**2
    return {"Full": full, "Linear": linear, "Second order": second}


def simulate():
    """Use physical-measure returns and common draws for all operators."""
    rng = np.random.default_rng(SEED)
    mean_x = (MU - 0.5 * SIGMA**2) * HORIZON
    sd_x = SIGMA * np.sqrt(HORIZON)
    x = rng.normal(mean_x, sd_x, N)
    return x, loss_operators(x)


def summarize(losses):
    summary = {}
    for name, values in losses.items():
        quantiles = np.quantile(values, [0.5, 0.95, 0.99])
        summary[name] = {
            "mean": float(np.mean(values)),
            "sd": 0.0 if name == "Linear" else float(np.std(values, ddof=1)),
            "median": float(quantiles[0]),
            "q95": float(quantiles[1]),
            "q99": float(quantiles[2]),
            "rmse_vs_full": float(np.sqrt(np.mean((values - losses["Full"]) ** 2))),
        }
    return summary


def plot_losses(losses):
    """Histogram densities with common $10 bins, including all draws."""
    full, second = losses["Full"], losses["Second order"]
    a = float(losses["Linear"][0])
    lower = 10 * np.floor(min(full.min(), second.min(), a) / 10)
    upper = 10 * np.ceil(max(full.max(), second.max(), a) / 10) + 10
    edges = np.arange(lower, upper + 10, 10)
    full_density, _ = np.histogram(full, bins=edges, density=True)
    second_density, _ = np.histogram(second, bins=edges, density=True)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75),
                             gridspec_kw={"width_ratios": [1.5, 1]},
                             layout="constrained")
    for ax in axes:
        ax.stairs(full_density, edges, fill=True, alpha=0.30,
                  color="#19557b", label="Full")
        ax.stairs(second_density, edges, color="#bf6c25", linewidth=1.1,
                  label="Second order")
        ax.set_xlabel("Portfolio loss ($)")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.18)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)

    axes[0].axvline(a, color="#333333", linestyle="--", linewidth=1,
                    label="Linear: point mass")
    axes[0].set_xlim(lower - 10, upper)
    axes[0].set_title("Ten-day loss distribution")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xlim(200, 1000)
    tail = (edges[:-1] >= 200) & (edges[:-1] < 1000)
    axes[1].set_ylim(0, 1.1 * max(full_density[tail].max(),
                                second_density[tail].max()))
    axes[1].set_title("Right tail (same densities)")
    fig.savefig(WORK / "loss_distributions.png", dpi=300)
    plt.close(fig)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    x, losses = simulate()
    summary = summarize(losses)
    delta, gamma, theta = put_greeks(T0, S0)
    results = {
        "parameters": {"mu": MU, "sigma": SIGMA, "r": R, "t": T0,
                       "T": T, "horizon": HORIZON, "S0": S0, "K": K,
                       "M": M, "N": N, "seed": SEED},
        "initial": {"put_price": float(put_price(T0, S0)), "delta": delta,
                    "gamma": gamma, "theta": theta,
                    "linear_constant": M * theta * HORIZON,
                    "quadratic_coefficient": 0.5 * M * gamma * S0**2},
        "log_return": {"mean": (MU - 0.5 * SIGMA**2) * HORIZON,
                       "sd": SIGMA * np.sqrt(HORIZON)},
        "summary": summary,
    }
    (WORK / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_losses(losses)

    print(f"{N:,} trials; seed = {SEED}; losses in dollars\n")
    print(f"{'Method':<15}{'Mean':>11}{'SD':>11}{'95%':>11}{'99%':>11}{'RMSE':>11}")
    for name, row in summary.items():
        print(f"{name:<15}{row['mean']:>11.2f}{row['sd']:>11.2f}"
              f"{row['q95']:>11.2f}{row['q99']:>11.2f}"
              f"{row['rmse_vs_full']:>11.2f}")


if __name__ == "__main__":
    main()
