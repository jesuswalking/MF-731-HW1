"""MF 731 HW3: historical and normal-EWMA one-day VaR backtests.

Run: python3 var_backtest.py
Use --report to rebuild the PDF with pdflatex. The cached S&P 500 Close
data in data/sp500_close.csv make the calculation reproducible offline.
Requires numpy, pandas, scipy and matplotlib.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig-backtest"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ALPHA, BETA, LAMBDA, THETA, M = 0.95, 0.05, 0.97, 0.97, 1010
START, END = "2016-05-23", "2026-05-22"
CODE_URL = "https://github.com/jesuswalking/MF-731-HW1/tree/main/HW3"


def read_prices(path):
    """Read unadjusted daily Close values, without filling market holidays."""
    frame = pd.read_csv(path, parse_dates=["Date"])
    if list(frame.columns) != ["Date", "Close"]:
        raise ValueError("The price CSV must have exactly Date,Close columns.")
    dates = frame["Date"]
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and in increasing order.")
    prices = frame["Close"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Every closing price must be finite and positive.")
    if dates.iloc[0].strftime("%Y-%m-%d") != START or dates.iloc[-1].strftime("%Y-%m-%d") != END:
        raise ValueError("The data must include the assignment's two endpoints.")
    if len(prices) <= M + 1:
        raise ValueError("Not enough prices for the initial window and a backtest.")
    return frame


def backtest(frame, window=M):
    """Forecast each return before observing it, then update the EWMA state.

    N prices give N-1 returns. The first `window` returns are training data;
    both methods evaluate the remaining N-1-window returns on the same dates.
    Historical VaR is the inverse empirical CDF of full losses, not a
    linear interpolation between order statistics.
    """
    closes = frame["Close"].to_numpy(dtype=float)
    returns = np.diff(np.log(closes))
    if window < 2 or window >= len(returns):
        raise ValueError("The rolling window must leave at least one test return.")
    mu = float(np.mean(returns[:window]))
    variance = float(np.var(returns[:window], ddof=1))
    initial = {"mean": mu, "variance": variance, "sd": np.sqrt(variance)}
    z = float(norm.ppf(1 - ALPHA))
    rank = int(np.ceil(ALPHA * window)) - 1
    rows = []
    for i in range(window, len(returns)):
        # The window ends at i-1: the return on the evaluation date is excluded.
        past_losses = -np.expm1(returns[i - window:i])
        historical_var = float(np.partition(past_losses, rank)[rank])
        normal_var = float(-np.expm1(mu + np.sqrt(variance) * z))
        loss = float(-np.expm1(returns[i]))
        rows.append({
            "Date": frame["Date"].iloc[i + 1],
            "LogReturn": float(returns[i]), "Loss": loss,
            "HistoricalVaR": historical_var, "NormalEWMAVaR": normal_var,
            "HistoricalExceedance": bool(loss > historical_var),
            "NormalEWMAExceedance": bool(loss > normal_var),
            "ForecastMean": mu, "ForecastVariance": variance,
        })
        # Both recursions use the state that existed before this return.
        variance = THETA * variance + (1 - THETA) * (returns[i] - mu) ** 2
        mu = LAMBDA * mu + (1 - LAMBDA) * returns[i]
    return pd.DataFrame(rows), initial


def summarize(frame, series, initial, data_path):
    periods = len(series)
    interval = binom.ppf([BETA / 2, 1 - BETA / 2], periods, 1 - ALPHA).astype(int)
    methods = {}
    for label, column in [("Historical", "HistoricalExceedance"),
                          ("Normal EWMA", "NormalEWMAExceedance")]:
        count = int(series[column].sum())
        methods[label] = {"exceedances": count, "rate": count / periods,
                          "within_interval": bool(interval[0] <= count <= interval[1])}
    observations = frame.iloc[np.r_[0:5, len(frame) - 5:len(frame)]].copy()
    observations["Date"] = observations["Date"].dt.strftime("%Y-%m-%d")
    return {
        "parameters": {"alpha": ALPHA, "beta": BETA, "lambda": LAMBDA,
                       "theta": THETA, "window": M},
        "data": {"source": "Yahoo Finance, ^GSPC daily Close",
                 "start": START, "end": END, "prices": len(frame),
                 "returns": len(frame) - 1,
                 "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                 "first_and_last_five": observations.to_dict("records")},
        "initial": initial,
        "evaluation": {"periods": periods,
                       "first_date": series["Date"].iloc[0].strftime("%Y-%m-%d"),
                       "last_date": series["Date"].iloc[-1].strftime("%Y-%m-%d"),
                       "expected_exceedances": periods * (1 - ALPHA),
                       "count_interval_95": interval.tolist(),
                       "interval_probability": float(binom.cdf(interval[1], periods, 1 - ALPHA)
                                                     - binom.cdf(interval[0] - 1, periods, 1 - ALPHA)),
                       "methods": methods},
        "code_url": CODE_URL,
    }


def plot_backtests(series):
    """Show both forecast series and every realized exceedance on shared axes."""
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True,
                             sharey=True, layout="constrained")
    dates = series["Date"]
    losses = 100 * series["Loss"]  # cents of loss on the constant $1 position
    for ax, title, var_column, exceed_column in [
        (axes[0], "Historical (rolling 1,010 returns)", "HistoricalVaR", "HistoricalExceedance"),
        (axes[1], "Normal EWMA (0.97)", "NormalEWMAVaR", "NormalEWMAExceedance")]:
        ax.plot(dates, losses, color="#b8bec5", linewidth=0.45,
                alpha=0.8, label="Realized loss")
        ax.plot(dates, 100 * series[var_column], color="#1c5073", linewidth=0.9,
                label="95% VaR")
        exceed = series[exceed_column]
        ax.scatter(dates[exceed], losses[exceed], marker="x", s=11,
                   linewidths=0.7, color="#b63e32", label="Exceedance", zorder=3)
        ax.set_title(title, fontsize=8.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, loc="lower right", fontsize=6.7)
    axes[0].set_ylabel("Loss / VaR (cents per $1)")
    fig.savefig(WORK / "var_backtest_plot.png", dpi=300)
    plt.close(fig)


def write_report_numbers(results):
    evaluation = results["evaluation"]
    lower, upper = evaluation["count_interval_95"]
    lines = [
        rf"\newcommand{{\PriceCount}}{{{results['data']['prices']:,}}}",
        rf"\newcommand{{\ReturnCount}}{{{results['data']['returns']:,}}}",
        rf"\newcommand{{\TestCount}}{{{evaluation['periods']:,}}}",
        rf"\newcommand{{\TestStart}}{{{evaluation['first_date']}}}",
        rf"\newcommand{{\InitialMean}}{{{results['initial']['mean']:.8f}}}",
        rf"\newcommand{{\InitialSD}}{{{results['initial']['sd']:.8f}}}",
        rf"\newcommand{{\ExpectedCount}}{{{evaluation['expected_exceedances']:.1f}}}",
        rf"\newcommand{{\CountLower}}{{{lower}}}",
        rf"\newcommand{{\CountUpper}}{{{upper}}}",
    ]
    records = results["data"]["first_and_last_five"]
    for i in range(5):
        first, last = records[i], records[i + 5]
        lines.append(rf"\newcommand{{\DataRow{chr(65 + i)}}}{{{first['Date']} & {first['Close']:,.2f} & {last['Date']} & {last['Close']:,.2f} \\}}")
    for label, prefix in [("Historical", "Hist"), ("Normal EWMA", "Normal")]:
        method = evaluation["methods"][label]
        lines.extend([
            rf"\newcommand{{\{prefix}Count}}{{{method['exceedances']}}}",
            rf"\newcommand{{\{prefix}Rate}}{{{100 * method['rate']:.2f}\%}}",
            rf"\newcommand{{\{prefix}Decision}}{{{'Inside' if method['within_interval'] else 'Outside'}}}",
        ])
    decisions = []
    for label, method in evaluation["methods"].items():
        count = method["exceedances"]
        description = "too few exceedances" if count < lower else (
            "too many exceedances" if count > upper else "a count inside the interval")
        decisions.append(f"{label} has {description}")
    lines.append(r"\newcommand{\CountConclusion}{" + ". ".join(decisions) + ".}")
    (WORK / "var_backtest_numbers.tex").write_text("\n".join(lines) + "\n")


def compile_report():
    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    job = "MF731_05_VaR_Backtest_Report"
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               f"-jobname={job}", f"-output-directory={output}", "var_backtest_report.tex"]
    for _ in range(2):
        run = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
    for suffix in (".aux", ".log", ".out"):
        (output / f"{job}{suffix}").unlink(missing_ok=True)
    return output / f"{job}.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sp500_close.csv",
                        help="cached Date,Close CSV (default: data/sp500_close.csv)")
    parser.add_argument("--report", action="store_true", help="also build the PDF report")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    prices = read_prices(args.data)
    series, initial = backtest(prices)
    results = summarize(prices, series, initial, args.data)
    series.to_csv(WORK / "var_backtest_series.csv", index=False, date_format="%Y-%m-%d")
    (WORK / "var_backtest_results.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_backtests(series)
    write_report_numbers(results)
    evaluation = results["evaluation"]
    print(f"N={len(prices)} prices; {len(prices) - 1} returns; J={len(series)} test days")
    print(f"Test period: {evaluation['first_date']} to {evaluation['last_date']}")
    print(f"95% binomial count interval: {evaluation['count_interval_95']}")
    for name, method in evaluation["methods"].items():
        status = "inside" if method["within_interval"] else "outside"
        print(f"{name}: {method['exceedances']} exceedances ({100 * method['rate']:.2f}%), {status}")
    if args.report:
        print(f"Report: {compile_report()}")


if __name__ == "__main__":
    main()
