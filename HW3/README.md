# MF 731 Homework 3

Three separate programs for the three VaR exercises. Tested with Python 3.13.7.

```bash
python3 -m pip install -r requirements.txt
python3 var_est.py
python3 var_10day.py
python3 var_backtest.py
```

All paths are relative to each script, so the programs can also be launched
from another directory. The included CSV snapshots reproduce the reports
without another download. Results are saved under `work/`.

| Program | Calculation |
| --- | --- |
| `var_est.py` | Historical and normal/EWMA VaR for MSFT, AAPL and GOOGL; 100,000 normal draws |
| `var_10day.py` | One-day and ten-day VaR for 100 delta-hedged puts; 40,000 stock paths |
| `var_backtest.py` | Rolling historical and normal/EWMA S&P 500 VaR, exceedances and count interval |

Both simulations use NumPy's `default_rng` and seed 731. All sample VaRs use
the inverse empirical CDF rather than interpolating between observations.
There is no extra option contract multiplier. The ten-day exercise holds
the initial hedge fixed and compounds its financing cash account.

## Data

`data/` contains Yahoo Finance daily unadjusted Close prices. The stock
sample is May 23, 2024 through May 22, 2026; the S&P 500 (`^GSPC`) sample
is May 23, 2016 through May 22, 2026. Both endpoints are included.
`data/manifest.json` records query URLs and SHA-256 hashes. The three-stock
program downloads missing data; the backtest uses its supplied CSV.

The stock portfolio uses the market capitalizations supplied in the handout.
EWMA covariance uses the mean before the current observation updates it.
Backtest forecasts exclude the return they predict: 2,515 prices give
2,514 returns and 1,504 test days after the first 1,010 training returns.
May 25, 2026 is Memorial Day, so the stock VaR is interpreted as a forecast
for the next trading session, May 26, without extra time scaling.

## Reports

The three reports are one page each; the ES worked solutions are two pages.
Report sources are in `work/`. With `pdflatex` installed, the backtest
report can be rebuilt using `python3 var_backtest.py --report`.
To compile either of the other VaR report sources, run from `work/`:

```bash
mkdir -p ../output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf var_est_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf var_10day_report.tex
```

Reference results in dollars:

| Exercise | Result |
| --- | --- |
| Stock portfolio, full historical / normal VaR | 21,043.05 / 16,609.78 |
| Put portfolio, one-day / scaled ten-day / aggregated ten-day VaR | 30.14 / 95.32 / 308.28 |
| Backtest, historical / normal-EWMA exceedances | 55 / 94; 95% interval [59, 92] |

The ES handout's strict-monotonicity assertion in Question 4 is false as
stated; the PDF gives a counterexample and the valid weak inequality.

Code repository: https://github.com/jesuswalking/MF-731-HW1/tree/main/HW3
