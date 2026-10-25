# MF 731 Homework 5 - Stress Testing

Exercise 09: a $1,000,000 market-cap-weighted MSFT/AAPL portfolio, with a
deterministic five-sigma first-day shock and path-specific EWMA updates.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 stress_testing.py
```

Python 3.9+ and NumPy 1.22+ are required. The submitted run used Python
3.13.7 and NumPy 2.4.4, with 50,000 paths and seed 731. Historical prices
are included, so no network access is needed to run the calculation.
The script resolves its default data and output directories relative to
its own location, including when launched from another directory.

Outputs are `work/stress_results.json` and `work/stressed_losses.csv`.
To choose another output directory or seed:

```bash
python3 stress_testing.py --output-dir /tmp/mf731-hw5-results --seed 731
```

## Results

| Quantity | Estimate |
| --- | ---: |
| Initial one-day 95% VaR | $14,383.87 |
| Initial ten-day VaR, square-root rule | $45,485.80 |
| Initial capital, three times ten-day VaR | $136,457.39 |
| (a) Mean stressed ten-day loss | $70,838.04 |
| (b) Stressed ten-day 95% VaR | $151,692.60 |
| (c) Exceed initial ten-day VaR | 69.91% (34,955 / 50,000) |
| (d) Exceed initial capital | 8.89% (4,445 / 50,000) |

The one-page report is `output/pdf/MF731_09_Stress_Testing_Report.pdf`.

## Data and calculation conventions

- Stock order is MSFT, AAPL. The supplied market caps, $448.77 and $575.11
  billion, determine fixed dollar positions of $438,303.3168 and $561,696.6832.
- `data/daily_close.csv` contains Yahoo Finance Close prices, adjusted for
  splits but not dividends. There are 1,260 common price dates, August 31,
  2011 through September 1, 2016. The preceding close makes the first of
  the 1,259 log returns fall on September 1, 2011.
- `data/manifest.json` records source URLs, download date, counts and SHA-256
  hashes. Raw Yahoo JSON snapshots are retained locally in `data/`; the
  supplied CSV and manifest are sufficient to reproduce the submission.
- Both EWMA states start at zero. Mean and covariance decay are 0.97;
  covariance residuals use the mean before the corresponding update.
- Initial one-day normal VaR is `-positions @ mean + z95 * portfolio_sd`.
  The ten-day benchmark is exactly `sqrt(10) * one_day_var`, as requested.
- Apple is shocked to its marginal mean minus five marginal standard
  deviations. Microsoft is shocked to its conditional mean minus five
  conditional standard deviations. Neither first-day return is sampled.
- Each path starts with this same shock, then simulates days 2 through 10
  with its own evolving mean and covariance. Losses use the initial dollar
  positions and summed log returns, as specified by the linearization.
- Empirical VaR is the 47,500th ordered loss; exceedances use strict `>`.
  Each reported frequency is a percentage of the 50,000 ten-day paths.

The results JSON also reports independent moment checks. If `mu2` and
`Sigma2` are the post-shock states and `x` is the first-day shock, then
the expected loss is `-positions @ (x + 9 * mu2)` = $70,802.45.
The exact loss variance is `11.3436 * (positions @ Sigma2 @ positions)`.
The simulated mean differs from its expectation by 0.16 standard errors.

## Rebuild the report

With TeX Live installed, run from this folder:

```bash
mkdir -p output/pdf tmp/pdfs
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MF731_09_Stress_Testing_Report -output-directory=tmp/pdfs work/stress_report.tex
cp tmp/pdfs/MF731_09_Stress_Testing_Report.pdf output/pdf/
```

The report describes the default run; changing the seed or other simulation
settings changes the Monte Carlo estimates.

Code: https://github.com/jesuswalking/MF-731-HW1/tree/main/HW5
