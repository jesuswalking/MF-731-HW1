# MF 731 Homework 1: SPY volatility

Python implementation of the three assignment tasks:

1. Annualized MA (100 trading days) and EWMA (lambda = 0.94), August 2021 through July 2026.
2. Gaussian GARCH(1,1) estimation using closing prices from June 2025 through May 2026.
3. 100 simulated volatility paths for the 42 trading days from June 1 through July 30, 2026.

## Run

Use Python 3.13 (tested with Python 3.13.7).

```bash
python -m pip install -r requirements.txt
python solution.py
```

The script writes `figures/ma_ewma.pdf`, `figures/forecast_paths.pdf`, `results.json`, `daily_forecasts.csv`, `100_paths.csv`, and `spy_close.csv` beside the script.

## Data and reproducibility

`spy_yahoo_raw.json` contains the public Yahoo Finance SPY data snapshot used for the submitted results, downloaded September 15, 2026. Including this cache allows the analysis to run without fetching price data or signing into a market-data account. Installing dependencies requires internet access.

Data source: [SPY historical prices](https://finance.yahoo.com/quote/SPY/history/).
Exact query: <https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=1609459200&period2=1785456000&interval=1d>.

SHA-256 of the cached response:
`d564c68f4527de1fc0c9c0cdc7c69772b21c0f1d4e325a46eb3ba14c95f23e9f`.

If the cache is removed, the script downloads the same date range from Yahoo Finance. Historical data may subsequently be revised or requests rate-limited.

## Modeling conventions

- Use unadjusted `Close` and daily logarithmic returns.
- MA uses the mean of the preceding 100 squared returns under a zero-mean approximation.
- EWMA starts at the sample variance of the preceding 100 returns.
- An estimate labeled day t uses observed returns through day t-1.
- The GARCH price sample is restricted before differencing, yielding 249 returns from 250 closing prices.
- GARCH fits a constant mean and daily returns in percentage points, using constrained Gaussian maximum likelihood and six initial parameter vectors.
- GARCH forecasts have a fixed origin of May 29, 2026; observed June and July returns are not used to generate the paths.
- Annualization uses 252 trading days; the random seed is 731.

## Reference results

| Parameter | Estimate |
| --- | ---: |
| mu | 0.09872994 |
| omega | 0.07069051 |
| alpha | 0.06079139 |
| beta | 0.81325360 |

Long-run annualized volatility: 11.8925%. Analytical annualized forecasts: 10.7453% on June 1 and 11.8881% on July 30, 2026.

The simulation bands are conditional on fitted parameters. MA and EWMA update from observed returns while the GARCH sample paths are fixed-origin simulations; this is a descriptive comparison, not a formal forecast-accuracy test.
