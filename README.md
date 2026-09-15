# MF 731 Homework 1: SPY volatility

The complete Python implementation, data snapshot, dependencies, and detailed
instructions are in [`github/mf731-hw1`](github/mf731-hw1).

## Run

With Python 3.13 installed, run from the repository root:

```bash
cd github/mf731-hw1
python -m pip install -r requirements.txt
python solution.py
```

The program estimates MA and EWMA volatility, fits a GARCH(1,1) model,
and generates 100 volatility paths for June 1 through July 30, 2026.
The included public price-data snapshot reproduces the results in the submitted PDF.
