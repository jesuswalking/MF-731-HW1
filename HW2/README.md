# MF 731 Homework 2: Loss operators

Full revaluation, linearized and second-order losses for a fixed delta hedge of 100 short puts over ten trading days.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 solution.py
```

The script saves numerical results to `work/results.json` and the loss-distribution figure to `work/loss_distributions.png`. No data download is needed.

## Reproducibility

The simulation uses 100,000 physical-measure log returns, NumPy's `default_rng` and seed 731. All three loss operators use the same draws, with the initial delta hedge held fixed.
