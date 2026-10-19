# MF 731 Homework 4

- `07`: proofs for the entropic risk measure, in `output/pdf/MF731_07_Risk_Solutions.pdf`.
- `09`: `scenario_analysis.py` and the short report `output/pdf/MF731_09_Scenario_Analysis_Report.pdf`.

Run exercise 09 with Python 3 (no external packages):

```bash
python3 scenario_analysis.py
```

The script enumerates all 3,888 scenarios. It saves every scenario to
`work/scenarios.csv` and the summary to `work/scenario_results.json`.
The default output path is beside the script, so it can be run from another
directory. To choose a different output directory:

```bash
python3 scenario_analysis.py --output-dir /tmp/mf731-hw4-results
```

The initial stock holdings are minus the two option deltas and stay fixed
over five days. The option is repriced at maturity `0.25 - 5/252` with each
scenario's new stock prices, volatilities and correlation. The portfolio is
defined exactly as in the handout. The weights are products of the five
given weights and are not normalized. The supplied physical drifts are not
needed because the stock returns are prescribed by the scenario grid.

| Result | Value |
| --- | ---: |
| Initial option price | 921.739866 |
| Initial deltas h1, h2 | -21.952543, 31.201877 |
| Worst portfolio loss | 1,000.361717 |
| Worst weighted loss | 1,313.785913 |
| Actual loss at the weighted maximizer | 672.658388 |
| Weight at the weighted maximizer | 1.953125 |

The worst loss has log returns `(0.20, 0.20)`, volatility factors `(0.5, 0.5)`
and correlation shift `+0.5`. The worst weighted loss has log returns
`(0.05, 0.05)`, volatility factors `(0.75, 0.75)` and correlation shift `+0.5`.

LaTeX sources are in `work/`. To rebuild the PDFs with TeX Live:

```bash
mkdir -p output/pdf
cd work
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MF731_07_Risk_Solutions -output-directory=../output/pdf risk_solutions.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MF731_09_Scenario_Analysis_Report -output-directory=../output/pdf scenario_report.tex
```

Code: https://github.com/jesuswalking/MF-731-HW1/tree/main/HW4
