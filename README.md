# Seasonal clock model

Code, data, and figures accompanying the seasonal dual-oscillator manuscript. Manuscript LaTeX sources are not included.

## Setup

Install Python 3.13 or later and uv, then run from this directory:

```sh
uv sync --locked
uv run python code/build_figure5.py
uv run python code/build_supplementary_figures.py
uv run python code/build_report_assets.py
```

## Contents

- `code/phase_model.py`: four-parameter mean-phase and phase-difference model; `TUNED` contains the manuscript parameters.
- `code/reference_phase_model.py` and `code/fit_reference_phase.py`: reference-model equations and target-loading/fitting utilities used in model comparisons.
- `code/dual_oscillator_minimal.py` and `code/dual_oscillator_pdf.py`: oscillator helpers and experimental target tables.
- `code/build_figure5.py`: Figure 5 panels, plotted values, and source-data workbook.
- `code/build_supplementary_figures.py`: supplementary figures, using saved comparison results and identifiability images.
- `code/build_report_assets.py`: comparison figures and a file-checksum manifest.
- `experimental_phase_vs_daylength.csv`: experimental peak-time data.
- `out/reference/`: observed waveforms and saved reference fits.
- `out/model_comparison/`: saved model comparison and validation results.
- `out/identifiability/`: saved identifiability images.
- `figure/`: manuscript figures and source data.

Figure reproduction uses saved fitted parameters and results. The complete optimization and identifiability workflows are not included in this manuscript code distribution; identifiability panels are reproduced from the supplied images.

All commands and paths are relative to the repository root.
