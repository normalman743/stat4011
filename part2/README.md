# Part 2 README

This folder contains the crime-analysis workflow: data cleaning, EDA/visualization, and sequence models (Transformer/LSTM) for daily crime prediction.

## Data not committed
- `orginaldata/` (raw CSVs) — large source files.
- `cleaned_data/` — processed outputs derived from the raw data.
All processing code is included so the data can be regenerated locally.

## What each script does
- `datacleaning.py` / `preprocess_to_daily_csv.py`: clean raw crime data and reshape to daily-level CSVs; writes into `cleaned_data/`.
- `data_analysis.py`, `analysis.py`, `clean_analysis.py`, `eda_analysis.py`: exploratory stats, feature inspection, and sanity checks over the cleaned datasets.
- `crime_analysis.py` / `crime_analysis_enhanced.py`: feature engineering and classical model experiments on the crime data.
- `crime_visualization.py`: plotting utilities; outputs to `crime_visualization_output/` and static PNGs (`crime_heatmap.png`, `crime_scatter.png`, `output.png`).
- `data_enhancement_output/`, `analysis_output*/`: generated reports/figures from EDA and model analysis.
- `loc.py`: location-related helpers used by analysis/visualization scripts.
- `transformer_train_v2.py`, `transformer_train_predict.py`, `predict_transformer_v2.py`: Transformer training and inference pipeline for time-series/daily crime counts.
- `lstm_basic.py`: baseline LSTM model for comparison.
- `STAT4011 Project 2.pdf`, `peer review.docx`: project write-up and peer review form (kept for reference; not code).

## Reproducing locally (outline)
1) Place raw CSVs in `orginaldata/` (see filenames mentioned in scripts).
2) Run `datacleaning.py` then `preprocess_to_daily_csv.py` to populate `cleaned_data/`.
3) Use EDA scripts (`data_analysis.py`, `analysis.py`, etc.) to inspect distributions and quality.
4) Train/evaluate models with `transformer_train_v2.py` or `lstm_basic.py`; use `transformer_train_predict.py` for inference.

## Notes
- Virtual environment artifacts (`bin/`, `lib/`, `include/`, `share/`, `pyvenv.cfg`) are ignored; create your own env before running.
- Large generated outputs under `analysis_output*`, `crime_visualization_output/`, `data_enhancement_output/` are not versioned; regenerate as needed.
