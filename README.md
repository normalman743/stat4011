# STAT4011 Project Workspace

Two main parts:
- **part1**: Account classification experiments (ensemble strategies, feature extraction, model tuning). See subfolders (Qi Zihan, v1–v7) for detailed code and CSV outputs.
- **part2**: Crime analysis and time-series modeling (EDA, cleaning, visualization, Transformer/LSTM training). See `part2/README.md` for usage details.

## Data policy
Large data directories are excluded from version control:
- `part2/orginaldata/` and `part2/cleaned_data/` (raw and processed crime datasets)
- Virtual environment artifacts (`**/bin/`, `**/lib/`, `**/include/`, `**/share/`, `pyvenv.cfg`, `.venv`, `venv/`)

All processing and modeling code is included so datasets and outputs can be regenerated locally.
