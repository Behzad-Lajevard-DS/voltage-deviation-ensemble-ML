# Reproduce the thesis results

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

## Expected behavior

The script should:

1. download the raw PV and weather data,
2. construct the cleaned modelling dataset,
3. save dataset files,
4. produce EDA outputs,
5. run model selection and final testing,
6. save:
   - validation model comparison
   - final test comparison
   - ablation results
   - predictions and residuals
   - bootstrap confidence interval
   - operational regime analysis
   - SHAP outputs
   - final summary

## Important environment variables

- `THESIS_OUTPUT_DIR`
- `THESIS_RAW_DATA_DIR`
- `THESIS_MOUNT_DRIVE`
- `THESIS_DRIVE_OUTPUT_DIR`

## Review note

This repository keeps the original final notebook in `notebooks/` for transparency and walkthrough purposes, while `run_pipeline.py` provides a script-style execution path for reproducibility.
