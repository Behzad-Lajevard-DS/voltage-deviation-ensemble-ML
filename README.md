# Thesis pipeline — GitHub-ready reproducible package

This repository contains the end-to-end thesis pipeline for hourly endpoint-level voltage deviation modelling in PV-rich low-voltage distribution networks.

## What the pipeline does

The pipeline automatically:

1. downloads the raw dataset from the original source,
2. constructs the modelling dataset,
3. performs EDA,
4. trains and compares the baseline and ensemble models,
5. runs Optuna tuning,
6. performs ablation analysis,
7. evaluates the final model on the unseen test set,
8. runs robustness checks, residual analysis, operational regime analysis, and SHAP interpretation,
9. saves all outputs locally.

## Main modelling setup

- **Target:** hourly endpoint-level voltage deviation from 230 V
- **Inputs:** customer-level PV generation, feeder/substation-level Load, and weather variables
- **Models:** Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Selection metric:** validation MAE
- **Final selected model in the tested thesis run:** Random Forest

## Reproducibility

This package is based on the final Sprint 6 notebook version.  
A full **Run All** execution of the final notebook was reported to complete successfully in approximately **4 hours and 30 minutes**.

## Repository structure

```text
.
├── run_pipeline.py
├── requirements.txt
├── README.md
├── reproduce_results.md
├── VALIDATION_CHECKS.md
├── .gitignore
├── notebooks/
│   ├── thesis_code_final.ipynb
│   └── thesis_pipeline_clean.ipynb
├── data/
│   └── raw/
├── results/
└── scripts/
    └── run_local.sh
```

## How to run locally

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the full pipeline

```bash
python run_pipeline.py
```

## Output location

By default, outputs are written to:

```text
./thesis_outputs_end_to_end/
```

You can override that with:

```bash
export THESIS_OUTPUT_DIR=/your/output/path
python run_pipeline.py
```

## Raw data download

The pipeline downloads the raw data automatically from the original source.  
You do **not** need to manually place the dataset in the repository.

If you want to reuse an already-downloaded copy, set:

```bash
export THESIS_RAW_DATA_DIR=/your/raw/data/folder
python run_pipeline.py
```

## Notes for reviewers / committee

- The pipeline is designed to run end-to-end.
- Internet access is required for automatic dataset download.
- Runtime may be several hours depending on machine speed.
- All figures, tables, predictions, SHAP outputs, and summary files are saved locally.

## Recommended GitHub usage

Upload this repository as a dedicated thesis repository, rather than mixing it into an unrelated course repository.


## Important note

This repository is intended to be executed locally with standard Python, not as a Google Colab-only project.


## Notebook files

- `notebooks/thesis_pipeline_clean.ipynb` is the review-friendly notebook version.
- `notebooks/thesis_code_final.ipynb` is retained as the original final notebook copy.
