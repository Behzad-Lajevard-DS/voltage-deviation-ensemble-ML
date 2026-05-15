#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end thesis pipeline exported from thesis_code_final.ipynb.

This script:
1. Downloads the raw UK Power Networks PV and weather data
2. Builds the final modelling datasets
3. Runs EDA, model training, Optuna tuning, ablation, robustness checks,
   residual analysis, operational regime analysis, and SHAP interpretation
4. Saves all outputs locally

Default output folder:
  ./thesis_outputs_end_to_end

Useful environment variables:
  THESIS_OUTPUT_DIR      -> override output directory
  THESIS_RAW_DATA_DIR    -> override raw data folder (default: data/raw)
"""

from __future__ import annotations

# Fallback display function for plain Python execution
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(obj):
        try:
            if hasattr(obj, 'to_string'):
                print(obj.to_string())
            else:
                print(obj)
        except Exception:
            print(obj)


# ===== Notebook code cell 2 =====
# ============================================================
# END-TO-END THESIS PIPELINE
# Raw Data -> Final Dataset -> EDA -> Optuna -> Ablation -> Test Comparison -> Robustness -> Residuals -> SHAP
# Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
# No CatBoost
# No Substation
# No Network Endpoints
# ============================================================


# ===== Notebook code cell 4 =====
# ============================================================
# 0. Install packages
# ============================================================

# Package installation was removed from the exported script.
# Install dependencies first with: pip install -r requirements.txt


# ===== Notebook code cell 6 =====
# ============================================================
# 1. Import libraries
# ============================================================

import os
import sys
import time
import shutil
import random
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IN_COLAB = False

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna
import shap

# Global reproducibility configuration.
# For exact hash-level reproducibility, restart the runtime after setting PYTHONHASHSEED.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Reproducibility seed:", SEED)
print("Python version:", sys.version.split()[0])
print("Platform:", platform.platform())


# ===== Notebook code cell 8 =====
# ============================================================
# 2. Execution mode
# ============================================================

print("Local execution mode is active.")


# ===== Notebook code cell 10 =====
# ============================================================
# 3. Output folders
# ============================================================

base_output_dir = os.environ.get(
    "THESIS_OUTPUT_DIR",
    os.path.join(os.getcwd(), "thesis_outputs_end_to_end")
)

dataset_output_dir = os.path.join(base_output_dir, "dataset_outputs")
eda_output_dir = os.path.join(base_output_dir, "eda_outputs")
model_output_dir = os.path.join(base_output_dir, "model_outputs")
shap_output_dir = os.path.join(base_output_dir, "shap_outputs")

for folder in [base_output_dir, dataset_output_dir, eda_output_dir, model_output_dir, shap_output_dir]:
    os.makedirs(folder, exist_ok=True)

print("Outputs will be saved to:")
print(base_output_dir)


# ===== Notebook code cell 13 =====
# ============================================================
# 4. Automatically download raw data and define file paths
# ============================================================

import os
import zipfile
import requests
from pathlib import Path

# London Datastore direct download links.
PV_DATA_URL = "https://data.london.gov.uk/download/2nlqm/81fb6b31-f6b2-4e12-b054-090319faec7b/PV%20Data.zip"
WEATHER_DATA_URL = "https://data.london.gov.uk/download/2nlqm/b4a7e790-8cb8-451c-b828-c4c5d8445705/Weather%20Data%202014-11-30.xlsx"

# Raw data folder inside the runtime/repository.
RAW_DIR = Path(os.environ.get("THESIS_RAW_DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

PV_ZIP_PATH = RAW_DIR / "PV Data.zip"
WEATHER_FILE = RAW_DIR / "Weather Data 2014-11-30.xlsx"


def download_file(url, output_path, chunk_size=1024 * 1024):
    """Download a file only when it is not already available locally."""
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already available: {output_path}")
        return

    print(f"Downloading: {output_path.name}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)

    print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MB)")


def extract_zip_if_needed(zip_path, extract_dir):
    """Extract a zip file once by using a marker file next to the zip."""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    marker = extract_dir / f".{zip_path.stem.replace(' ', '_').lower()}_extracted"

    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return

    print(f"Extracting: {zip_path.name}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    marker.touch()
    print(f"Extracted to: {extract_dir}")


def extract_nested_zips(root_dir, max_rounds=5):
    """
    Extract zip files that are inside the downloaded PV archive.

    The London Datastore PV file contains nested zip files. The hourly CSV files
    are not always directly visible after extracting only the first PV Data.zip.
    This function recursively extracts inner zip files until the required CSVs
    become searchable under data/raw/.
    """
    root_dir = Path(root_dir)

    for round_idx in range(max_rounds):
        zip_files = sorted(root_dir.rglob("*.zip"))
        extracted_any = False

        for zip_file in zip_files:
            if zip_file.resolve() == PV_ZIP_PATH.resolve():
                continue

            destination = zip_file.parent / zip_file.stem
            marker = zip_file.parent / f".{zip_file.stem.replace(' ', '_').lower()}_extracted"

            if marker.exists():
                continue

            print(f"Extracting nested zip: {zip_file}")
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(destination)
            marker.touch()
            extracted_any = True

        if not extracted_any:
            break


def find_file_flexible(root_dir, required_keywords, extension=None):
    """
    Find a file recursively using case-insensitive keyword matching.

    This is safer than exact filename matching because downloaded archives can
    contain extra folders, slightly different spacing, or nested zip structures.
    """
    root_dir = Path(root_dir)
    candidates = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if extension is not None and path.suffix.lower() != extension.lower():
            continue

        name = path.name.lower()
        if all(keyword.lower() in name for keyword in required_keywords):
            candidates.append(path)

    if not candidates:
        print("\nAvailable CSV/XLSX files found under data/raw:")
        for path in sorted(root_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                print(" -", path)
        raise FileNotFoundError(
            f"Could not find a file containing keywords {required_keywords} under {root_dir}"
        )

    # Prefer the shortest path/name; this usually selects the real raw file rather than duplicates.
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


# Download and prepare raw files.
download_file(PV_DATA_URL, PV_ZIP_PATH)
download_file(WEATHER_DATA_URL, WEATHER_FILE)

# First extract the outer PV archive.
extract_zip_if_needed(PV_ZIP_PATH, RAW_DIR)

# Then extract inner zip files, because the required hourly CSV files are nested.
extract_nested_zips(RAW_DIR)

# Required raw files used in this thesis pipeline.
file_path_customer = str(find_file_flexible(RAW_DIR, ["customer", "endpoints"], extension=".csv"))
file_path_feeders = str(find_file_flexible(RAW_DIR, ["feeders"], extension=".csv"))
file_path_weather = str(WEATHER_FILE)

# DATA_ROOT is kept for compatibility with the original notebook structure.
DATA_ROOT = str(Path(file_path_customer).parent)

print("\nRaw data is ready.")
print("Customer endpoint file:", file_path_customer)
print("Feeder file:", file_path_feeders)
print("Weather file:", file_path_weather)
print("DATA_ROOT:", DATA_ROOT)


# ===== Notebook code cell 15 =====
# ============================================================
# 5. Check file paths
# ============================================================

for path in [file_path_customer, file_path_feeders, file_path_weather]:
    print(path)
    print("Exists:", os.path.exists(path))
    print("-" * 80)

if not os.path.exists(file_path_customer):
    raise FileNotFoundError("Customer file not found. Check file_path_customer.")

if not os.path.exists(file_path_feeders):
    raise FileNotFoundError("Feeders file not found. Check file_path_feeders.")

if not os.path.exists(file_path_weather):
    raise FileNotFoundError("Weather file not found. Check file_path_weather.")


# ===== Notebook code cell 17 =====
# ============================================================
# 6. Load raw data
# ============================================================

customer_df = pd.read_csv(file_path_customer, low_memory=False)
feeders_df = pd.read_csv(file_path_feeders, low_memory=False)
weather_df = pd.read_excel(file_path_weather)

print("\nRaw data loaded:")
print("Customer shape:", customer_df.shape)
print("Feeders shape:", feeders_df.shape)
print("Weather shape:", weather_df.shape)


# ===== Notebook code cell 19 =====
# ============================================================
# 7. Check required columns
# ============================================================

required_customer_cols = [
    "SerialNo",
    "Substation",
    "datetime",
    "V_MIN_Filtered",
    "V_MAX_Filtered",
    "P_GEN_MIN",
    "P_GEN_MAX"
]

required_feeder_cols = [
    "SerialNo",
    "Substation",
    "datetime",
    "IA_MIN_Filtered",
    "IA_MAX_Filtered",
    "IB_MIN_Filtered",
    "IB_MAX_Filtered",
    "IC_MIN_Filtered",
    "IC_MAX_Filtered"
]

required_weather_cols = [
    "Date",
    "Time",
    "SolarRad",
    "TempOut",
    "OutHum",
    "WindSpeed"
]


def check_required_columns(df, required_cols, df_name):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {df_name}: {missing_cols}")
    print(f"OK: all required columns exist in {df_name}")


check_required_columns(customer_df, required_customer_cols, "customer_df")
check_required_columns(feeders_df, required_feeder_cols, "feeders_df")
check_required_columns(weather_df, required_weather_cols, "weather_df")


# ===== Notebook code cell 21 =====
# ============================================================
# 8. Datetime conversion
# ============================================================

customer_df["datetime"] = pd.to_datetime(customer_df["datetime"], errors="coerce")
feeders_df["datetime"] = pd.to_datetime(feeders_df["datetime"], errors="coerce")

weather_df["datetime"] = pd.to_datetime(
    weather_df["Date"].astype(str) + " " + weather_df["Time"].astype(str),
    errors="coerce"
)

weather_df["datetime_hour"] = weather_df["datetime"].dt.floor("h")


# ===== Notebook code cell 23 =====
# ============================================================
# 9. Numeric conversion
# ============================================================

customer_numeric_cols = [
    "V_MIN_Filtered",
    "V_MAX_Filtered",
    "P_GEN_MIN",
    "P_GEN_MAX"
]

for col in customer_numeric_cols:
    customer_df[col] = pd.to_numeric(customer_df[col], errors="coerce")

feeder_numeric_cols = [
    "IA_MIN_Filtered",
    "IA_MAX_Filtered",
    "IB_MIN_Filtered",
    "IB_MAX_Filtered",
    "IC_MIN_Filtered",
    "IC_MAX_Filtered"
]

for col in feeder_numeric_cols:
    feeders_df[col] = pd.to_numeric(feeders_df[col], errors="coerce")

weather_columns = [
    "SolarRad",
    "TempOut",
    "OutHum",
    "WindSpeed"
]

for col in weather_columns:
    weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")


# ===== Notebook code cell 25 =====
# ============================================================
# 10. Construct target variable Y
# ============================================================

customer_df["V_REP_Filtered"] = 0.5 * (
    customer_df["V_MIN_Filtered"] + customer_df["V_MAX_Filtered"]
)

customer_df["Y"] = customer_df["V_REP_Filtered"] - 230

customer_df["Y_max"] = customer_df["V_MAX_Filtered"] - 230
customer_df["Y_min"] = customer_df["V_MIN_Filtered"] - 230
customer_df["Y_abs"] = customer_df["Y"].abs()


# ===== Notebook code cell 27 =====
# ============================================================
# 11. Construct PV feature
# ============================================================

customer_df["PV_raw"] = 0.5 * (
    customer_df["P_GEN_MIN"] + customer_df["P_GEN_MAX"]
)

pv_median = customer_df["PV_raw"].median(skipna=True)

if pv_median < 0:
    customer_df["PV"] = -customer_df["PV_raw"]
else:
    customer_df["PV"] = customer_df["PV_raw"]

customer_df["PV"] = customer_df["PV"].clip(lower=0)

print("\nPV_raw median:", pv_median)
print("PV sign correction applied:", pv_median < 0)
print("\nPV summary:")
print(customer_df["PV"].describe())


# ===== Notebook code cell 29 =====
# ============================================================
# 12. Construct feeder-level Load
# ============================================================

feeders_df["IA_REP_Filtered"] = 0.5 * (
    feeders_df["IA_MIN_Filtered"] + feeders_df["IA_MAX_Filtered"]
)

feeders_df["IB_REP_Filtered"] = 0.5 * (
    feeders_df["IB_MIN_Filtered"] + feeders_df["IB_MAX_Filtered"]
)

feeders_df["IC_REP_Filtered"] = 0.5 * (
    feeders_df["IC_MIN_Filtered"] + feeders_df["IC_MAX_Filtered"]
)

feeders_df["Load"] = (
    feeders_df["IA_REP_Filtered"] +
    feeders_df["IB_REP_Filtered"] +
    feeders_df["IC_REP_Filtered"]
)

feeder_agg = (
    feeders_df
    .groupby(["Substation", "datetime"], as_index=False)
    .agg(
        Load=("Load", "sum"),
        Feeder_Count=("SerialNo", "nunique")
    )
)

print("\nFeeder aggregated shape:", feeder_agg.shape)


# ===== Notebook code cell 31 =====
# ============================================================
# 13. Construct hourly weather data
# ============================================================

weather_hourly = (
    weather_df
    .groupby("datetime_hour", as_index=False)[weather_columns]
    .mean()
)

print("Weather hourly shape:", weather_hourly.shape)


# ===== Notebook code cell 33 =====
# ============================================================
# 14. Merge customer + feeder load
# ============================================================

rows_before_merge = len(customer_df)

cust_plus_feeder = customer_df.merge(
    feeder_agg,
    on=["Substation", "datetime"],
    how="left"
)

rows_after_feeder_merge = len(cust_plus_feeder)

print("\nRows before feeder merge:", rows_before_merge)
print("Rows after feeder merge:", rows_after_feeder_merge)

if rows_before_merge == rows_after_feeder_merge:
    print("OK: no row duplication after feeder merge.")
else:
    print("WARNING: row duplication happened after feeder merge.")


# ===== Notebook code cell 35 =====
# ============================================================
# 15. Merge weather
# ============================================================

cust_plus_feeder["datetime_hour"] = cust_plus_feeder["datetime"].dt.floor("h")

merged_df = cust_plus_feeder.merge(
    weather_hourly,
    on="datetime_hour",
    how="left"
)

rows_after_weather_merge = len(merged_df)

print("\nRows after weather merge:", rows_after_weather_merge)

if rows_before_merge == rows_after_weather_merge:
    print("OK: no row duplication after weather merge.")
else:
    print("WARNING: row duplication happened after weather merge.")


# ===== Notebook code cell 37 =====
# ============================================================
# 16. Time filtering
# ============================================================

start_date = pd.Timestamp("2013-11-26 00:00:00")
end_date = pd.Timestamp("2014-11-30 23:59:59")

merged_df = merged_df[
    (merged_df["datetime"] >= start_date) &
    (merged_df["datetime"] <= end_date)
].copy()

print("\nShape after time filtering:", merged_df.shape)
print("Datetime range:", merged_df["datetime"].min(), "to", merged_df["datetime"].max())


# ===== Notebook code cell 39 =====
# ============================================================
# 17. Construct final datasets
# ============================================================

final_df = merged_df[
    [
        "SerialNo",
        "Substation",
        "datetime",
        "Y",
        "Y_max",
        "Y_min",
        "Y_abs",
        "V_REP_Filtered",
        "V_MIN_Filtered",
        "V_MAX_Filtered",
        "PV",
        "PV_raw",
        "Load",
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed",
        "Feeder_Count"
    ]
].copy()

required_model_columns = [
    "SerialNo",
    "Substation",
    "datetime",
    "Y",
    "PV",
    "Load",
    "SolarRad",
    "TempOut",
    "OutHum",
    "WindSpeed"
]

clean_full_df = final_df.dropna(subset=required_model_columns).copy()

model_only_df = clean_full_df[
    [
        "SerialNo",
        "Substation",
        "datetime",
        "Y",
        "PV",
        "Load",
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed"
    ]
].copy()


# ===== Notebook code cell 41 =====
# ============================================================
# 18. Save constructed datasets
# ============================================================

final_df.to_csv(
    os.path.join(dataset_output_dir, "final_dataset_uncleaned_correct_target.csv"),
    index=False
)

final_df.to_excel(
    os.path.join(dataset_output_dir, "final_dataset_uncleaned_correct_target.xlsx"),
    index=False
)

clean_full_df.to_csv(
    os.path.join(dataset_output_dir, "final_dataset_clean_full_correct_target.csv"),
    index=False
)

clean_full_df.to_excel(
    os.path.join(dataset_output_dir, "final_dataset_clean_full_correct_target.xlsx"),
    index=False
)

model_only_df.to_csv(
    os.path.join(dataset_output_dir, "final_dataset_model_only_correct_target.csv"),
    index=False
)

model_only_df.to_excel(
    os.path.join(dataset_output_dir, "final_dataset_model_only_correct_target.xlsx"),
    index=False
)

dataset_summary_df = pd.DataFrame({
    "Item": [
        "Rows in raw customer_df",
        "Rows after time filtering",
        "Rows in final uncleaned dataset",
        "Rows in clean full dataset",
        "Rows in model-only dataset",
        "Rows removed due to missing required model variables",
        "Start datetime",
        "End datetime",
        "Unique customer endpoints",
        "Unique substations",
        "Positive Y percentage",
        "Negative Y percentage",
        "Positive Y_max percentage",
        "Negative Y_min percentage",
        "PV_raw median",
        "PV sign correction applied"
    ],
    "Value": [
        len(customer_df),
        len(merged_df),
        len(final_df),
        len(clean_full_df),
        len(model_only_df),
        len(final_df) - len(clean_full_df),
        final_df["datetime"].min(),
        final_df["datetime"].max(),
        final_df["SerialNo"].nunique(),
        final_df["Substation"].nunique(),
        round((final_df["Y"] > 0).mean() * 100, 2),
        round((final_df["Y"] < 0).mean() * 100, 2),
        round((final_df["Y_max"] > 0).mean() * 100, 2),
        round((final_df["Y_min"] < 0).mean() * 100, 2),
        pv_median,
        pv_median < 0
    ]
})

dataset_summary_df.to_excel(
    os.path.join(dataset_output_dir, "dataset_construction_summary.xlsx"),
    index=False
)

dataset_summary_df.to_csv(
    os.path.join(dataset_output_dir, "dataset_construction_summary.csv"),
    index=False
)

print("\nDataset construction summary:")
display(dataset_summary_df)


# ===== Notebook code cell 44 =====
# ============================================================
# 19. Use constructed model-only dataset
# ============================================================

df = model_only_df.copy()

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

numeric_cols = [
    "Y",
    "PV",
    "Load",
    "SolarRad",
    "TempOut",
    "OutHum",
    "WindSpeed"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("datetime").reset_index(drop=True)

required_columns = [
    "SerialNo",
    "datetime",
    "Y",
    "PV",
    "Load",
    "SolarRad",
    "TempOut",
    "OutHum",
    "WindSpeed"
]

original_rows = len(df)

df_clean = df.dropna(subset=required_columns).copy()

clean_rows = len(df_clean)
removed_rows = original_rows - clean_rows
removed_percent = removed_rows / original_rows * 100

print("\nModelling dataset shape before cleaning:", df.shape)
print("Modelling dataset shape after cleaning:", df_clean.shape)

df_clean.to_excel(
    os.path.join(model_output_dir, "dataset_used_for_modelling_clean.xlsx"),
    index=False
)

df_clean.to_csv(
    os.path.join(model_output_dir, "dataset_used_for_modelling_clean.csv"),
    index=False
)


# ===== Notebook code cell 46 =====
# ============================================================
# 20. Target sanity check
# ============================================================

positive_y_pct = (df_clean["Y"] > 0).mean() * 100
negative_y_pct = (df_clean["Y"] < 0).mean() * 100

print("\nTarget Y summary:")
print(df_clean["Y"].describe())
print("Positive Y percentage:", round(positive_y_pct, 2), "%")
print("Negative Y percentage:", round(negative_y_pct, 2), "%")

if negative_y_pct == 0:
    print("\nWARNING: Y has no negative values. Check target construction.")


# ===== Notebook code cell 48 =====
# ============================================================
# 21. Define target and feature sets
# ============================================================

target_col = "Y"

feature_sets = {
    "PV only": [
        "PV"
    ],
    "PV + Weather": [
        "PV",
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed"
    ],
    "PV + Load": [
        "PV",
        "Load"
    ],
    "PV + Weather + Load": [
        "PV",
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed",
        "Load"
    ]
}

full_feature_set_name = "PV + Weather + Load"
full_features = feature_sets[full_feature_set_name]


# ===== Notebook code cell 50 =====
# ============================================================
# 22. EDA
# ============================================================

print("\n================ EDA START ================")

eda_features = full_features + [target_col]

df_clean["hour"] = df_clean["datetime"].dt.hour
df_clean["month"] = df_clean["datetime"].dt.month
df_clean["dayofweek"] = df_clean["datetime"].dt.dayofweek

eda_summary = pd.DataFrame({
    "Item": [
        "Original rows",
        "Rows after cleaning",
        "Removed rows",
        "Removed percentage",
        "Number of columns",
        "Number of customer endpoints",
        "Start datetime",
        "End datetime",
        "Positive Y percentage",
        "Negative Y percentage"
    ],
    "Value": [
        original_rows,
        clean_rows,
        removed_rows,
        round(removed_percent, 2),
        df_clean.shape[1],
        df_clean["SerialNo"].nunique(),
        df_clean["datetime"].min(),
        df_clean["datetime"].max(),
        round(positive_y_pct, 2),
        round(negative_y_pct, 2)
    ]
})

display(eda_summary)

eda_summary.to_excel(os.path.join(eda_output_dir, "eda_dataset_summary.xlsx"), index=False)
eda_summary.to_csv(os.path.join(eda_output_dir, "eda_dataset_summary.csv"), index=False)

descriptive_stats = df_clean[eda_features].describe().T
display(descriptive_stats)

descriptive_stats.to_excel(os.path.join(eda_output_dir, "eda_descriptive_statistics.xlsx"))
descriptive_stats.to_csv(os.path.join(eda_output_dir, "eda_descriptive_statistics.csv"))

missing_summary = pd.DataFrame({
    "Column": eda_features,
    "Missing_Count": df_clean[eda_features].isna().sum().values,
    "Missing_Percentage": df_clean[eda_features].isna().mean().values * 100
})

display(missing_summary)

missing_summary.to_excel(os.path.join(eda_output_dir, "eda_missing_summary.xlsx"), index=False)
missing_summary.to_csv(os.path.join(eda_output_dir, "eda_missing_summary.csv"), index=False)

corr_matrix = df_clean[eda_features].corr()
display(corr_matrix)

corr_matrix.to_excel(os.path.join(eda_output_dir, "eda_correlation_matrix.xlsx"))
corr_matrix.to_csv(os.path.join(eda_output_dir, "eda_correlation_matrix.csv"))

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(eda_output_dir, "eda_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

for col in eda_features:
    plt.figure(figsize=(8, 5))
    plt.hist(df_clean[col].dropna(), bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, f"eda_distribution_{col}.png"), dpi=300, bbox_inches="tight")
    plt.show()

for col in full_features:
    plt.figure(figsize=(8, 5))
    plt.scatter(df_clean[col], df_clean[target_col], alpha=0.25)
    plt.title(f"{col} vs Voltage Deviation")
    plt.xlabel(col)
    plt.ylabel("Voltage Deviation from 230 V")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, f"eda_scatter_{col}_vs_Y.png"), dpi=300, bbox_inches="tight")
    plt.show()

hourly_summary = df_clean.groupby("hour")[eda_features].mean().reset_index()
display(hourly_summary)

hourly_summary.to_excel(os.path.join(eda_output_dir, "eda_hourly_summary.xlsx"), index=False)
hourly_summary.to_csv(os.path.join(eda_output_dir, "eda_hourly_summary.csv"), index=False)

for col in eda_features:
    plt.figure(figsize=(8, 5))
    plt.plot(hourly_summary["hour"], hourly_summary[col], marker="o")
    plt.title(f"Average {col} by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel(col)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, f"eda_hourly_average_{col}.png"), dpi=300, bbox_inches="tight")
    plt.show()

monthly_summary = df_clean.groupby("month")[eda_features].mean().reset_index()
display(monthly_summary)

monthly_summary.to_excel(os.path.join(eda_output_dir, "eda_monthly_summary.xlsx"), index=False)
monthly_summary.to_csv(os.path.join(eda_output_dir, "eda_monthly_summary.csv"), index=False)

for col in eda_features:
    plt.figure(figsize=(8, 5))
    plt.plot(monthly_summary["month"], monthly_summary[col], marker="o")
    plt.title(f"Average {col} by Month")
    plt.xlabel("Month")
    plt.ylabel(col)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, f"eda_monthly_average_{col}.png"), dpi=300, bbox_inches="tight")
    plt.show()

plt.figure(figsize=(12, 5))
data_by_hour = [
    df_clean[df_clean["hour"] == h][target_col].dropna()
    for h in range(24)
]
plt.boxplot(data_by_hour, labels=list(range(24)), showfliers=False)
plt.title("Voltage Deviation by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Voltage Deviation from 230 V")
plt.tight_layout()
plt.savefig(os.path.join(eda_output_dir, "eda_boxplot_Y_by_hour.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
data_by_month = [
    df_clean[df_clean["month"] == m][target_col].dropna()
    for m in range(1, 13)
]
plt.boxplot(data_by_month, labels=list(range(1, 13)), showfliers=False)
plt.title("Voltage Deviation by Month")
plt.xlabel("Month")
plt.ylabel("Voltage Deviation from 230 V")
plt.tight_layout()
plt.savefig(os.path.join(eda_output_dir, "eda_boxplot_Y_by_month.png"), dpi=300, bbox_inches="tight")
plt.show()

over_under_summary = pd.DataFrame({
    "Group": [
        "Positive deviation Y > 0",
        "Negative deviation Y < 0",
        "Near nominal Y = 0"
    ],
    "Count": [
        (df_clean[target_col] > 0).sum(),
        (df_clean[target_col] < 0).sum(),
        (df_clean[target_col] == 0).sum()
    ]
})

over_under_summary["Percentage"] = over_under_summary["Count"] / len(df_clean) * 100

display(over_under_summary)

over_under_summary.to_excel(
    os.path.join(eda_output_dir, "eda_positive_negative_deviation_summary.xlsx"),
    index=False
)

over_under_summary.to_csv(
    os.path.join(eda_output_dir, "eda_positive_negative_deviation_summary.csv"),
    index=False
)

print("================ EDA FINISHED ================\n")


# ===== Notebook code cell 52 =====
# ============================================================
# 23. Time-aware Train / Validation / Test split
# ============================================================

unique_times = np.array(sorted(df_clean["datetime"].unique()))

train_cutoff = unique_times[int(len(unique_times) * 0.70)]
val_cutoff = unique_times[int(len(unique_times) * 0.85)]

train_df = df_clean[df_clean["datetime"] <= train_cutoff].copy()

val_df = df_clean[
    (df_clean["datetime"] > train_cutoff) &
    (df_clean["datetime"] <= val_cutoff)
].copy()

test_df = df_clean[df_clean["datetime"] > val_cutoff].copy()

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain period:")
print(train_df["datetime"].min(), "to", train_df["datetime"].max())

print("\nValidation period:")
print(val_df["datetime"].min(), "to", val_df["datetime"].max())

print("\nTest period:")
print(test_df["datetime"].min(), "to", test_df["datetime"].max())

y_train = train_df[target_col]
y_val = val_df[target_col]
y_test = test_df[target_col]

X_train_full = train_df[full_features]
X_val_full = val_df[full_features]
X_test_full = test_df[full_features]

split_summary_df = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Rows": [len(train_df), len(val_df), len(test_df)],
    "Start datetime": [train_df["datetime"].min(), val_df["datetime"].min(), test_df["datetime"].min()],
    "End datetime": [train_df["datetime"].max(), val_df["datetime"].max(), test_df["datetime"].max()]
})

display(split_summary_df)

split_summary_df.to_excel(
    os.path.join(model_output_dir, "chronological_train_validation_test_split.xlsx"),
    index=False
)

split_summary_df.to_csv(
    os.path.join(model_output_dir, "chronological_train_validation_test_split.csv"),
    index=False
)


# ===== Notebook code cell 54 =====
# ============================================================
# 24. Evaluation functions
# ============================================================


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2



def train_and_evaluate(model, X_train, y_train, X_eval, y_eval):
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    y_pred = fitted_model.predict(X_eval)
    mae, rmse, r2 = get_metrics(y_eval, y_pred)
    return fitted_model, y_pred, mae, rmse, r2


# ===== Notebook code cell 56 =====
# ============================================================
# 25. Linear Regression baseline
# ============================================================

baseline_model = LinearRegression()

baseline_fitted, baseline_val_pred, baseline_val_mae, baseline_val_rmse, baseline_val_r2 = train_and_evaluate(
    baseline_model,
    X_train_full,
    y_train,
    X_val_full,
    y_val
)

baseline_result = {
    "Model": "Linear Regression",
    "Tuned": "No",
    "Feature Set": full_feature_set_name,
    "MAE": baseline_val_mae,
    "RMSE": baseline_val_rmse,
    "R²": baseline_val_r2,
    "Best Params": "N/A"
}


# ===== Notebook code cell 58 =====
# ============================================================
# 26. Optuna tuning for ensemble models
# ============================================================

N_TRIALS = 100

print("\nOptuna trials per model:", N_TRIALS)

optuna.logging.set_verbosity(optuna.logging.WARNING)

tuned_model_results = []
tuned_model_templates = {}


# ------------------------------
# 26.1 Random Forest
# ------------------------------

def objective_random_forest(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=SEED,
        n_jobs=1
    )
    model.fit(X_train_full, y_train)
    pred = model.predict(X_val_full)
    return mean_absolute_error(y_val, pred)


study_rf = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_rf.optimize(objective_random_forest, n_trials=N_TRIALS, show_progress_bar=True)

best_rf = RandomForestRegressor(
    **study_rf.best_params,
    random_state=SEED,
    n_jobs=1
)

rf_fitted, rf_val_pred, rf_val_mae, rf_val_rmse, rf_val_r2 = train_and_evaluate(
    best_rf,
    X_train_full,
    y_train,
    X_val_full,
    y_val
)

tuned_model_templates["Random Forest"] = best_rf

tuned_model_results.append({
    "Model": "Random Forest",
    "Tuned": "Yes - Optuna",
    "Feature Set": full_feature_set_name,
    "MAE": rf_val_mae,
    "RMSE": rf_val_rmse,
    "R²": rf_val_r2,
    "Best Params": str(study_rf.best_params)
})


# ------------------------------
# 26.2 Gradient Boosting
# ------------------------------

def objective_gradient_boosting(trial):
    model = GradientBoostingRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_state=SEED
    )
    model.fit(X_train_full, y_train)
    pred = model.predict(X_val_full)
    return mean_absolute_error(y_val, pred)


study_gb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_gb.optimize(objective_gradient_boosting, n_trials=N_TRIALS, show_progress_bar=True)

best_gb = GradientBoostingRegressor(
    **study_gb.best_params,
    random_state=SEED
)

gb_fitted, gb_val_pred, gb_val_mae, gb_val_rmse, gb_val_r2 = train_and_evaluate(
    best_gb,
    X_train_full,
    y_train,
    X_val_full,
    y_val
)

tuned_model_templates["Gradient Boosting"] = best_gb

tuned_model_results.append({
    "Model": "Gradient Boosting",
    "Tuned": "Yes - Optuna",
    "Feature Set": full_feature_set_name,
    "MAE": gb_val_mae,
    "RMSE": gb_val_rmse,
    "R²": gb_val_r2,
    "Best Params": str(study_gb.best_params)
})


# ------------------------------
# 26.3 XGBoost
# ------------------------------

def objective_xgboost(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1
    )
    model.fit(X_train_full, y_train)
    pred = model.predict(X_val_full)
    return mean_absolute_error(y_val, pred)


study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(objective_xgboost, n_trials=N_TRIALS, show_progress_bar=True)

best_xgb = XGBRegressor(
    **study_xgb.best_params,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=1
)

xgb_fitted, xgb_val_pred, xgb_val_mae, xgb_val_rmse, xgb_val_r2 = train_and_evaluate(
    best_xgb,
    X_train_full,
    y_train,
    X_val_full,
    y_val
)

tuned_model_templates["XGBoost"] = best_xgb

tuned_model_results.append({
    "Model": "XGBoost",
    "Tuned": "Yes - Optuna",
    "Feature Set": full_feature_set_name,
    "MAE": xgb_val_mae,
    "RMSE": xgb_val_rmse,
    "R²": xgb_val_r2,
    "Best Params": str(study_xgb.best_params)
})


# ------------------------------
# 26.4 LightGBM
# ------------------------------

def objective_lightgbm(trial):
    model = LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 10, 100),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_col_wise=True
    )
    model.fit(X_train_full, y_train)
    pred = model.predict(X_val_full)
    return mean_absolute_error(y_val, pred)


study_lgbm = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgbm.optimize(objective_lightgbm, n_trials=N_TRIALS, show_progress_bar=True)

best_lgbm = LGBMRegressor(
    **study_lgbm.best_params,
    random_state=SEED,
    n_jobs=1,
    verbose=-1
)

lgbm_fitted, lgbm_val_pred, lgbm_val_mae, lgbm_val_rmse, lgbm_val_r2 = train_and_evaluate(
    best_lgbm,
    X_train_full,
    y_train,
    X_val_full,
    y_val
)

tuned_model_templates["LightGBM"] = best_lgbm

tuned_model_results.append({
    "Model": "LightGBM",
    "Tuned": "Yes - Optuna",
    "Feature Set": full_feature_set_name,
    "MAE": lgbm_val_mae,
    "RMSE": lgbm_val_rmse,
    "R²": lgbm_val_r2,
    "Best Params": str(study_lgbm.best_params)
})


# ===== Notebook code cell 60 =====
# ============================================================
# 27. Untuned vs Tuned comparison
# ============================================================

untuned_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=SEED
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_col_wise=True
    )
}

untuned_results = []

for model_name, model in untuned_models.items():
    fitted_model, val_pred, val_mae, val_rmse, val_r2 = train_and_evaluate(
        model,
        X_train_full,
        y_train,
        X_val_full,
        y_val
    )

    untuned_results.append({
        "Model": model_name,
        "Untuned MAE": val_mae,
        "Untuned RMSE": val_rmse,
        "Untuned R²": val_r2
    })

untuned_results_df = pd.DataFrame(untuned_results)

tuned_results_df = pd.DataFrame(tuned_model_results)[[
    "Model",
    "MAE",
    "RMSE",
    "R²"
]].rename(columns={
    "MAE": "Tuned MAE",
    "RMSE": "Tuned RMSE",
    "R²": "Tuned R²"
})

tuning_comparison_df = untuned_results_df.merge(
    tuned_results_df,
    on="Model",
    how="inner"
)

tuning_comparison_df["MAE Improvement"] = (
    tuning_comparison_df["Untuned MAE"] -
    tuning_comparison_df["Tuned MAE"]
)

tuning_comparison_df["MAE Improvement %"] = (
    tuning_comparison_df["MAE Improvement"] /
    tuning_comparison_df["Untuned MAE"] * 100
)

tuning_comparison_df = tuning_comparison_df.sort_values(
    "MAE Improvement %",
    ascending=False
).reset_index(drop=True).round(4)

display(tuning_comparison_df)

tuning_comparison_df.to_excel(
    os.path.join(model_output_dir, "untuned_vs_tuned_comparison.xlsx"),
    index=False
)

tuning_comparison_df.to_csv(
    os.path.join(model_output_dir, "untuned_vs_tuned_comparison.csv"),
    index=False
)


# ===== Notebook code cell 62 =====
# ============================================================
# 28. Validation model comparison
# ============================================================

validation_results = [baseline_result] + tuned_model_results

validation_results_df = pd.DataFrame(validation_results)

validation_results_df = validation_results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

validation_results_df["MAE"] = validation_results_df["MAE"].round(4)
validation_results_df["RMSE"] = validation_results_df["RMSE"].round(4)
validation_results_df["R²"] = validation_results_df["R²"].round(4)

display(validation_results_df)

validation_output_xlsx = os.path.join(
    model_output_dir,
    "validation_model_selection_results_optuna.xlsx"
)

validation_output_csv = os.path.join(
    model_output_dir,
    "validation_model_selection_results_optuna.csv"
)

validation_results_df.to_excel(validation_output_xlsx, index=False)
validation_results_df.to_csv(validation_output_csv, index=False)

best_model_name = validation_results_df.iloc[0]["Model"]

print("\nSelected best model based on Validation MAE:")
print(best_model_name)

if best_model_name == "Linear Regression":
    best_model_template = baseline_model
else:
    best_model_template = tuned_model_templates[best_model_name]


# ===== Notebook code cell 64 =====
# ============================================================
# 29. Ablation analysis
# ============================================================

ablation_results = []

for feature_set_name, features in feature_sets.items():
    X_train_ablation = train_df[features]
    X_val_ablation = val_df[features]

    fitted_model, val_pred, val_mae, val_rmse, val_r2 = train_and_evaluate(
        model=best_model_template,
        X_train=X_train_ablation,
        y_train=y_train,
        X_eval=X_val_ablation,
        y_eval=y_val
    )

    ablation_results.append({
        "Selected Model": best_model_name,
        "Feature Set": feature_set_name,
        "MAE": val_mae,
        "RMSE": val_rmse,
        "R²": val_r2
    })

ablation_results_df = pd.DataFrame(ablation_results)

ablation_results_df = ablation_results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True).round(4)

display(ablation_results_df)

ablation_output_xlsx = os.path.join(
    model_output_dir,
    "ablation_results_selected_model_optuna.xlsx"
)

ablation_output_csv = os.path.join(
    model_output_dir,
    "ablation_results_selected_model_optuna.csv"
)

ablation_results_df.to_excel(ablation_output_xlsx, index=False)
ablation_results_df.to_csv(ablation_output_csv, index=False)


# ===== Notebook code cell 66 =====
# ============================================================
# 30. Final training and test evaluation for all models
# ============================================================

train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("datetime").reset_index(drop=True)

X_train_val_full = train_val_df[full_features]
y_train_val = train_val_df[target_col]

# ------------------------------------------------------------
# 30.1 Build final model templates
# ------------------------------------------------------------

final_model_templates = {
    "Linear Regression": baseline_model,
    "Random Forest": tuned_model_templates["Random Forest"],
    "Gradient Boosting": tuned_model_templates["Gradient Boosting"],
    "XGBoost": tuned_model_templates["XGBoost"],
    "LightGBM": tuned_model_templates["LightGBM"]
}

# ------------------------------------------------------------
# 30.2 Evaluate every candidate model on the unseen test set
# ------------------------------------------------------------

all_test_results = []
final_fitted_models = {}
final_test_predictions = {}

for model_name, model_template in final_model_templates.items():
    model = clone(model_template)
    model.fit(X_train_val_full, y_train_val)
    y_pred_test_model = model.predict(X_test_full)

    test_mae_model, test_rmse_model, test_r2_model = get_metrics(
        y_test,
        y_pred_test_model
    )

    all_test_results.append({
        "Model": model_name,
        "Selected_on_Validation": model_name == best_model_name,
        "Feature Set": full_feature_set_name,
        "Test MAE": test_mae_model,
        "Test RMSE": test_rmse_model,
        "Test R²": test_r2_model
    })

    final_fitted_models[model_name] = model
    final_test_predictions[model_name] = y_pred_test_model

all_test_results_df = pd.DataFrame(all_test_results)

all_test_results_df = all_test_results_df.sort_values(
    by="Test MAE",
    ascending=True
).reset_index(drop=True)

all_test_results_df_rounded = all_test_results_df.copy()
all_test_results_df_rounded["Test MAE"] = all_test_results_df_rounded["Test MAE"].round(4)
all_test_results_df_rounded["Test RMSE"] = all_test_results_df_rounded["Test RMSE"].round(4)
all_test_results_df_rounded["Test R²"] = all_test_results_df_rounded["Test R²"].round(4)

display(all_test_results_df_rounded)

all_test_results_df_rounded.to_excel(
    os.path.join(model_output_dir, "final_test_comparison_all_models.xlsx"),
    index=False
)

all_test_results_df_rounded.to_csv(
    os.path.join(model_output_dir, "final_test_comparison_all_models.csv"),
    index=False
)

# ------------------------------------------------------------
# 30.3 Keep the validation-selected model as the official final model
# ------------------------------------------------------------

final_model = final_fitted_models[best_model_name]
test_pred = final_test_predictions[best_model_name]

test_mae, test_rmse, test_r2 = get_metrics(y_test, test_pred)

final_test_results_df = pd.DataFrame([{
    "Final Model Selected on Validation": best_model_name,
    "Feature Set": full_feature_set_name,
    "Test MAE": round(test_mae, 4),
    "Test RMSE": round(test_rmse, 4),
    "Test R²": round(test_r2, 4)
}])

display(final_test_results_df)

final_test_output_xlsx = os.path.join(
    model_output_dir,
    "final_test_results_selected_model_optuna.xlsx"
)

final_test_output_csv = os.path.join(
    model_output_dir,
    "final_test_results_selected_model_optuna.csv"
)

final_test_results_df.to_excel(final_test_output_xlsx, index=False)
final_test_results_df.to_csv(final_test_output_csv, index=False)

# ------------------------------------------------------------
# 30.4 Identify best test model for transparent reporting only
# ------------------------------------------------------------

best_test_model_name = all_test_results_df.iloc[0]["Model"]
best_test_mae = all_test_results_df.iloc[0]["Test MAE"]
best_test_rmse = all_test_results_df.iloc[0]["Test RMSE"]
best_test_r2 = all_test_results_df.iloc[0]["Test R²"]

test_selection_note_df = pd.DataFrame([{
    "Validation-selected model": best_model_name,
    "Best model on test set": best_test_model_name,
    "Same model?": best_model_name == best_test_model_name,
    "Validation-selected Test MAE": round(test_mae, 4),
    "Best Test MAE": round(best_test_mae, 4),
    "Note": "The official final model is selected based on validation MAE. The test set is used only for final evaluation and transparent comparison."
}])

display(test_selection_note_df)

test_selection_note_df.to_excel(
    os.path.join(model_output_dir, "test_selection_note.xlsx"),
    index=False
)

test_selection_note_df.to_csv(
    os.path.join(model_output_dir, "test_selection_note.csv"),
    index=False
)


# ===== Notebook code cell 68 =====
# ============================================================
# 31. TimeSeriesSplit robustness check
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

X_all_non_test = train_val_df[full_features]
y_all_non_test = train_val_df[target_col]

cv_results = []
split_number = 1

for train_index, val_index in tscv.split(X_all_non_test):
    X_cv_train = X_all_non_test.iloc[train_index]
    X_cv_val = X_all_non_test.iloc[val_index]

    y_cv_train = y_all_non_test.iloc[train_index]
    y_cv_val = y_all_non_test.iloc[val_index]

    cv_model = clone(best_model_template)
    cv_model.fit(X_cv_train, y_cv_train)

    cv_pred = cv_model.predict(X_cv_val)

    cv_mae, cv_rmse, cv_r2 = get_metrics(y_cv_val, cv_pred)

    cv_results.append({
        "Split": split_number,
        "Model": best_model_name,
        "MAE": cv_mae,
        "RMSE": cv_rmse,
        "R²": cv_r2,
        "Train Size": len(X_cv_train),
        "Validation Size": len(X_cv_val)
    })

    split_number += 1

cv_results_df = pd.DataFrame(cv_results).round(4)

cv_summary_df = pd.DataFrame([{
    "Model": best_model_name,
    "Average MAE": cv_results_df["MAE"].mean(),
    "STD MAE": cv_results_df["MAE"].std(),
    "Average RMSE": cv_results_df["RMSE"].mean(),
    "Average R²": cv_results_df["R²"].mean()
}]).round(4)

display(cv_results_df)
display(cv_summary_df)

cv_results_df.to_excel(
    os.path.join(model_output_dir, "timeseries_cv_results_best_model.xlsx"),
    index=False
)

cv_results_df.to_csv(
    os.path.join(model_output_dir, "timeseries_cv_results_best_model.csv"),
    index=False
)

cv_summary_df.to_excel(
    os.path.join(model_output_dir, "timeseries_cv_summary_best_model.xlsx"),
    index=False
)

cv_summary_df.to_csv(
    os.path.join(model_output_dir, "timeseries_cv_summary_best_model.csv"),
    index=False
)


# ===== Notebook code cell 70 =====
# ============================================================
# 32. Predictions and residuals
# ============================================================

predictions_df = test_df.copy()
predictions_df["Predicted_Y"] = test_pred
predictions_df["Residual"] = predictions_df["Y"] - predictions_df["Predicted_Y"]
predictions_df["ABS_Residual"] = predictions_df["Residual"].abs()

predictions_output_xlsx = os.path.join(
    model_output_dir,
    "final_model_predictions_optuna.xlsx"
)

predictions_output_csv = os.path.join(
    model_output_dir,
    "final_model_predictions_optuna.csv"
)

predictions_df.to_excel(predictions_output_xlsx, index=False)
predictions_df.to_csv(predictions_output_csv, index=False)


# ===== Notebook code cell 72 =====
# ============================================================
# 33. Bootstrap CI for Test MAE
# ============================================================

N_BOOTSTRAP = 1000
rng = np.random.default_rng(42)

absolute_errors = predictions_df["ABS_Residual"].values

bootstrap_maes = []

for _ in range(N_BOOTSTRAP):
    sample = rng.choice(
        absolute_errors,
        size=len(absolute_errors),
        replace=True
    )
    bootstrap_maes.append(np.mean(sample))

bootstrap_maes = np.array(bootstrap_maes)

mae_ci_lower = np.percentile(bootstrap_maes, 2.5)
mae_ci_upper = np.percentile(bootstrap_maes, 97.5)

mae_ci_df = pd.DataFrame([{
    "Test MAE": test_mae,
    "95% CI Lower": mae_ci_lower,
    "95% CI Upper": mae_ci_upper,
    "Bootstrap Samples": N_BOOTSTRAP
}]).round(4)

display(mae_ci_df)

mae_ci_df.to_excel(
    os.path.join(model_output_dir, "test_mae_bootstrap_confidence_interval.xlsx"),
    index=False
)

mae_ci_df.to_csv(
    os.path.join(model_output_dir, "test_mae_bootstrap_confidence_interval.csv"),
    index=False
)

plt.figure(figsize=(8, 5))
plt.hist(bootstrap_maes, bins=50)
plt.axvline(mae_ci_lower, linestyle="--", label="2.5% CI")
plt.axvline(mae_ci_upper, linestyle="--", label="97.5% CI")
plt.axvline(test_mae, linestyle="-", label="Test MAE")
plt.title("Bootstrap Distribution of Test MAE")
plt.xlabel("MAE")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "test_mae_bootstrap_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ===== Notebook code cell 74 =====
# ============================================================
# 34. Actual vs Predicted plots
# ============================================================

plt.figure(figsize=(14, 5))
plt.plot(predictions_df["datetime"], predictions_df["Y"], label="Actual", alpha=0.7)
plt.plot(predictions_df["datetime"], predictions_df["Predicted_Y"], label="Predicted", alpha=0.7)
plt.title(f"Actual vs Predicted Voltage Deviation - {best_model_name}")
plt.xlabel("Datetime")
plt.ylabel("Voltage Deviation from 230 V")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "actual_vs_predicted_full_test.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plot_sample = predictions_df.sort_values("datetime").head(1000)

plt.figure(figsize=(14, 5))
plt.plot(plot_sample["datetime"], plot_sample["Y"], label="Actual", alpha=0.8)
plt.plot(plot_sample["datetime"], plot_sample["Predicted_Y"], label="Predicted", alpha=0.8)
plt.title("Actual vs Predicted Voltage Deviation - First 1000 Test Observations")
plt.xlabel("Datetime")
plt.ylabel("Voltage Deviation from 230 V")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "actual_vs_predicted_first_1000.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ===== Notebook code cell 76 =====
# ============================================================
# 35. Residual analysis
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(predictions_df["Residual"], bins=50)
plt.title("Residual Distribution")
plt.xlabel("Residual: Actual - Predicted")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "residual_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(predictions_df["Predicted_Y"], predictions_df["Residual"], alpha=0.3)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Voltage Deviation")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "residuals_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

residual_summary = predictions_df["Residual"].describe()
print("\nResidual summary:")
print(residual_summary)

residual_summary.to_excel(
    os.path.join(model_output_dir, "residual_summary.xlsx")
)

residual_summary.to_csv(
    os.path.join(model_output_dir, "residual_summary.csv")
)

error_by_hour = predictions_df.groupby("hour").agg(
    MAE=("ABS_Residual", "mean"),
    Bias=("Residual", "mean"),
    Count=("Residual", "count")
).reset_index()

display(error_by_hour)

error_by_hour.to_excel(
    os.path.join(model_output_dir, "error_by_hour.xlsx"),
    index=False
)

error_by_hour.to_csv(
    os.path.join(model_output_dir, "error_by_hour.csv"),
    index=False
)

error_by_customer = predictions_df.groupby("SerialNo").agg(
    MAE=("ABS_Residual", "mean"),
    Bias=("Residual", "mean"),
    Count=("Residual", "count")
).reset_index().sort_values("MAE", ascending=False)

display(error_by_customer.head(20))

error_by_customer.to_excel(
    os.path.join(model_output_dir, "error_by_customer.xlsx"),
    index=False
)

error_by_customer.to_csv(
    os.path.join(model_output_dir, "error_by_customer.csv"),
    index=False
)


# ===== Notebook code cell 78 =====
# ============================================================
# 36. Operational regime error analysis
# ============================================================

pv_median_test = predictions_df["PV"].median()
load_median_test = predictions_df["Load"].median()


def assign_operational_regime(row):
    if row["PV"] >= pv_median_test and row["Load"] < load_median_test:
        return "High PV / Low Load"
    elif row["PV"] < pv_median_test and row["Load"] >= load_median_test:
        return "Low PV / High Load"
    elif row["PV"] >= pv_median_test and row["Load"] >= load_median_test:
        return "High PV / High Load"
    else:
        return "Low PV / Low Load"


predictions_df["Operational_Regime"] = predictions_df.apply(
    assign_operational_regime,
    axis=1
)

regime_error_df = predictions_df.groupby("Operational_Regime").agg(
    Count=("Residual", "count"),
    MAE=("ABS_Residual", "mean"),
    RMSE=("Residual", lambda x: np.sqrt(np.mean(x ** 2))),
    Bias=("Residual", "mean"),
    Mean_Actual_Y=("Y", "mean"),
    Mean_Predicted_Y=("Predicted_Y", "mean"),
    Mean_PV=("PV", "mean"),
    Mean_Load=("Load", "mean")
).reset_index()

regime_error_df = regime_error_df.sort_values("MAE", ascending=False).round(4)

display(regime_error_df)

regime_error_df.to_excel(
    os.path.join(model_output_dir, "operational_regime_error_analysis.xlsx"),
    index=False
)

regime_error_df.to_csv(
    os.path.join(model_output_dir, "operational_regime_error_analysis.csv"),
    index=False
)

plt.figure(figsize=(10, 5))
plt.bar(regime_error_df["Operational_Regime"], regime_error_df["MAE"])
plt.title("MAE by Operational Regime")
plt.xlabel("Operational Regime")
plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(model_output_dir, "mae_by_operational_regime.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

predictions_df.to_excel(predictions_output_xlsx, index=False)
predictions_df.to_csv(predictions_output_csv, index=False)


# ===== Notebook code cell 80 =====
# ============================================================
# 37. Feature importance
# ============================================================

if hasattr(final_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": full_features,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    display(importance_df)

    importance_df.to_excel(
        os.path.join(model_output_dir, "final_model_feature_importance.xlsx"),
        index=False
    )

    importance_df.to_csv(
        os.path.join(model_output_dir, "final_model_feature_importance.csv"),
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Built-in Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_output_dir, "built_in_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
else:
    print("\nFinal model does not provide built-in feature_importances_.")


# ===== Notebook code cell 82 =====
# ============================================================
# 38. SHAP analysis
# ============================================================

X_test_shap_full = test_df[full_features].copy()

SHAP_SAMPLE_SIZE = len(X_test_shap_full)

print("\n================ SHAP START ================")
print("SHAP model:", best_model_name)
print("SHAP sample size:", SHAP_SAMPLE_SIZE)

X_test_shap = X_test_shap_full.copy()

start_time = time.time()

if best_model_name == "Linear Regression":
    explainer = shap.LinearExplainer(final_model, X_train_val_full)
    shap_values = explainer.shap_values(X_test_shap)
else:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(
        X_test_shap,
        check_additivity=False
    )

end_time = time.time()

print("SHAP calculation time in minutes:", round((end_time - start_time) / 60, 2))
print("SHAP values shape:", np.array(shap_values).shape)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": full_features,
    "Mean_ABS_SHAP": mean_abs_shap
}).sort_values("Mean_ABS_SHAP", ascending=False).reset_index(drop=True)

display(shap_importance_df)

shap_importance_df.to_excel(
    os.path.join(shap_output_dir, "shap_global_importance.xlsx"),
    index=False
)

shap_importance_df.to_csv(
    os.path.join(shap_output_dir, "shap_global_importance.csv"),
    index=False
)

shap.summary_plot(
    shap_values,
    X_test_shap,
    plot_type="bar",
    show=False
)

plt.title(f"SHAP Global Feature Importance - {best_model_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(shap_output_dir, "shap_global_bar.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

shap.summary_plot(
    shap_values,
    X_test_shap,
    show=False
)

plt.title(f"SHAP Summary Plot - {best_model_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(shap_output_dir, "shap_summary_beeswarm.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

for feature in full_features:
    shap.dependence_plot(
        feature,
        shap_values,
        X_test_shap,
        show=False
    )

    plt.title(f"SHAP Dependence Plot - {feature}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(shap_output_dir, f"shap_dependence_{feature}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

if "PV" in X_test_shap.columns and "Load" in X_test_shap.columns:
    shap.dependence_plot(
        "PV",
        shap_values,
        X_test_shap,
        interaction_index="Load",
        show=False
    )

    plt.title("SHAP Dependence Plot - PV with Load Interaction")
    plt.tight_layout()
    plt.savefig(
        os.path.join(shap_output_dir, "shap_dependence_PV_interaction_Load.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    shap.dependence_plot(
        "Load",
        shap_values,
        X_test_shap,
        interaction_index="PV",
        show=False
    )

    plt.title("SHAP Dependence Plot - Load with PV Interaction")
    plt.tight_layout()
    plt.savefig(
        os.path.join(shap_output_dir, "shap_dependence_Load_interaction_PV.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 38.X SHAP interaction values - all pairwise interactions
# ============================================================

if best_model_name != "Linear Regression":
    print("\nComputing SHAP interaction values for all feature pairs...")

    shap_interaction_values = explainer.shap_interaction_values(
        X_test_shap
    )

    # --------------------------------------------------------
    # 1. Mean absolute interaction matrix
    # --------------------------------------------------------

    interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)

    interaction_matrix_df = pd.DataFrame(
        interaction_matrix,
        index=full_features,
        columns=full_features
    )

    display(interaction_matrix_df)

    interaction_matrix_df.to_excel(
        os.path.join(shap_output_dir, "shap_interaction_matrix.xlsx")
    )

    interaction_matrix_df.to_csv(
        os.path.join(shap_output_dir, "shap_interaction_matrix.csv")
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(interaction_matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(full_features)), full_features, rotation=45, ha="right")
    plt.yticks(range(len(full_features)), full_features)
    plt.title("Mean Absolute SHAP Interaction Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(shap_output_dir, "shap_interaction_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # --------------------------------------------------------
    # 2. Top pairwise interactions table
    # --------------------------------------------------------

    interaction_pairs = []

    for i in range(len(full_features)):
        for j in range(i + 1, len(full_features)):
            interaction_pairs.append({
                "Feature_1": full_features[i],
                "Feature_2": full_features[j],
                "Mean_ABS_SHAP_Interaction": interaction_matrix[i, j]
            })

    interaction_pairs_df = pd.DataFrame(interaction_pairs).sort_values(
        "Mean_ABS_SHAP_Interaction",
        ascending=False
    ).reset_index(drop=True)

    display(interaction_pairs_df)

    interaction_pairs_df.to_excel(
        os.path.join(shap_output_dir, "shap_top_interaction_pairs.xlsx"),
        index=False
    )

    interaction_pairs_df.to_csv(
        os.path.join(shap_output_dir, "shap_top_interaction_pairs.csv"),
        index=False
    )

    # --------------------------------------------------------
    # 3. Dependence plots for all PV interactions
    # --------------------------------------------------------

    pv_interaction_features = [
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed",
        "Load"
    ]

    for interaction_feature in pv_interaction_features:
        shap.dependence_plot(
            "PV",
            shap_values,
            X_test_shap,
            interaction_index=interaction_feature,
            show=False
        )

        plt.title(f"SHAP Dependence Plot - PV with {interaction_feature} Interaction")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                shap_output_dir,
                f"shap_dependence_PV_interaction_{interaction_feature}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    # --------------------------------------------------------
    # 4. Dependence plots for Load with weather and PV
    # --------------------------------------------------------

    load_interaction_features = [
        "SolarRad",
        "TempOut",
        "OutHum",
        "WindSpeed",
        "PV"
    ]

    for interaction_feature in load_interaction_features:
        shap.dependence_plot(
            "Load",
            shap_values,
            X_test_shap,
            interaction_index=interaction_feature,
            show=False
        )

        plt.title(f"SHAP Dependence Plot - Load with {interaction_feature} Interaction")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                shap_output_dir,
                f"shap_dependence_Load_interaction_{interaction_feature}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

else:
    print("\nSHAP interaction values are not computed for Linear Regression.")

shap_values_df = pd.DataFrame(
    shap_values,
    columns=[f"SHAP_{feature}" for feature in full_features],
    index=X_test_shap.index
)

shap_values_df.to_excel(
    os.path.join(shap_output_dir, "shap_values_sample.xlsx"),
    index=True
)

shap_values_df.to_csv(
    os.path.join(shap_output_dir, "shap_values_sample.csv"),
    index=True
)

local_df = predictions_df.loc[X_test_shap.index].copy()
local_df["ABS_Residual"] = local_df["Residual"].abs()

top_error_indices = local_df.sort_values(
    "ABS_Residual",
    ascending=False
).head(5).index

local_explanations = []

for idx in top_error_indices:
    sample_position = list(X_test_shap.index).index(idx)

    local_explanation_df = pd.DataFrame({
        "Index": idx,
        "Feature": full_features,
        "Feature_Value": X_test_shap.loc[idx, full_features].values,
        "SHAP_Value": shap_values[sample_position],
        "Datetime": local_df.loc[idx, "datetime"],
        "Actual_Y": local_df.loc[idx, "Y"],
        "Predicted_Y": local_df.loc[idx, "Predicted_Y"],
        "Residual": local_df.loc[idx, "Residual"]
    })

    local_explanations.append(local_explanation_df)

local_explanations_df = pd.concat(local_explanations, axis=0)

display(local_explanations_df)

local_explanations_df.to_excel(
    os.path.join(shap_output_dir, "local_shap_high_error_observations.xlsx"),
    index=False
)

local_explanations_df.to_csv(
    os.path.join(shap_output_dir, "local_shap_high_error_observations.csv"),
    index=False
)

print("================ SHAP FINISHED ================\n")


# ===== Notebook code cell 84 =====
# ============================================================
# 39. Save Optuna best parameters
# ============================================================

optuna_params_df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM"
    ],
    "Best_Params": [
        str(study_rf.best_params),
        str(study_gb.best_params),
        str(study_xgb.best_params),
        str(study_lgbm.best_params)
    ],
    "Best_Validation_MAE": [
        study_rf.best_value,
        study_gb.best_value,
        study_xgb.best_value,
        study_lgbm.best_value
    ]
})

display(optuna_params_df)

optuna_params_df.to_excel(
    os.path.join(model_output_dir, "optuna_best_parameters.xlsx"),
    index=False
)

optuna_params_df.to_csv(
    os.path.join(model_output_dir, "optuna_best_parameters.csv"),
    index=False
)


# ===== Notebook code cell 86 =====
# ============================================================
# 40. Final summary
# ============================================================

summary_info = pd.DataFrame({
    "Item": [
        "Pipeline type",
        "Raw customer rows",
        "Rows in constructed model-only dataset",
        "Rows after modelling cleaning",
        "Removed rows in modelling cleaning",
        "Removed percentage in modelling cleaning",
        "Number of customer endpoints",
        "Start datetime",
        "End datetime",
        "Positive Y percentage",
        "Negative Y percentage",
        "Best model based on validation",
        "Best model on final test set",
        "Validation-selected model same as best test model",
        "Final feature set",
        "Final Test MAE",
        "Final Test RMSE",
        "Final Test R²",
        "Best Test MAE across all models",
        "Best Test RMSE across all models",
        "Best Test R² across all models",
        "Test MAE 95% CI Lower",
        "Test MAE 95% CI Upper",
        "TimeSeries CV Average MAE",
        "TimeSeries CV STD MAE",
        "Best operational regime by MAE",
        "Worst operational regime by MAE",
        "SHAP sample size",
        "Most important SHAP feature"
    ],
    "Value": [
        "End-to-end raw data to modelling pipeline",
        len(customer_df),
        len(model_only_df),
        clean_rows,
        removed_rows,
        round(removed_percent, 2),
        df_clean["SerialNo"].nunique(),
        df_clean["datetime"].min(),
        df_clean["datetime"].max(),
        round(positive_y_pct, 2),
        round(negative_y_pct, 2),
        best_model_name,
        best_test_model_name,
        best_model_name == best_test_model_name,
        full_feature_set_name,
        round(test_mae, 4),
        round(test_rmse, 4),
        round(test_r2, 4),
        round(best_test_mae, 4),
        round(best_test_rmse, 4),
        round(best_test_r2, 4),
        round(mae_ci_lower, 4),
        round(mae_ci_upper, 4),
        round(cv_summary_df["Average MAE"].iloc[0], 4),
        round(cv_summary_df["STD MAE"].iloc[0], 4),
        regime_error_df.sort_values("MAE", ascending=True).iloc[0]["Operational_Regime"],
        regime_error_df.sort_values("MAE", ascending=False).iloc[0]["Operational_Regime"],
        len(X_test_shap),
        shap_importance_df.iloc[0]["Feature"]
    ]
})

display(summary_info)

summary_output_xlsx = os.path.join(
    base_output_dir,
    "thesis_end_to_end_summary.xlsx"
)

summary_output_csv = os.path.join(
    base_output_dir,
    "thesis_end_to_end_summary.csv"
)

summary_info.to_excel(summary_output_xlsx, index=False)
summary_info.to_csv(summary_output_csv, index=False)


# ===== Notebook code cell 88 =====
# ============================================================
# 41. Output location confirmation
# ============================================================

print("Outputs remain in the local output folder.")


# ===== Notebook code cell 90 =====
# ============================================================
# 42. Final print
# ============================================================

print("\nPipeline finished successfully.")

print("\nMain local output folder:")
print(base_output_dir)

print("\nOptional backup output folder:")
print(base_output_dir)

print("\nKey generated files:")
print(os.path.join(dataset_output_dir, "final_dataset_model_only_correct_target.xlsx"))
print(validation_output_xlsx)
print(ablation_output_xlsx)
print(final_test_output_xlsx)
print(predictions_output_xlsx)
print(summary_output_xlsx)
print(os.path.join(model_output_dir, "final_test_comparison_all_models.xlsx"))
print(os.path.join(model_output_dir, "test_selection_note.xlsx"))
print(os.path.join(model_output_dir, "untuned_vs_tuned_comparison.xlsx"))
print(os.path.join(model_output_dir, "optuna_best_parameters.xlsx"))
print(os.path.join(model_output_dir, "test_mae_bootstrap_confidence_interval.xlsx"))
print(os.path.join(model_output_dir, "operational_regime_error_analysis.xlsx"))
print(os.path.join(model_output_dir, "timeseries_cv_results_best_model.xlsx"))
print(os.path.join(model_output_dir, "timeseries_cv_summary_best_model.xlsx"))
print(os.path.join(shap_output_dir, "shap_global_importance.xlsx"))
print(os.path.join(shap_output_dir, "shap_global_bar.png"))
print(os.path.join(shap_output_dir, "shap_summary_beeswarm.png"))
print(os.path.join(shap_output_dir, "shap_dependence_PV_interaction_Load.png"))
print(os.path.join(shap_output_dir, "shap_dependence_Load_interaction_PV.png"))
print(os.path.join(shap_output_dir, "shap_interaction_matrix.xlsx"))
print(os.path.join(shap_output_dir, "shap_interaction_matrix.png"))
print(os.path.join(shap_output_dir, "shap_top_interaction_pairs.xlsx"))
print(os.path.join(shap_output_dir, "shap_dependence_PV_interaction_SolarRad.png"))
print(os.path.join(shap_output_dir, "shap_dependence_PV_interaction_TempOut.png"))
print(os.path.join(shap_output_dir, "shap_dependence_PV_interaction_OutHum.png"))
print(os.path.join(shap_output_dir, "shap_dependence_PV_interaction_WindSpeed.png"))
print(os.path.join(shap_output_dir, "shap_dependence_Load_interaction_SolarRad.png"))
print(os.path.join(shap_output_dir, "shap_dependence_Load_interaction_TempOut.png"))
print(os.path.join(shap_output_dir, "shap_dependence_Load_interaction_OutHum.png"))
print(os.path.join(shap_output_dir, "shap_dependence_Load_interaction_WindSpeed.png"))


