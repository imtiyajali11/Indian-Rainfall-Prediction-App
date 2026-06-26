
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("ignore")


DEFAULT_DATA_PATH = "Indian Rainfall Dataset District-wise Daily Measurements.csv"
DEFAULT_OUTPUT_DIR = "artifacts"
RANDOM_STATE = 42


def get_day_suffix(day: int) -> str:
   
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return data


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load dataset using automatic separator detection."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {data_path}\n\n"
            "Fix:\n"
            "1. Keep the CSV file in the same folder as train_model.py, OR\n"
            "2. Pass the file path manually:\n"
            '   python3 train_model.py --data "path/to/your_file.csv"\n'
        )

    data = pd.read_csv(data_path, sep=None, engine="python")
    return data


def detect_day_columns(data: pd.DataFrame) -> List[str]:
    """
    Detect daily rainfall columns.

    Supported formats:
    - 1, 2, 3, ..., 31
    - 1st, 2nd, 3rd, ..., 31st
    - day_1, day_2, ..., day_31
    - d1, d2, ..., d31
    """
    possible_day_cols = []

    for day in range(1, 32):
        possible_day_cols.extend(
            [
                str(day),
                f"{day}{get_day_suffix(day)}",
                f"day_{day}",
                f"d{day}",
            ]
        )

    day_cols = [col for col in possible_day_cols if col in data.columns]

    if len(day_cols) == 0:
        raise ValueError(
            "\nNo daily rainfall columns were detected.\n\n"
            "Expected columns like:\n"
            "- 1st, 2nd, 3rd, ..., 31st\n"
            "- 1, 2, 3, ..., 31\n"
            "- day_1, day_2, ..., day_31\n"
            "- d1, d2, ..., d31\n\n"
            f"Available columns are:\n{data.columns.tolist()}\n"
        )

    return day_cols


def standardize_month_column(data: pd.DataFrame) -> pd.DataFrame:
    """Convert month names to month numbers if the month column is textual."""
    data = data.copy()

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    if data["month"].dtype == "object":
        month_text = data["month"].astype(str).str.strip().str.lower()
        mapped_month = month_text.map(month_map)

        # If mapping worked for at least some rows, use mapped values.
        # Otherwise try direct numeric conversion.
        if mapped_month.notna().sum() > 0:
            data["month"] = mapped_month
        else:
            data["month"] = pd.to_numeric(data["month"], errors="coerce")
    else:
        data["month"] = pd.to_numeric(data["month"], errors="coerce")

    data = data.dropna(subset=["month"])
    data["month"] = data["month"].astype(int)

    return data


def prepare_training_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Prepare model features and target.

    Features:
    - state
    - district
    - month

    Target:
    - total_rainfall = sum of all detected daily rainfall columns
    """
    data = clean_column_names(data)

    required_columns = ["state", "district", "month"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns are: {data.columns.tolist()}\n"
        )

    data = standardize_month_column(data)
    day_cols = detect_day_columns(data)

    for col in day_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[day_cols] = data[day_cols].fillna(0)

    # Target from notebook: monthly total rainfall
    data["total_rainfall"] = data[day_cols].sum(axis=1)

    model_data = data[["state", "district", "month", "total_rainfall"]].copy()
    model_data = model_data.dropna()

    model_data["state"] = model_data["state"].astype(str).str.strip()
    model_data["district"] = model_data["district"].astype(str).str.strip()
    model_data["month"] = model_data["month"].astype(int)

    X = model_data[["state", "district", "month"]]
    y = model_data["total_rainfall"]

    return X, y, model_data, day_cols


def build_pipeline(model) -> Pipeline:
    """Create preprocessing and model pipeline."""
    categorical_features = ["state", "district"]
    numeric_features = ["month"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def get_models() -> Dict[str, object]:
    """Return candidate models for comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2_Score": round(float(r2), 4),
    }


def train_and_compare_models(X: pd.DataFrame, y: pd.Series) -> Tuple[str, Pipeline, pd.DataFrame]:
    """Train multiple models and return the best model based on RMSE."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = []
    trained_models = {}

    for model_name, model in get_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, results_df


def create_metadata(model_data: pd.DataFrame, day_cols: List[str], best_model_name: str) -> Dict[str, object]:
    """Create metadata required by the Streamlit application."""
    states = sorted(model_data["state"].astype(str).unique().tolist())
    districts = sorted(model_data["district"].astype(str).unique().tolist())

    districts_by_state = {}

    for state in states:
        state_districts = (
            model_data.loc[model_data["state"].astype(str) == state, "district"]
            .astype(str)
            .unique()
            .tolist()
        )
        districts_by_state[state] = sorted(state_districts)

    metadata = {
        "best_model": best_model_name,
        "features": ["state", "district", "month"],
        "target": "total_rainfall",
        "day_columns": day_cols,
        "states": states,
        "districts": districts,
        "districts_by_state": districts_by_state,
        "total_rows": int(model_data.shape[0]),
    }

    return metadata


def save_artifacts(
    best_model: Pipeline,
    results_df: pd.DataFrame,
    metadata: Dict[str, object],
    output_dir: Path,
) -> None:
    """Save trained model, model results, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rainfall_model.joblib"
    results_path = output_dir / "model_results.csv"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(best_model, model_path)
    results_df.to_csv(results_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSaved files:")
    print(f"- {model_path}")
    print(f"- {results_path}")
    print(f"- {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Indian Rainfall Prediction model.")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to rainfall CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where trained model artifacts will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    print("Starting training pipeline...")
    print(f"Dataset path: {data_path}")

    data = load_dataset(data_path)
    print(f"Dataset loaded successfully. Shape: {data.shape}")

    X, y, model_data, day_cols = prepare_training_data(data)

    print(f"Detected day columns: {len(day_cols)}")
    print(f"Training rows: {len(model_data)}")
    print("Features:", X.columns.tolist())
    print("Target: total_rainfall")

    best_model_name, best_model, results_df = train_and_compare_models(X, y)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest model selected: {best_model_name}")

    metadata = create_metadata(model_data, day_cols, best_model_name)
    save_artifacts(best_model, results_df, metadata, output_dir)

    print("\nTraining completed successfully.")
    print("\nNext step:")
    print("Run your Streamlit app with:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()
