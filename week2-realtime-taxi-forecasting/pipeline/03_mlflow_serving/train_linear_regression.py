import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# This is the plain training script.
# It does NOT use MLflow yet.
#
# The purpose of this file is to show the normal machine learning workflow:
# 1. load a prepared training table;
# 2. choose input features and output target;
# 3. split into train and test sets;
# 4. preprocess categorical and numeric columns;
# 5. train a regression model;
# 6. print metrics in the terminal.
#
# Later, train_linear_regression_mlflow.py will keep the same idea but add
# MLflow tracking, dataset logging, model logging, and optional registration.
SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "v0": "taxi_weather_2h_prediction_dataset_0606_rainfall.csv",
    "v1": "taxi_weather_2h_prediction_dataset_0617_rainfall.csv",
}

def parse_args():
    # Command-line arguments make the script reusable.
    # Change simple training settings without editing Python code.
    parser = argparse.ArgumentParser(
        description="Train a simple linear regression model for 2-hour taxi availability prediction."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to a training dataset CSV. If omitted, --data-version is used.",
    )
    parser.add_argument(
        "--data-version",
        choices=sorted(DATASETS),
        default="v1",
        help="Dataset version. v0=0606, v1=0617.",
    )
    parser.add_argument(
        "--feature-version",
        choices=["v0", "v1"],
        default="v1",
        help=(
            "Feature set version. v0 uses simple time features; "
            "v1 matches the PPT feature set with rain forecast, current rainfall, and current taxi count."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for the test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test split and model training.",
    )
    return parser.parse_args()


def resolve_data_path(args):
    # If the user gives an absolute path, use it directly.
    # Otherwise, treat the path as relative to this script folder.
    path_value = args.data_path or DATASETS[args.data_version]
    data_path = Path(path_value)
    if data_path.is_absolute():
        return data_path
    return (SCRIPT_DIR / data_path).resolve()


def regression_metrics(y_true, predictions):
    # Regression metrics used throughout this lesson.
    # MAE is the easiest to explain: average error in number of taxis.
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5
    return mae, rmse


def select_features(feature_version):
    if feature_version == "v0":
        return ["planning_area_id", "hour_of_day", "day_of_week", "is_weekend"]
    if feature_version == "v1":
        return [
            "planning_area_id",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_rain_forecast_2h",
            "current_rainfall_5min_mm",
            "current_taxi_count",
        ]
    raise ValueError(f"Unknown feature version: {feature_version}")


def main():
    args = parse_args()

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Load the selected dataset and add the same simple engineered features used
    # by train_linear_regression_mlflow.py.
    # Each row represents one planning area at one timestamp.
    df = pd.read_csv(data_path, low_memory=False)
    features = select_features(args.feature_version)

    # The target is the value we want to predict:
    # taxi availability in the same planning area about 2 hours later.
    target_column = "target_taxi_count_2h"

    # X means input features.
    # y means target/output.
    X = df[features]
    y = df[target_column]

    print("=" * 72)
    print("Taxi Availability Prediction: Plain Training Script")
    print("=" * 72)
    print(f"Data version: {args.data_version}")
    print(f"Dataset path: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Feature version: {args.feature_version}")
    print(f"Features: {','.join(features)}")
    print(f"Target column: {target_column}")
    print()

    # Split the dataset into training data and test data.
    # The model learns from the training data.
    # We evaluate the model on the test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Keep preprocessing and model together, matching the MLflow version.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    train_mae, _ = regression_metrics(y_train, train_predictions)

    # Final evaluation.
    final_predictions = test_predictions
    final_mae, final_rmse = regression_metrics(y_test, final_predictions)

    print()
    print("Training Linear Regression")
    print("-" * 72)
    print(f"Train MAE: {train_mae:.3f}")
    print()
    print("Final test results")
    print("-" * 72)
    print(f"MAE:  {final_mae:.3f}")
    print(f"RMSE: {final_rmse:.3f}")
    print()

    # Show a few predictions so the metrics connect back to concrete rows.
    preview = X_test.copy()
    preview["actual_target_taxi_count_2h"] = y_test.values
    preview["predicted_target_taxi_count_2h"] = final_predictions.round(1)
    preview_columns = features + [
        "actual_target_taxi_count_2h",
        "predicted_target_taxi_count_2h",
    ]

    print("Prediction preview")
    print("-" * 72)
    print(preview[preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
