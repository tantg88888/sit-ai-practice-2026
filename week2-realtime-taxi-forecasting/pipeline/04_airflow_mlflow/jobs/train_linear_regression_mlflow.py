import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "v0": "taxi_weather_2h_prediction_dataset_0606_rainfall.csv",
    "v1": "taxi_weather_2h_prediction_dataset_0617_rainfall.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model and log it to MLflow."
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-version", choices=sorted(DATASETS), default="v1")
    parser.add_argument("--feature-version", choices=["v0", "v1"], default="v1")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment-name", default="taxi_availability_2h_prediction")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--model-name", default="taxi_availability_2h_model")
    parser.add_argument("--model-alias", default="champion")
    return parser.parse_args()


def resolve_data_path(args):
    path_value = args.data_path or DATASETS[args.data_version]
    data_path = Path(path_value)
    if data_path.is_absolute():
        return data_path
    return (SCRIPT_DIR / data_path).resolve()


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


def regression_metrics(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5
    return mae, rmse


def set_model_alias(client, model_name, alias, version):
    client.set_registered_model_alias(model_name, alias, version)


def build_logged_model_name(args):
    return f"LR_data_{args.data_version}_features_{args.feature_version}"


def main():
    args = parse_args()

    try:
        import mlflow
        import mlflow.sklearn
    except ModuleNotFoundError as exc:
        raise SystemExit("MLflow is not installed. Run: pip install mlflow") from exc

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    features = select_features(args.feature_version)
    target_column = "target_taxi_count_2h"

    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    run_name = (
        args.run_name
        or f"linear_regression_data_{args.data_version}_features_{args.feature_version}"
    )
    logged_model_name = build_logged_model_name(args)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("=" * 72)
    print("Taxi Availability Prediction: MLflow Training Script")
    print("=" * 72)
    print(f"Run name: {run_name}")
    print(f"Data version: {args.data_version}")
    print(f"Dataset path: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Feature version: {args.feature_version}")
    print(f"Features: {','.join(features)}")
    print(f"Target column: {target_column}")
    print()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("feature_version", args.feature_version)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("data_path", data_path.name)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("logged_model_name", logged_model_name)

        print("Training Linear Regression")
        print("-" * 72)
        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        train_mae, _ = regression_metrics(y_train, train_predictions)
        mae, rmse = regression_metrics(y_test, test_predictions)

        mlflow.log_metric("train_MAE", train_mae)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        if args.register_model:
            print("Skipping model artifact upload and registration in Airflow demo mode.")

        print()
        print("Final test results")
        print("-" * 72)
        print(f"Train MAE: {train_mae:.3f}")
        print(f"MAE:       {mae:.3f}")
        print(f"RMSE:      {rmse:.3f}")
        print()
        print("Logged to MLflow.")


if __name__ == "__main__":
    main()
