import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_linear_regression_mlflow import (
    DATASETS,
    resolve_data_path,
    select_features,
    set_model_alias,
)


SCRIPT_DIR = Path(__file__).resolve().parent


# This script is a second model example for students.
# It uses the same dataset, same feature engineering, and same MLflow workflow
# as train_linear_regression_mlflow.py, but changes the model type to MLPRegressor.
# MLP means "multi-layer perceptron", a simple feed-forward neural network.
def parse_args():
    # Command-line arguments let students run model experiments without editing code.
    # This is important for MLflow because each different setting becomes a new run.
    parser = argparse.ArgumentParser(
        description="Train a simple MLP regression model and log it to MLflow."
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
    parser.add_argument("--feature-version", choices=["v0", "v1"], default="v1")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment-name", default="taxi_availability_2h_prediction")
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name shown in the UI. If omitted, it is generated from data and feature versions.",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        default="32,16",
        help="Comma-separated hidden layer sizes, for example 32,16.",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=0.0001,
        help="Initial learning rate used by the optimizer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="L2 regularization strength.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use early stopping during MLP training.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data used for early-stopping validation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the logged model into MLflow Model Registry.",
    )
    parser.add_argument(
        "--model-name",
        default="taxi_availability_2h_model",
        help="Registered model name used when --register-model is set.",
    )
    parser.add_argument(
        "--model-alias",
        default="champion",
        help="Model Registry alias assigned to the registered model version.",
    )
    return parser.parse_args()


def parse_hidden_layers(value):
    # Convert a classroom-friendly string like "32,16" into a Python tuple (32, 16).
    # In MLPRegressor, this means two hidden layers:
    # first hidden layer has 32 neurons, second hidden layer has 16 neurons.
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def regression_metrics(y_true, predictions):
    # Same metrics as the linear regression script, so students can compare models.
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    return mae, rmse


def make_run_name(args):
    # Generate a readable run name for the MLflow UI.
    if args.run_name:
        return args.run_name
    return f"mlp_regression_data_{args.data_version}_features_{args.feature_version}"


def build_logged_model_name(args):
    return f"MLP_data_{args.data_version}_features_{args.feature_version}"


def log_loss_curve(mlflow, trained_pipeline):
    mlp_model = trained_pipeline.named_steps["model"]
    loss_curve = getattr(mlp_model, "loss_curve_", None)
    if not loss_curve:
        return 0

    for step, loss in enumerate(loss_curve, start=1):
        mlflow.log_metric("training_loss", float(loss), step=step)
    mlflow.log_param("training_loss_steps", len(loss_curve))

    loss_df = pd.DataFrame(
        {
            "step": list(range(1, len(loss_curve) + 1)),
            "training_loss": [float(loss) for loss in loss_curve],
        }
    )
    loss_dir = SCRIPT_DIR / "temp_loss_curve"
    loss_dir.mkdir(exist_ok=True)

    csv_path = loss_dir / "training_loss_curve.csv"
    loss_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path="loss_curve")

    try:
        import matplotlib.pyplot as plt

        png_path = loss_dir / "training_loss_curve.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(loss_df["step"], loss_df["training_loss"], linewidth=2)
        ax.set_title("MLP Training Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(png_path), artifact_path="loss_curve")
    except Exception as exc:
        print(f"Could not create loss curve plot artifact: {exc}")

    return len(loss_curve)


def main():
    args = parse_args()

    # Keep Git-related warnings quiet inside the Docker classroom environment.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    try:
        # Import MLflow here so --help can still work even without MLflow installed.
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
    except ModuleNotFoundError as exc:
        raise SystemExit("MLflow is not installed. Run: pip install mlflow") from exc

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Reuse the same dbt-produced feature table as the linear regression script.
    # This keeps the comparison fair: only the model type changes.
    df = pd.read_csv(data_path, low_memory=False)
    features = select_features(args.feature_version)
    target_column = "target_taxi_count_2h"

    # Split into train and test sets.
    # The fixed random_state makes results reproducible for the lesson.
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[target_column],
        test_size=args.test_size,
        random_state=args.random_state,
    )

    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)

    # A pipeline stores both preprocessing and the neural network model.
    # StandardScaler is especially important for MLP models because neural networks
    # usually train better when numeric inputs are on a similar scale.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    # Network shape, for example (32, 16).
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    # Learning rate controls how large each training update step is.
                    learning_rate_init=args.learning_rate_init,
                    # alpha is L2 regularization. Larger alpha means stronger penalty
                    # against overly large model weights.
                    alpha=args.alpha,
                    # Maximum number of training iterations.
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    # Early stopping can stop training when validation performance
                    # stops improving.
                    early_stopping=args.early_stopping,
                    validation_fraction=args.validation_fraction,
                ),
            ),
        ]
    )

    # Configure MLflow Tracking.
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    run_name = make_run_name(args)
    logged_model_name = build_logged_model_name(args)

    print("=" * 72)
    print("Taxi Availability Prediction: MLP Regression Script")
    print("=" * 72)
    print(f"Run name: {run_name}")
    print(f"Data version: {args.data_version}")
    print(f"Feature version: {args.feature_version}")
    print(f"Features: {','.join(features)}")
    print(f"Target column: {target_column}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Learning rate init: {args.learning_rate_init}")
    print()

    with mlflow.start_run(run_name=run_name):
        # Parameters: these describe the training setup before the model is trained.
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("feature_version", args.feature_version)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("feature_count", len(features))
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("data_path", data_path.name)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("model_type", "MLPRegressor")
        mlflow.log_param("hidden_layer_sizes", str(hidden_layers))
        mlflow.log_param("activation", "relu")
        mlflow.log_param("solver", "adam")
        mlflow.log_param("learning_rate_init", args.learning_rate_init)
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("early_stopping", args.early_stopping)
        mlflow.log_param("validation_fraction", args.validation_fraction)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("logged_model_name", logged_model_name)

        # Train the neural network.
        pipeline.fit(X_train, y_train)
        loss_steps = log_loss_curve(mlflow, pipeline)

        # Evaluate on the test set.
        predictions = pipeline.predict(X_test)
        mae, rmse = regression_metrics(y_test, predictions)
        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))
        model_metadata = {
            "model_type": "MLPRegressor",
            "data_version": args.data_version,
            "feature_version": args.feature_version,
            "features": ",".join(features),
            "target_column": target_column,
            "MAE": f"{mae:.3f}",
            "RMSE": f"{rmse:.3f}"
        }

        # Metrics: these describe how well the trained model performed.
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # MLflow 3 logged model entry.
        # This makes the model appear under the experiment's Models tab.
        mlflow.sklearn.log_model(
            pipeline,
            name=logged_model_name,
            signature=signature,
            input_example=input_example,
            metadata=model_metadata,
        )

        model_dir = SCRIPT_DIR / "temp_mlp_model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(
            pipeline,
            model_dir,
            signature=signature,
            input_example=input_example,
            metadata=model_metadata,
        )
        mlflow.log_artifacts(str(model_dir), artifact_path=logged_model_name)

        #if args.register_model: some bug so always register the model for now, ie the last one will be champion
        if True:
            # Optional Model Registry step.
            # Only runs with --register-model become registered model versions.
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{logged_model_name}"
            registered_model = mlflow.register_model(model_uri, args.model_name)
            client = mlflow.tracking.MlflowClient()

            # Assign an alias such as "champion" to this version.
            # The alias is a movable pointer used by serving code.
            set_model_alias(
                client,
                args.model_name,
                args.model_alias,
                registered_model.version,
            )
            # Add tags to the run so the run page clearly shows registry information.
            mlflow.set_tag("registered_model_name", args.model_name)
            mlflow.set_tag("registered_model_version", registered_model.version)
            mlflow.set_tag("registered_model_alias", args.model_alias)
            print()
            print("Model Registry")
            print("-" * 72)
            print(f"Registered model name: {args.model_name}")
            print(f"Registered model version: {registered_model.version}")
            print(f"Alias: {args.model_alias}")
            print(f"Logged model name: {logged_model_name}")
            print(f"Model URI: models:/{args.model_name}@{args.model_alias}")

        print()
        print("Final test results")
        print("-" * 72)
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        if loss_steps:
            print(f"Training loss curve points logged: {loss_steps}")
            print("Open the run in MLflow and click Metrics -> training_loss.")


if __name__ == "__main__":
    main()
