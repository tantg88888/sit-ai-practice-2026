FROM ghcr.io/mlflow/mlflow:v3.13.0

# One teaching image for Day 4:
# - MLflow tracking and model registry
# - FastAPI dashboard serving
# - Feast feature store demo
# - scikit-learn model training scripts
RUN pip install --no-cache-dir \
    duckdb \
    geopandas \
    feast \
    pyarrow \
    "dbt-core>=1.9,<1.10" \
    "dbt-sqlite>=1.9,<1.10" \
    "scipy>=1.15" \
    "scikit-learn>=1.6" \
    "numexpr>=2.10" \
    "bottleneck>=1.4" \
    fastapi \
    uvicorn \
    requests \
    pandas

WORKDIR /workspace
