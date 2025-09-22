from pathlib import Path
import os, json
import pandas as pd
import mlflow
from sklearn.datasets import load_breast_cancer

EXPERIMENT = "BC Wisconsin - Data Validation"

def main():
    # Use a base path to construct relative paths
    base_path = Path(__file__).resolve().parents[2]
    artifacts = base_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Load dataset from sklearn
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame

    # Save raw dataset
    raw_path = artifacts / "raw_bc.csv"
    df.to_csv(raw_path, index=False)

    # Create validation report
    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "target_counts": {int(k): int(v) for k, v in df["target"].value_counts().to_dict().items()},
    }
    report_path = artifacts / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="01_data_validation"):
        # Log params and metrics
        mlflow.log_param("dataset", "breast_cancer_wisconsin")
        mlflow.log_metric("n_rows", report["n_rows"])
        mlflow.log_metric("n_cols", report["n_cols"])
        mlflow.log_metric("missing_total", report["missing_total"])
        for k, v in report["target_counts"].items():
            mlflow.log_metric(f"target_count_{k}", v)

        # Log artifacts using paths relative to the project root
        mlflow.log_artifact(str(raw_path.relative_to(base_path)))
        mlflow.log_artifact(str(report_path.relative_to(base_path)))

    print(f"[OK] Saved raw -> {raw_path}")
    print(f"[OK] Saved report -> {report_path}")

if __name__ == "__main__":
    main()
