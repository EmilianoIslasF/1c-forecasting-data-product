from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import awswrangler as wr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LOGGER = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "shop_id",
    "item_id",
    "category_id",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_mean_3",
    "lag_mean_6",
    "had_sales_lag_1",
    "had_sales_lag_3",
]

TARGET_COLUMN = "item_cnt_month"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML model and write forecasts to Gold."
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where model artifacts and Gold tables will be stored.",
    )

    parser.add_argument(
        "--ml-database",
        default="forecasting_ml",
        help="Glue database with ML feature tables.",
    )

    parser.add_argument(
        "--gold-database",
        default="forecasting_gold",
        help="Glue database for model outputs.",
    )

    parser.add_argument(
        "--gold-prefix",
        default="forecasting/gold",
        help="S3 prefix for Gold model output tables.",
    )

    parser.add_argument(
        "--artifacts-prefix",
        default="forecasting/artifacts/models",
        help="S3 prefix for model artifacts.",
    )

    parser.add_argument(
        "--local-model-dir",
        default="artifacts/models",
        help="Local directory to save model.joblib and metrics.json.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=1_500_000,
        help="Maximum training rows sampled for faster MVP training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def read_table(database: str, table: str) -> pd.DataFrame:
    LOGGER.info("Reading table: %s.%s", database, table)

    df = wr.s3.read_parquet_table(
        database=database,
        table=table,
    )

    assert not df.empty, f"Table {database}.{table} is empty"

    LOGGER.info(
        "Loaded %s.%s with %s rows and %s columns",
        database,
        table,
        len(df),
        len(df.columns),
    )

    return df


def validate_features(df: pd.DataFrame, table_name: str, require_target: bool) -> None:
    required_columns = FEATURE_COLUMNS.copy()

    if require_target:
        required_columns.append(TARGET_COLUMN)

    missing = [col for col in required_columns if col not in df.columns]

    assert not missing, f"{table_name} missing columns: {missing}"


def prepare_xy(
    df: pd.DataFrame,
    table_name: str,
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    validate_features(df, table_name, require_target=require_target)

    data = df.copy()

    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    x = data[FEATURE_COLUMNS]

    y = None
    if require_target:
        y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").fillna(0)
        y = y.clip(lower=0, upper=20)

    return x, y


def maybe_sample_train(
    train: pd.DataFrame,
    max_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(train) <= max_train_rows:
        LOGGER.info("Using full training set with %s rows", len(train))
        return train

    LOGGER.info(
        "Sampling training set from %s to %s rows",
        len(train),
        max_train_rows,
    )

    return train.sample(
        n=max_train_rows,
        random_state=seed,
    ).reset_index(drop=True)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> HistGradientBoostingRegressor:
    LOGGER.info("Training HistGradientBoostingRegressor")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )

    model.fit(x_train, y_train)

    LOGGER.info("Model training completed")

    return model


def predict_clipped(
    model: HistGradientBoostingRegressor,
    x: pd.DataFrame,
) -> np.ndarray:
    preds = model.predict(x)
    preds = np.clip(preds, 0, 20)
    return preds


def build_model_evaluation(
    validation: pd.DataFrame,
    model_predictions: np.ndarray,
) -> pd.DataFrame:
    LOGGER.info("Building model_evaluation")

    evaluation = validation.copy()

    evaluation["actual_item_cnt_month"] = pd.to_numeric(
        evaluation["item_cnt_month"],
        errors="coerce",
    ).fillna(0).clip(lower=0, upper=20)

    evaluation["model_prediction"] = model_predictions
    evaluation["baseline_prediction"] = pd.to_numeric(
        evaluation["lag_1"],
        errors="coerce",
    ).fillna(0).clip(lower=0, upper=20)

    evaluation["model_error"] = (
        evaluation["model_prediction"] - evaluation["actual_item_cnt_month"]
    )
    evaluation["baseline_error"] = (
        evaluation["baseline_prediction"] - evaluation["actual_item_cnt_month"]
    )

    evaluation["model_absolute_error"] = evaluation["model_error"].abs()
    evaluation["baseline_absolute_error"] = evaluation["baseline_error"].abs()

    evaluation["model_squared_error"] = evaluation["model_error"] ** 2
    evaluation["baseline_squared_error"] = evaluation["baseline_error"] ** 2

    keep_columns = [
        "date_block_num",
        "year",
        "month",
        "shop_id",
        "item_id",
        "category_id",
        "category_name",
        "item_name",
        "actual_item_cnt_month",
        "model_prediction",
        "baseline_prediction",
        "model_error",
        "baseline_error",
        "model_absolute_error",
        "baseline_absolute_error",
        "model_squared_error",
        "baseline_squared_error",
    ]

    available_columns = [col for col in keep_columns if col in evaluation.columns]

    evaluation = evaluation[available_columns].copy()

    LOGGER.info("model_evaluation has %s rows", len(evaluation))

    return evaluation


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_total": float(np.sum(y_pred)),
        "actual_total": float(np.sum(y_true)),
    }


def build_model_metrics_global(evaluation: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Building model_metrics_global")

    actual = evaluation["actual_item_cnt_month"]
    model_pred = evaluation["model_prediction"]
    baseline_pred = evaluation["baseline_prediction"]

    model_metrics = compute_metrics(actual, model_pred)
    baseline_metrics = compute_metrics(actual, baseline_pred)

    row = {
        "validation_date_block_num": int(evaluation["date_block_num"].max()),
        "n_rows": int(len(evaluation)),
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "model_r2": model_metrics["r2"],
        "model_prediction_total": model_metrics["prediction_total"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_r2": baseline_metrics["r2"],
        "baseline_prediction_total": baseline_metrics["prediction_total"],
        "actual_total": model_metrics["actual_total"],
        "mae_improvement": baseline_metrics["mae"] - model_metrics["mae"],
        "rmse_improvement": baseline_metrics["rmse"] - model_metrics["rmse"],
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return pd.DataFrame([row])


def build_model_metrics_by_category(evaluation: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Building model_metrics_by_category")

    metrics = (
        evaluation.groupby(["category_id", "category_name"], as_index=False)
        .agg(
            n_rows=("item_id", "size"),
            actual_total=("actual_item_cnt_month", "sum"),
            model_prediction_total=("model_prediction", "sum"),
            baseline_prediction_total=("baseline_prediction", "sum"),
            model_mae=("model_absolute_error", "mean"),
            baseline_mae=("baseline_absolute_error", "mean"),
            model_mse=("model_squared_error", "mean"),
            baseline_mse=("baseline_squared_error", "mean"),
            model_bias=("model_error", "mean"),
            baseline_bias=("baseline_error", "mean"),
        )
    )

    metrics["model_rmse"] = metrics["model_mse"] ** 0.5
    metrics["baseline_rmse"] = metrics["baseline_mse"] ** 0.5
    metrics["mae_improvement"] = metrics["baseline_mae"] - metrics["model_mae"]
    metrics["rmse_improvement"] = metrics["baseline_rmse"] - metrics["model_rmse"]

    metrics = metrics.drop(columns=["model_mse", "baseline_mse"])

    LOGGER.info("model_metrics_by_category has %s rows", len(metrics))

    return metrics


def build_model_forecast_next_month(
    inference: pd.DataFrame,
    model_predictions: np.ndarray,
) -> pd.DataFrame:
    LOGGER.info("Building model_forecast_next_month")

    forecast = inference.copy()

    forecast["model_prediction"] = model_predictions
    forecast["baseline_prediction"] = pd.to_numeric(
        forecast["lag_1"],
        errors="coerce",
    ).fillna(0).clip(lower=0, upper=20)

    forecast["prediction_month"] = forecast["date_block_num"]
    forecast["generated_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    keep_columns = [
        "prediction_month",
        "date_block_num",
        "year",
        "month",
        "shop_id",
        "item_id",
        "category_id",
        "category_name",
        "item_name",
        "model_prediction",
        "baseline_prediction",
        "lag_1",
        "lag_mean_3",
        "lag_mean_6",
        "had_sales_lag_1",
        "had_sales_lag_3",
        "generated_at_utc",
    ]

    available_columns = [col for col in keep_columns if col in forecast.columns]

    forecast = forecast[available_columns].copy()

    LOGGER.info("model_forecast_next_month has %s rows", len(forecast))

    return forecast


def write_gold_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    gold_prefix: str,
    table_name: str,
    partition_cols: list[str] | None = None,
) -> None:
    path = f"s3://{bucket}/{gold_prefix}/{table_name}/"

    LOGGER.info("Deleting Gold table if exists: %s.%s", database, table_name)

    wr.catalog.delete_table_if_exists(
        database=database,
        table=table_name,
    )

    LOGGER.info("Writing Gold table %s to %s", table_name, path)

    wr.s3.to_parquet(
        df=df,
        path=path,
        dataset=True,
        database=database,
        table=table_name,
        mode="overwrite",
        compression="snappy",
        partition_cols=partition_cols,
        index=False,
        sanitize_columns=True,
    )

    LOGGER.info("Finished writing %s rows to %s", len(df), table_name)


def save_artifacts(
    model: HistGradientBoostingRegressor,
    metrics: pd.DataFrame,
    bucket: str,
    artifacts_prefix: str,
    local_model_dir: str,
) -> None:
    LOGGER.info("Saving model artifacts")

    local_dir = Path(local_model_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = local_dir / "model.joblib"
    metrics_path = local_dir / "model_metrics.json"

    joblib.dump(model, model_path)

    metrics_dict = metrics.iloc[0].to_dict()

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    model_s3_uri = f"s3://{bucket}/{artifacts_prefix}/model.joblib"
    metrics_s3_uri = f"s3://{bucket}/{artifacts_prefix}/model_metrics.json"

    wr.s3.upload(
        local_file=str(model_path),
        path=model_s3_uri,
    )

    wr.s3.upload(
        local_file=str(metrics_path),
        path=metrics_s3_uri,
    )

    LOGGER.info("Saved model to %s", model_s3_uri)
    LOGGER.info("Saved metrics to %s", metrics_s3_uri)


def run(
    bucket: str,
    ml_database: str,
    gold_database: str,
    gold_prefix: str,
    artifacts_prefix: str,
    local_model_dir: str,
    max_train_rows: int,
    seed: int,
) -> None:
    wr.catalog.create_database(
        name=gold_database,
        exist_ok=True,
    )

    train = read_table(ml_database, "train_features")
    validation = read_table(ml_database, "validation_features")
    inference = read_table(ml_database, "inference_features")

    train_sample = maybe_sample_train(
        train=train,
        max_train_rows=max_train_rows,
        seed=seed,
    )

    x_train, y_train = prepare_xy(
        train_sample,
        table_name="train_features",
        require_target=True,
    )

    x_validation, y_validation = prepare_xy(
        validation,
        table_name="validation_features",
        require_target=True,
    )

    x_inference, _ = prepare_xy(
        inference,
        table_name="inference_features",
        require_target=False,
    )

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
    )

    validation_predictions = predict_clipped(
        model=model,
        x=x_validation,
    )

    inference_predictions = predict_clipped(
        model=model,
        x=x_inference,
    )

    model_evaluation = build_model_evaluation(
        validation=validation,
        model_predictions=validation_predictions,
    )

    model_metrics_global = build_model_metrics_global(
        evaluation=model_evaluation,
    )

    model_metrics_by_category = build_model_metrics_by_category(
        evaluation=model_evaluation,
    )

    model_forecast_next_month = build_model_forecast_next_month(
        inference=inference,
        model_predictions=inference_predictions,
    )

    write_gold_table(
        df=model_evaluation,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="model_evaluation",
        partition_cols=["date_block_num"],
    )

    write_gold_table(
        df=model_metrics_global,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="model_metrics_global",
    )

    write_gold_table(
        df=model_metrics_by_category,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="model_metrics_by_category",
    )

    write_gold_table(
        df=model_forecast_next_month,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="model_forecast_next_month",
        partition_cols=["date_block_num"],
    )

    save_artifacts(
        model=model,
        metrics=model_metrics_global,
        bucket=bucket,
        artifacts_prefix=artifacts_prefix,
        local_model_dir=local_model_dir,
    )

    LOGGER.info("Model training and batch prediction completed successfully.")
    LOGGER.info("\n%s", model_metrics_global.to_string(index=False))


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        run(
            bucket=args.bucket,
            ml_database=args.ml_database,
            gold_database=args.gold_database,
            gold_prefix=args.gold_prefix,
            artifacts_prefix=args.artifacts_prefix,
            local_model_dir=args.local_model_dir,
            max_train_rows=args.max_train_rows,
            seed=args.seed,
        )
    except Exception:
        LOGGER.exception("Model training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
