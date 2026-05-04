# Entrena el modelo final, evalúa desempeño y escribe resultados en Gold.
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import awswrangler as wr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LOGGER = logging.getLogger(__name__)

# Configuración base del target, clipping y reproducibilidad.
TARGET_COL = "item_cnt_month"
CLIP_MIN = 0
CLIP_MAX = 20
SEED = 42

# Features usadas por el modelo.
FEATURE_COLUMNS = [
    "date_block_num",
    "shop_id",
    "item_id",
    "item_category_id",
    "month",
    "year",
    "item_cnt_month_lag_1",
    "item_cnt_month_lag_2",
    "item_cnt_month_lag_3",
    "item_cnt_month_lag_6",
    "item_cnt_month_lag_12",
]


def configure_logging() -> None:
    # Configura logs para monitorear el entrenamiento.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    # Define argumentos para ejecutar el script desde terminal.
    parser = argparse.ArgumentParser(
        description="Train original GradientBoostingRegressor model and write outputs to Gold."
    )

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--ml-database", default="forecasting_ml")
    parser.add_argument("--gold-database", default="forecasting_gold")
    parser.add_argument("--gold-prefix", default="forecasting/gold")
    parser.add_argument("--artifacts-prefix", default="forecasting/artifacts/models")
    parser.add_argument("--local-model-dir", default="artifacts/models")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)

    return parser.parse_args()


def read_table(database: str, table: str) -> pd.DataFrame:
    # Lee una tabla desde Glue/S3.
    LOGGER.info("Reading table: %s.%s", database, table)

    df = wr.s3.read_parquet_table(
        database=database,
        table=table,
    )

    assert not df.empty, f"{database}.{table} is empty"

    LOGGER.info(
        "Loaded %s.%s rows=%s cols=%s",
        database,
        table,
        len(df),
        len(df.columns),
    )

    return df


def prepare_xy(
    df: pd.DataFrame,
    table_name: str,
    require_target: bool,
) -> tuple[pd.DataFrame, pd.Series | None]:
    # Separa features y target para entrenamiento o inferencia.
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    assert not missing, f"{table_name} missing features: {missing}"

    data = df.copy()

    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    x = data[FEATURE_COLUMNS].copy()

    y = None
    if require_target:
        assert TARGET_COL in data.columns, f"{table_name} missing target: {TARGET_COL}"
        y = pd.to_numeric(data[TARGET_COL], errors="coerce").fillna(0)
        y = y.clip(CLIP_MIN, CLIP_MAX)

    return x, y


def train_ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
) -> Ridge:
    # Entrena un baseline Ridge.
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def train_gbr(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
) -> GradientBoostingRegressor:
    # Entrena el modelo principal Gradient Boosting.
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
    )

    model.fit(x_train, y_train)

    return model


def predict_clipped(model: Any, x: pd.DataFrame) -> np.ndarray:
    # Genera predicciones restringidas al rango del target.
    preds = model.predict(x)
    preds = np.clip(preds, CLIP_MIN, CLIP_MAX)
    return preds


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    # Calcula métricas principales de evaluación.
    y_pred = np.clip(y_pred, CLIP_MIN, CLIP_MAX)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = None

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "prediction_total": float(np.sum(y_pred)),
        "actual_total": float(np.sum(y_true)),
    }


def build_model_evaluation(
    validation: pd.DataFrame,
    model_predictions: np.ndarray,
) -> pd.DataFrame:
    # Construye tabla de evaluación a nivel tienda-producto.
    LOGGER.info("Building model_evaluation")

    evaluation = validation.copy()

    evaluation["actual_item_cnt_month"] = pd.to_numeric(
        evaluation[TARGET_COL],
        errors="coerce",
    ).fillna(0).clip(CLIP_MIN, CLIP_MAX)

    evaluation["model_prediction"] = model_predictions

    evaluation["baseline_prediction"] = pd.to_numeric(
        evaluation["item_cnt_month_lag_1"],
        errors="coerce",
    ).fillna(0).clip(CLIP_MIN, CLIP_MAX)

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
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "category_id",
        "category_name",
        "actual_item_cnt_month",
        "model_prediction",
        "baseline_prediction",
        "model_error",
        "baseline_error",
        "model_absolute_error",
        "baseline_absolute_error",
        "model_squared_error",
        "baseline_squared_error",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "item_cnt_month_lag_6",
        "item_cnt_month_lag_12",
    ]

    available_columns = [col for col in keep_columns if col in evaluation.columns]

    return evaluation[available_columns].copy()


def build_model_metrics_global(evaluation: pd.DataFrame) -> pd.DataFrame:
    # Calcula métricas globales del modelo y baseline.
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
        "model_mape": model_metrics["mape"],
        "model_prediction_total": model_metrics["prediction_total"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_r2": baseline_metrics["r2"],
        "baseline_mape": baseline_metrics["mape"],
        "baseline_prediction_total": baseline_metrics["prediction_total"],
        "actual_total": model_metrics["actual_total"],
        "mae_improvement": baseline_metrics["mae"] - model_metrics["mae"],
        "rmse_improvement": baseline_metrics["rmse"] - model_metrics["rmse"],
        "model_name": "GradientBoostingRegressor",
        "feature_set": "task_01_original_features",
        "features": ",".join(FEATURE_COLUMNS),
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return pd.DataFrame([row])


def build_model_metrics_by_category(evaluation: pd.DataFrame) -> pd.DataFrame:
    # Calcula métricas agregadas por categoría.
    LOGGER.info("Building model_metrics_by_category")

    metrics = (
        evaluation.groupby(
            [
                "category_id",
                "category_name",
            ],
            as_index=False,
        )
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

    return metrics


def build_model_forecast_next_month(
    inference: pd.DataFrame,
    model_predictions: np.ndarray,
) -> pd.DataFrame:
    # Construye tabla final de pronóstico para el siguiente mes.
    LOGGER.info("Building model_forecast_next_month")

    forecast = inference.copy()

    forecast["model_prediction"] = model_predictions

    forecast["baseline_prediction"] = pd.to_numeric(
        forecast["item_cnt_month_lag_1"],
        errors="coerce",
    ).fillna(0).clip(CLIP_MIN, CLIP_MAX)

    forecast["prediction_month"] = forecast["date_block_num"]
    forecast["lag_1"] = forecast["item_cnt_month_lag_1"]

    forecast["lag_mean_3"] = forecast[
        [
            "item_cnt_month_lag_1",
            "item_cnt_month_lag_2",
            "item_cnt_month_lag_3",
        ]
    ].mean(axis=1)

    forecast["lag_mean_6"] = forecast[
        [
            "item_cnt_month_lag_1",
            "item_cnt_month_lag_2",
            "item_cnt_month_lag_3",
            "item_cnt_month_lag_6",
        ]
    ].mean(axis=1)

    forecast["had_sales_lag_1"] = (forecast["item_cnt_month_lag_1"] > 0).astype("int8")

    forecast["had_sales_lag_3"] = (
        forecast[
            [
                "item_cnt_month_lag_1",
                "item_cnt_month_lag_2",
                "item_cnt_month_lag_3",
            ]
        ].sum(axis=1)
        > 0
    ).astype("int8")

    forecast["generated_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    keep_columns = [
        "id",
        "prediction_month",
        "date_block_num",
        "year",
        "month",
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "category_id",
        "category_name",
        "model_prediction",
        "baseline_prediction",
        "lag_1",
        "lag_mean_3",
        "lag_mean_6",
        "had_sales_lag_1",
        "had_sales_lag_3",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "item_cnt_month_lag_6",
        "item_cnt_month_lag_12",
        "generated_at_utc",
    ]

    available_columns = [col for col in keep_columns if col in forecast.columns]

    return forecast[available_columns].copy()


def write_gold_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    gold_prefix: str,
    table_name: str,
    partition_cols: list[str] | None = None,
) -> None:
    # Escribe una tabla Gold en S3 y la registra en Glue.
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
    model: GradientBoostingRegressor,
    metrics: pd.DataFrame,
    ridge_rmse_valid: float,
    bucket: str,
    artifacts_prefix: str,
    local_model_dir: str,
    model_params: dict[str, Any],
) -> None:
    # Guarda modelo entrenado y métricas como artefactos locales y en S3.
    LOGGER.info("Saving model artifacts")

    local_dir = Path(local_model_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = local_dir / "model.joblib"
    metrics_path = local_dir / "model_metrics.json"

    joblib.dump(model, model_path)

    metrics_dict = metrics.iloc[0].to_dict()
    metrics_dict["baseline_ridge_rmse_valid"] = ridge_rmse_valid
    metrics_dict["model_params"] = model_params
    metrics_dict["target_clip"] = [CLIP_MIN, CLIP_MAX]
    metrics_dict["features_list"] = FEATURE_COLUMNS

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

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
    alpha: float,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
) -> None:
    # Orquesta entrenamiento, evaluación, forecast y guardado de artefactos.
    wr.catalog.create_database(
        name=gold_database,
        exist_ok=True,
    )

    train = read_table(ml_database, "train_features")
    validation = read_table(ml_database, "validation_features")
    inference = read_table(ml_database, "inference_features")

    x_train, y_train = prepare_xy(
        train,
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

    LOGGER.info("Training Ridge baseline")
    ridge = train_ridge(
        x_train=x_train,
        y_train=y_train,
        alpha=alpha,
    )

    ridge_pred = predict_clipped(
        model=ridge,
        x=x_validation,
    )

    ridge_rmse_valid = float(np.sqrt(mean_squared_error(y_validation, ridge_pred)))
    LOGGER.info("Ridge RMSE valid: %.4f", ridge_rmse_valid)

    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "random_state": seed,
    }

    LOGGER.info("Training GradientBoostingRegressor with original params")
    gbr = train_gbr(
        x_train=x_train,
        y_train=y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
    )

    validation_predictions = predict_clipped(
        model=gbr,
        x=x_validation,
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

    LOGGER.info("Re-training GradientBoostingRegressor with train + validation")
    x_all = pd.concat([x_train, x_validation], ignore_index=True)
    y_all = pd.concat([y_train, y_validation], ignore_index=True)

    gbr.fit(x_all, y_all)

    inference_predictions = predict_clipped(
        model=gbr,
        x=x_inference,
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
        model=gbr,
        metrics=model_metrics_global,
        ridge_rmse_valid=ridge_rmse_valid,
        bucket=bucket,
        artifacts_prefix=artifacts_prefix,
        local_model_dir=local_model_dir,
        model_params=model_params,
    )

    LOGGER.info("Model training completed successfully.")
    LOGGER.info("\n%s", model_metrics_global.to_string(index=False))


def main() -> None:
    # Punto de entrada del script.
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
            alpha=args.alpha,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            seed=args.seed,
        )
    except Exception:
        LOGGER.exception("Model training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()