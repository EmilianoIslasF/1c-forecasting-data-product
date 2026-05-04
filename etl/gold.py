# ETL Gold: construye tablas analíticas y métricas para consumo del dashboard.
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import awswrangler as wr


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    # Configura logs para monitorear la ejecución del ETL.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    # Define argumentos para ejecutar el script desde terminal.
    parser = argparse.ArgumentParser(
        description="Gold ETL for forecasting-data-product."
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where gold datasets will be stored.",
    )

    parser.add_argument(
        "--silver-database",
        default="forecasting_silver",
        help="Glue database name for Silver tables.",
    )

    parser.add_argument(
        "--gold-database",
        default="forecasting_gold",
        help="Glue database name for Gold tables.",
    )

    parser.add_argument(
        "--gold-prefix",
        default="forecasting/gold",
        help="S3 prefix where Gold parquet datasets will be written.",
    )

    return parser.parse_args()


def read_table(database: str, table: str) -> pd.DataFrame:
    # Lee una tabla desde Glue/S3 y valida que tenga datos.
    LOGGER.info("Reading table: %s.%s", database, table)

    df = wr.s3.read_parquet_table(
        database=database,
        table=table,
    )

    assert not df.empty, f"Table {database}.{table} is empty"

    if "date_block_num" in df.columns:
        df["date_block_num"] = pd.to_numeric(
            df["date_block_num"],
            errors="coerce",
        ).astype("int64")

    LOGGER.info(
        "Loaded %s.%s with %s rows and %s columns",
        database,
        table,
        len(df),
        len(df.columns),
    )

    LOGGER.info(
        "Columns in %s.%s: %s",
        database,
        table,
        list(df.columns),
    )

    return df


def build_demand_history(sales_monthly_enriched: pd.DataFrame) -> pd.DataFrame:
    # Construye la tabla histórica granular de demanda.
    LOGGER.info("Building demand_history")

    required_columns = [
        "date_block_num",
        "year",
        "month",
        "year_month",
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "category_id",
        "category_name",
        "item_cnt_month",
        "item_cnt_month_clipped",
        "revenue_month",
        "avg_item_price",
        "transactions",
        "first_sale_date",
        "last_sale_date",
    ]

    missing = [
        col for col in required_columns if col not in sales_monthly_enriched.columns
    ]

    assert not missing, f"sales_monthly_enriched missing columns: {missing}"

    demand_history = sales_monthly_enriched[required_columns].copy()

    assert not demand_history.empty, "demand_history is empty"

    LOGGER.info("demand_history has %s rows", len(demand_history))

    return demand_history


def build_category_monthly(demand_history: pd.DataFrame) -> pd.DataFrame:
    # Agrega la demanda mensual a nivel categoría.
    LOGGER.info("Building category_monthly")

    category_monthly = (
        demand_history.groupby(
            [
                "date_block_num",
                "year",
                "month",
                "year_month",
                "category_id",
                "category_name",
            ],
            as_index=False,
        )
        .agg(
            total_item_cnt_month=("item_cnt_month", "sum"),
            total_item_cnt_month_clipped=("item_cnt_month_clipped", "sum"),
            total_revenue_month=("revenue_month", "sum"),
            avg_item_price=("avg_item_price", "mean"),
            active_items=("item_id", "nunique"),
            active_shops=("shop_id", "nunique"),
            active_shop_item_pairs=("item_id", "size"),
            total_transactions=("transactions", "sum"),
        )
    )

    assert not category_monthly.empty, "category_monthly is empty"

    LOGGER.info("category_monthly has %s rows", len(category_monthly))

    return category_monthly


def build_shop_monthly(demand_history: pd.DataFrame) -> pd.DataFrame:
    # Agrega la demanda mensual a nivel tienda.
    LOGGER.info("Building shop_monthly")

    shop_monthly = (
        demand_history.groupby(
            [
                "date_block_num",
                "year",
                "month",
                "year_month",
                "shop_id",
                "shop_name",
            ],
            as_index=False,
        )
        .agg(
            total_item_cnt_month=("item_cnt_month", "sum"),
            total_item_cnt_month_clipped=("item_cnt_month_clipped", "sum"),
            total_revenue_month=("revenue_month", "sum"),
            avg_item_price=("avg_item_price", "mean"),
            active_items=("item_id", "nunique"),
            active_categories=("category_id", "nunique"),
            active_shop_item_pairs=("item_id", "size"),
            total_transactions=("transactions", "sum"),
        )
    )

    assert not shop_monthly.empty, "shop_monthly is empty"

    LOGGER.info("shop_monthly has %s rows", len(shop_monthly))

    return shop_monthly


def build_product_kpis(demand_history: pd.DataFrame) -> pd.DataFrame:
    # Calcula KPIs históricos por producto.
    LOGGER.info("Building product_kpis")

    product_kpis = (
        demand_history.groupby(
            [
                "item_id",
                "item_name",
                "category_id",
                "category_name",
            ],
            as_index=False,
        )
        .agg(
            total_item_cnt_month=("item_cnt_month", "sum"),
            total_item_cnt_month_clipped=("item_cnt_month_clipped", "sum"),
            total_revenue_month=("revenue_month", "sum"),
            avg_item_price=("avg_item_price", "mean"),
            active_months=("date_block_num", "nunique"),
            active_shops=("shop_id", "nunique"),
            total_transactions=("transactions", "sum"),
            first_date_block_num=("date_block_num", "min"),
            last_date_block_num=("date_block_num", "max"),
        )
    )

    product_kpis["avg_sales_per_active_month"] = (
        product_kpis["total_item_cnt_month"] / product_kpis["active_months"]
    )

    assert not product_kpis.empty, "product_kpis is empty"

    LOGGER.info("product_kpis has %s rows", len(product_kpis))

    return product_kpis


def build_baseline_forecast_next_month(
    forecast_input: pd.DataFrame,
    sales_monthly: pd.DataFrame,
) -> pd.DataFrame:
    # Construye un forecast baseline usando demanda reciente.
    LOGGER.info("Building baseline_forecast_next_month")

    required_forecast_cols = [
        "id",
        "date_block_num",
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "category_id",
        "category_name",
    ]

    missing_forecast = [
        col for col in required_forecast_cols if col not in forecast_input.columns
    ]

    assert not missing_forecast, f"forecast_input missing columns: {missing_forecast}"

    max_train_month = int(sales_monthly["date_block_num"].max())

    last_month = sales_monthly[
        sales_monthly["date_block_num"] == max_train_month
    ][
        [
            "shop_id",
            "item_id",
            "item_cnt_month_clipped",
        ]
    ].rename(
        columns={
            "item_cnt_month_clipped": "baseline_last_month",
        }
    )

    last_3_months = sales_monthly[
        sales_monthly["date_block_num"] >= max_train_month - 2
    ]

    last_3_avg = (
        last_3_months.groupby(
            [
                "shop_id",
                "item_id",
            ],
            as_index=False,
        )
        .agg(
            baseline_3_month_avg=("item_cnt_month_clipped", "mean"),
        )
    )

    forecast = forecast_input[required_forecast_cols].copy()

    forecast = forecast.merge(
        last_month,
        on=[
            "shop_id",
            "item_id",
        ],
        how="left",
    ).merge(
        last_3_avg,
        on=[
            "shop_id",
            "item_id",
        ],
        how="left",
    )

    forecast["baseline_last_month"] = forecast["baseline_last_month"].fillna(0)
    forecast["baseline_3_month_avg"] = forecast["baseline_3_month_avg"].fillna(0)

    forecast["baseline_prediction"] = forecast["baseline_last_month"].clip(
        lower=0,
        upper=20,
    )

    forecast["training_last_date_block_num"] = max_train_month
    forecast["prediction_month"] = forecast["date_block_num"]
    forecast["generated_at_utc"] = pd.Timestamp.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    assert not forecast.empty, "baseline_forecast_next_month is empty"

    LOGGER.info("baseline_forecast_next_month has %s rows", len(forecast))

    return forecast


def build_baseline_evaluation(
    sales_monthly: pd.DataFrame,
    item_catalog: pd.DataFrame,
    shop_catalog: pd.DataFrame,
) -> pd.DataFrame:
    # Evalúa el baseline contra el último mes histórico.
    LOGGER.info("Building baseline_evaluation")

    validation_month = int(sales_monthly["date_block_num"].max())
    prediction_source_month = validation_month - 1

    actual = sales_monthly[
        sales_monthly["date_block_num"] == validation_month
    ][
        [
            "shop_id",
            "item_id",
            "item_cnt_month_clipped",
        ]
    ].rename(
        columns={
            "item_cnt_month_clipped": "actual_item_cnt_month",
        }
    )

    prediction = sales_monthly[
        sales_monthly["date_block_num"] == prediction_source_month
    ][
        [
            "shop_id",
            "item_id",
            "item_cnt_month_clipped",
        ]
    ].rename(
        columns={
            "item_cnt_month_clipped": "baseline_prediction",
        }
    )

    evaluation = actual.merge(
        prediction,
        on=[
            "shop_id",
            "item_id",
        ],
        how="left",
    )

    evaluation["baseline_prediction"] = evaluation["baseline_prediction"].fillna(0)

    evaluation = evaluation.merge(
        item_catalog,
        on="item_id",
        how="left",
    ).merge(
        shop_catalog,
        on="shop_id",
        how="left",
    )

    evaluation["validation_date_block_num"] = validation_month
    evaluation["prediction_source_date_block_num"] = prediction_source_month
    evaluation["error"] = (
        evaluation["baseline_prediction"] - evaluation["actual_item_cnt_month"]
    )
    evaluation["absolute_error"] = evaluation["error"].abs()
    evaluation["squared_error"] = evaluation["error"] ** 2

    assert not evaluation.empty, "baseline_evaluation is empty"

    LOGGER.info("baseline_evaluation has %s rows", len(evaluation))

    return evaluation


def build_baseline_metrics_global(evaluation: pd.DataFrame) -> pd.DataFrame:
    # Calcula métricas globales del baseline.
    LOGGER.info("Building baseline_metrics_global")

    metrics = pd.DataFrame(
        [
            {
                "validation_date_block_num": int(
                    evaluation["validation_date_block_num"].max()
                ),
                "prediction_source_date_block_num": int(
                    evaluation["prediction_source_date_block_num"].max()
                ),
                "n_shop_item_pairs": len(evaluation),
                "actual_total": evaluation["actual_item_cnt_month"].sum(),
                "prediction_total": evaluation["baseline_prediction"].sum(),
                "mae": evaluation["absolute_error"].mean(),
                "rmse": evaluation["squared_error"].mean() ** 0.5,
                "bias": evaluation["error"].mean(),
                "generated_at_utc": pd.Timestamp.utcnow().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        ]
    )

    return metrics


def build_baseline_metrics_by_category(evaluation: pd.DataFrame) -> pd.DataFrame:
    # Calcula métricas del baseline por categoría.
    LOGGER.info("Building baseline_metrics_by_category")

    metrics = (
        evaluation.groupby(
            [
                "category_id",
                "category_name",
                "validation_date_block_num",
                "prediction_source_date_block_num",
            ],
            as_index=False,
        )
        .agg(
            n_shop_item_pairs=("item_id", "size"),
            actual_total=("actual_item_cnt_month", "sum"),
            prediction_total=("baseline_prediction", "sum"),
            mae=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
            bias=("error", "mean"),
        )
    )

    metrics["rmse"] = metrics["mean_squared_error"] ** 0.5
    metrics = metrics.drop(columns=["mean_squared_error"])

    assert not metrics.empty, "baseline_metrics_by_category is empty"

    LOGGER.info("baseline_metrics_by_category has %s rows", len(metrics))

    return metrics


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

    LOGGER.info("Deleting Glue table if it already exists: %s.%s", database, table_name)

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

    LOGGER.info("Finished writing Gold table: %s rows=%s", table_name, len(df))


def run_gold_etl(
    bucket: str,
    silver_database: str,
    gold_database: str,
    gold_prefix: str,
) -> None:
    # Orquesta el flujo completo de la capa Gold.
    LOGGER.info("Creating Glue database if needed: %s", gold_database)

    wr.catalog.create_database(
        name=gold_database,
        exist_ok=True,
    )

    sales_monthly = read_table(silver_database, "sales_monthly")
    sales_monthly_enriched = read_table(silver_database, "sales_monthly_enriched")
    item_catalog = read_table(silver_database, "item_catalog")
    shop_catalog = read_table(silver_database, "shop_catalog")
    forecast_input = read_table(silver_database, "forecast_input")

    demand_history = build_demand_history(sales_monthly_enriched)
    category_monthly = build_category_monthly(demand_history)
    shop_monthly = build_shop_monthly(demand_history)
    product_kpis = build_product_kpis(demand_history)

    baseline_forecast_next_month = build_baseline_forecast_next_month(
        forecast_input=forecast_input,
        sales_monthly=sales_monthly,
    )

    baseline_evaluation = build_baseline_evaluation(
        sales_monthly=sales_monthly,
        item_catalog=item_catalog,
        shop_catalog=shop_catalog,
    )

    baseline_metrics_global = build_baseline_metrics_global(
        evaluation=baseline_evaluation,
    )

    baseline_metrics_by_category = build_baseline_metrics_by_category(
        evaluation=baseline_evaluation,
    )

    write_gold_table(
        df=demand_history,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="demand_history",
        partition_cols=["date_block_num"],
    )

    write_gold_table(
        df=category_monthly,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="category_monthly",
        partition_cols=["date_block_num"],
    )

    write_gold_table(
        df=shop_monthly,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="shop_monthly",
        partition_cols=["date_block_num"],
    )

    write_gold_table(
        df=product_kpis,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="product_kpis",
    )

    write_gold_table(
        df=baseline_forecast_next_month,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="baseline_forecast_next_month",
        partition_cols=["date_block_num"],
    )

    write_gold_table(
        df=baseline_evaluation,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="baseline_evaluation",
        partition_cols=["validation_date_block_num"],
    )

    write_gold_table(
        df=baseline_metrics_global,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="baseline_metrics_global",
    )

    write_gold_table(
        df=baseline_metrics_by_category,
        bucket=bucket,
        database=gold_database,
        gold_prefix=gold_prefix,
        table_name="baseline_metrics_by_category",
    )

    LOGGER.info("Gold ETL completed successfully.")


def main() -> None:
    # Punto de entrada del script.
    configure_logging()
    args = parse_args()

    try:
        run_gold_etl(
            bucket=args.bucket,
            silver_database=args.silver_database,
            gold_database=args.gold_database,
            gold_prefix=args.gold_prefix,
        )
    except Exception:
        LOGGER.exception("Gold ETL failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()