from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import awswrangler as wr


LOGGER = logging.getLogger(__name__)


LAG_MONTHS = [1, 2, 3, 6, 12]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ML features for forecasting-data-product."
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where ML feature tables will be stored.",
    )

    parser.add_argument(
        "--silver-database",
        default="forecasting_silver",
        help="Glue database with Silver tables.",
    )

    parser.add_argument(
        "--ml-database",
        default="forecasting_ml",
        help="Glue database for ML feature tables.",
    )

    parser.add_argument(
        "--ml-prefix",
        default="forecasting/ml",
        help="S3 prefix for ML feature tables.",
    )

    return parser.parse_args()


def read_silver_table(database: str, table: str) -> pd.DataFrame:
    LOGGER.info("Reading Silver table: %s.%s", database, table)

    df = wr.s3.read_parquet_table(
        database=database,
        table=table,
    )

    assert not df.empty, f"Silver table {database}.{table} is empty"

    LOGGER.info(
        "Loaded %s.%s with %s rows and %s columns",
        database,
        table,
        len(df),
        len(df.columns),
    )

    return df


def prepare_sales_monthly(sales_monthly: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Preparing sales_monthly")

    required_columns = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_cnt_month_clipped",
    ]

    missing = [col for col in required_columns if col not in sales_monthly.columns]
    assert not missing, f"sales_monthly missing columns: {missing}"

    sales = sales_monthly[required_columns].copy()

    for col in ["date_block_num", "shop_id", "item_id"]:
        sales[col] = pd.to_numeric(sales[col], errors="coerce").astype("int64")

    sales["item_cnt_month_clipped"] = pd.to_numeric(
        sales["item_cnt_month_clipped"],
        errors="coerce",
    ).fillna(0)

    sales = (
        sales.groupby(
            ["date_block_num", "shop_id", "item_id"],
            as_index=False,
        )
        .agg(item_cnt_month=("item_cnt_month_clipped", "sum"))
    )

    sales["item_cnt_month"] = sales["item_cnt_month"].clip(lower=0, upper=20)

    LOGGER.info("Prepared sales_monthly with %s rows", len(sales))

    return sales


def prepare_catalogs(
    item_catalog: pd.DataFrame,
    forecast_input: pd.DataFrame,
) -> pd.DataFrame:
    LOGGER.info("Preparing shop-item catalog for ML matrix")

    item_cols = [
        "item_id",
        "category_id",
        "category_name",
        "item_name",
    ]

    item_missing = [col for col in item_cols if col not in item_catalog.columns]
    assert not item_missing, f"item_catalog missing columns: {item_missing}"

    input_required = ["shop_id", "item_id"]
    input_missing = [col for col in input_required if col not in forecast_input.columns]
    assert not input_missing, f"forecast_input missing columns: {input_missing}"

    pairs = forecast_input[["shop_id", "item_id"]].drop_duplicates().copy()

    pairs["shop_id"] = pd.to_numeric(pairs["shop_id"], errors="coerce").astype("int64")
    pairs["item_id"] = pd.to_numeric(pairs["item_id"], errors="coerce").astype("int64")

    catalog = pairs.merge(
        item_catalog[item_cols],
        on="item_id",
        how="left",
    )

    catalog["category_id"] = pd.to_numeric(
        catalog["category_id"],
        errors="coerce",
    ).fillna(-1).astype("int64")

    assert not catalog.empty, "ML catalog is empty"

    LOGGER.info("Prepared ML catalog with %s shop-item pairs", len(catalog))

    return catalog


def build_month_grid(
    catalog: pd.DataFrame,
    sales: pd.DataFrame,
    forecast_input: pd.DataFrame,
) -> pd.DataFrame:
    LOGGER.info("Building shop-item-month grid")

    max_train_month = int(sales["date_block_num"].max())
    inference_month = int(forecast_input["date_block_num"].max())

    LOGGER.info("Max train month: %s", max_train_month)
    LOGGER.info("Inference month: %s", inference_month)

    months = pd.DataFrame(
        {
            "date_block_num": list(range(0, inference_month + 1)),
        }
    )

    catalog = catalog.copy()
    catalog["_key"] = 1
    months["_key"] = 1

    grid = catalog.merge(months, on="_key", how="inner").drop(columns=["_key"])

    LOGGER.info("Grid has %s rows before merging sales", len(grid))

    full = grid.merge(
        sales,
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    full["item_cnt_month"] = full["item_cnt_month"].fillna(0).clip(lower=0, upper=20)

    full["year"] = 2013 + full["date_block_num"] // 12
    full["month"] = (full["date_block_num"] % 12) + 1

    full = full.sort_values(
        ["shop_id", "item_id", "date_block_num"],
    ).reset_index(drop=True)

    LOGGER.info("Full ML matrix has %s rows", len(full))

    return full


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Adding lag features")

    df = df.copy()

    group = df.groupby(["shop_id", "item_id"])["item_cnt_month"]

    for lag in LAG_MONTHS:
        df[f"lag_{lag}"] = group.shift(lag)

    df["lag_mean_3"] = df[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    df["lag_mean_6"] = df[["lag_1", "lag_2", "lag_3", "lag_6"]].mean(axis=1)

    df["had_sales_lag_1"] = (df["lag_1"].fillna(0) > 0).astype("int64")
    df["had_sales_lag_3"] = (
        df[["lag_1", "lag_2", "lag_3"]].fillna(0).sum(axis=1) > 0
    ).astype("int64")

    lag_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "lag_mean_3",
        "lag_mean_6",
    ]

    for col in lag_columns:
        df[col] = df[col].fillna(0)

    LOGGER.info("Lag features added")

    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Splitting train, validation and inference features")

    max_month = int(df["date_block_num"].max())
    validation_month = max_month - 1
    inference_month = max_month

    train = df[
        (df["date_block_num"] >= 12)
        & (df["date_block_num"] < validation_month)
    ].copy()

    validation = df[df["date_block_num"] == validation_month].copy()
    inference = df[df["date_block_num"] == inference_month].copy()

    LOGGER.info("Train rows: %s", len(train))
    LOGGER.info("Validation rows: %s", len(validation))
    LOGGER.info("Inference rows: %s", len(inference))

    assert not train.empty, "train_features is empty"
    assert not validation.empty, "validation_features is empty"
    assert not inference.empty, "inference_features is empty"

    return train, validation, inference


def write_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    prefix: str,
    table_name: str,
    partition_cols: list[str] | None = None,
) -> None:
    path = f"s3://{bucket}/{prefix}/{table_name}/"

    LOGGER.info("Deleting table if exists: %s.%s", database, table_name)

    wr.catalog.delete_table_if_exists(
        database=database,
        table=table_name,
    )

    LOGGER.info("Writing table %s to %s", table_name, path)

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


def run(
    bucket: str,
    silver_database: str,
    ml_database: str,
    ml_prefix: str,
) -> None:
    LOGGER.info("Creating ML database if needed: %s", ml_database)

    wr.catalog.create_database(
        name=ml_database,
        exist_ok=True,
    )

    sales_monthly = read_silver_table(silver_database, "sales_monthly")
    item_catalog = read_silver_table(silver_database, "item_catalog")
    forecast_input = read_silver_table(silver_database, "forecast_input")

    sales = prepare_sales_monthly(sales_monthly)
    catalog = prepare_catalogs(item_catalog, forecast_input)

    full = build_month_grid(
        catalog=catalog,
        sales=sales,
        forecast_input=forecast_input,
    )

    features = add_lag_features(full)

    train, validation, inference = split_features(features)

    write_table(
        df=train,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="train_features",
        partition_cols=["date_block_num"],
    )

    write_table(
        df=validation,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="validation_features",
        partition_cols=["date_block_num"],
    )

    write_table(
        df=inference,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="inference_features",
        partition_cols=["date_block_num"],
    )

    LOGGER.info("ML feature build completed successfully.")


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        run(
            bucket=args.bucket,
            silver_database=args.silver_database,
            ml_database=args.ml_database,
            ml_prefix=args.ml_prefix,
        )
    except Exception:
        LOGGER.exception("ML feature build failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
