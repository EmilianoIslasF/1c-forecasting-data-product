from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import awswrangler as wr


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silver ETL for forecasting-data-product."
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where silver datasets will be stored.",
    )

    parser.add_argument(
        "--bronze-database",
        default="forecasting_bronze",
        help="Glue database name for Bronze tables.",
    )

    parser.add_argument(
        "--silver-database",
        default="forecasting_silver",
        help="Glue database name for Silver tables.",
    )

    parser.add_argument(
        "--silver-prefix",
        default="forecasting/silver",
        help="S3 prefix where Silver parquet datasets will be written.",
    )

    return parser.parse_args()


def read_bronze_table(database: str, table: str) -> pd.DataFrame:
    LOGGER.info("Reading Bronze table: %s.%s", database, table)

    path = wr.catalog.get_table_location(
        database=database,
        table=table,
    )

    df = wr.s3.read_parquet(path=path)

    assert not df.empty, f"Bronze table {database}.{table} is empty"

    LOGGER.info(
        "Loaded Bronze table %s.%s with %s rows and %s columns",
        database,
        table,
        len(df),
        len(df.columns),
    )

    return df


def clean_sales_train(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Cleaning sales_train")

    required_columns = [
        "date",
        "date_block_num",
        "shop_id",
        "item_id",
        "item_price",
        "item_cnt_day",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    assert not missing, f"sales_train missing columns: {missing}"

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce",
    )

    numeric_columns = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_price",
        "item_cnt_day",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "date",
            "date_block_num",
            "shop_id",
            "item_id",
            "item_price",
            "item_cnt_day",
        ]
    )

    df["date_block_num"] = df["date_block_num"].astype("int64")
    df["shop_id"] = df["shop_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")

    # Quitamos registros con precio negativo o cero.
    df = df[df["item_price"] > 0]

    df["revenue_day"] = df["item_price"] * df["item_cnt_day"]

    assert not df.empty, "Clean sales_train is empty"

    LOGGER.info("Clean sales_train has %s rows", len(df))

    return df


def build_sales_monthly(sales: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Building sales_monthly")

    sales_monthly = (
        sales.groupby(
            ["date_block_num", "shop_id", "item_id"],
            as_index=False,
        )
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            revenue_month=("revenue_day", "sum"),
            avg_item_price=("item_price", "mean"),
            transactions=("item_cnt_day", "size"),
            first_sale_date=("date", "min"),
            last_sale_date=("date", "max"),
        )
    )

    sales_monthly["item_cnt_month_clipped"] = sales_monthly[
        "item_cnt_month"
    ].clip(lower=0, upper=20)

    sales_monthly["year"] = 2013 + sales_monthly["date_block_num"] // 12
    sales_monthly["month"] = (sales_monthly["date_block_num"] % 12) + 1

    sales_monthly["year_month"] = pd.to_datetime(
        {
            "year": sales_monthly["year"],
            "month": sales_monthly["month"],
            "day": 1,
        }
    ).dt.strftime("%Y-%m")

    assert not sales_monthly.empty, "sales_monthly is empty"

    duplicated = sales_monthly.duplicated(
        subset=["date_block_num", "shop_id", "item_id"]
    ).sum()

    assert duplicated == 0, "sales_monthly has duplicated shop-item-month rows"

    LOGGER.info("sales_monthly has %s rows", len(sales_monthly))

    return sales_monthly


def build_item_catalog(
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
) -> pd.DataFrame:
    LOGGER.info("Building item_catalog")

    items = items.copy()
    item_categories = item_categories.copy()

    required_items = ["item_id", "item_name", "category_id"]
    required_categories = ["item_category_id", "item_category_name"]

    missing_items = [col for col in required_items if col not in items.columns]
    missing_categories = [
        col for col in required_categories if col not in item_categories.columns
    ]

    assert not missing_items, f"items missing columns: {missing_items}"
    assert not missing_categories, (
        f"item_categories missing columns: {missing_categories}"
    )

    items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce")
    items["category_id"] = pd.to_numeric(items["category_id"], errors="coerce")

    item_categories["item_category_id"] = pd.to_numeric(
        item_categories["item_category_id"],
        errors="coerce",
    )

    items = items.dropna(subset=["item_id", "category_id"])
    item_categories = item_categories.dropna(subset=["item_category_id"])

    items["item_id"] = items["item_id"].astype("int64")
    items["category_id"] = items["category_id"].astype("int64")
    item_categories["item_category_id"] = item_categories[
        "item_category_id"
    ].astype("int64")

    item_categories = item_categories.rename(
        columns={
            "item_category_id": "category_id",
            "item_category_name": "category_name",
        }
    )

    item_catalog = items.merge(
        item_categories,
        on="category_id",
        how="left",
    )

    assert not item_catalog.empty, "item_catalog is empty"
    assert item_catalog["item_id"].notna().all(), "item_catalog has null item_id"

    LOGGER.info("item_catalog has %s rows", len(item_catalog))

    return item_catalog


def build_shop_catalog(shops: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Building shop_catalog")

    required_columns = ["shop_id", "shop_name"]
    missing = [col for col in required_columns if col not in shops.columns]

    assert not missing, f"shops missing columns: {missing}"

    shop_catalog = shops.copy()

    shop_catalog["shop_id"] = pd.to_numeric(
        shop_catalog["shop_id"],
        errors="coerce",
    )

    shop_catalog = shop_catalog.dropna(subset=["shop_id"])
    shop_catalog["shop_id"] = shop_catalog["shop_id"].astype("int64")

    assert not shop_catalog.empty, "shop_catalog is empty"
    assert shop_catalog["shop_id"].notna().all(), "shop_catalog has null shop_id"

    LOGGER.info("shop_catalog has %s rows", len(shop_catalog))

    return shop_catalog


def build_sales_monthly_enriched(
    sales_monthly: pd.DataFrame,
    item_catalog: pd.DataFrame,
    shop_catalog: pd.DataFrame,
) -> pd.DataFrame:
    LOGGER.info("Building sales_monthly_enriched")

    enriched = sales_monthly.merge(
        item_catalog,
        on="item_id",
        how="left",
    ).merge(
        shop_catalog,
        on="shop_id",
        how="left",
    )

    assert not enriched.empty, "sales_monthly_enriched is empty"

    LOGGER.info("sales_monthly_enriched has %s rows", len(enriched))

    return enriched


def build_forecast_input(
    test: pd.DataFrame,
    sales_monthly: pd.DataFrame,
    item_catalog: pd.DataFrame,
    shop_catalog: pd.DataFrame,
) -> pd.DataFrame:
    LOGGER.info("Building forecast_input")

    required_columns = ["id", "shop_id", "item_id"]
    missing = [col for col in required_columns if col not in test.columns]

    assert not missing, f"test missing columns: {missing}"

    forecast_input = test.copy()

    forecast_input["id"] = pd.to_numeric(
        forecast_input["id"],
        errors="coerce",
    )

    forecast_input["shop_id"] = pd.to_numeric(
        forecast_input["shop_id"],
        errors="coerce",
    )

    forecast_input["item_id"] = pd.to_numeric(
        forecast_input["item_id"],
        errors="coerce",
    )

    forecast_input = forecast_input.dropna(
        subset=[
            "id",
            "shop_id",
            "item_id",
        ]
    )

    forecast_input["id"] = forecast_input["id"].astype("int64")
    forecast_input["shop_id"] = forecast_input["shop_id"].astype("int64")
    forecast_input["item_id"] = forecast_input["item_id"].astype("int64")

    next_date_block_num = int(sales_monthly["date_block_num"].max()) + 1

    forecast_input["date_block_num"] = next_date_block_num

    forecast_input = forecast_input.merge(
        item_catalog,
        on="item_id",
        how="left",
    ).merge(
        shop_catalog,
        on="shop_id",
        how="left",
    )

    assert not forecast_input.empty, "forecast_input is empty"

    LOGGER.info("forecast_input has %s rows", len(forecast_input))

    return forecast_input


def write_silver_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    silver_prefix: str,
    table_name: str,
    partition_cols: list[str] | None = None,
) -> None:
    path = f"s3://{bucket}/{silver_prefix}/{table_name}/"

    LOGGER.info("Deleting Glue table if it already exists: %s.%s", database, table_name)

    wr.catalog.delete_table_if_exists(
        database=database,
        table=table_name,
    )

    LOGGER.info("Writing Silver table %s to %s", table_name, path)

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

    LOGGER.info("Finished writing Silver table: %s rows=%s", table_name, len(df))


def run_silver_etl(
    bucket: str,
    bronze_database: str,
    silver_database: str,
    silver_prefix: str,
) -> None:
    LOGGER.info("Creating Glue database if needed: %s", silver_database)

    wr.catalog.create_database(
        name=silver_database,
        exist_ok=True,
    )

    sales_train = read_bronze_table(bronze_database, "sales_train")
    items = read_bronze_table(bronze_database, "items")
    item_categories = read_bronze_table(bronze_database, "item_categories")
    shops = read_bronze_table(bronze_database, "shops")
    test = read_bronze_table(bronze_database, "test")

    sales_clean = clean_sales_train(sales_train)

    sales_monthly = build_sales_monthly(sales_clean)
    item_catalog = build_item_catalog(items, item_categories)
    shop_catalog = build_shop_catalog(shops)

    sales_monthly_enriched = build_sales_monthly_enriched(
        sales_monthly=sales_monthly,
        item_catalog=item_catalog,
        shop_catalog=shop_catalog,
    )

    forecast_input = build_forecast_input(
        test=test,
        sales_monthly=sales_monthly,
        item_catalog=item_catalog,
        shop_catalog=shop_catalog,
    )

    write_silver_table(
        df=item_catalog,
        bucket=bucket,
        database=silver_database,
        silver_prefix=silver_prefix,
        table_name="item_catalog",
    )

    write_silver_table(
        df=shop_catalog,
        bucket=bucket,
        database=silver_database,
        silver_prefix=silver_prefix,
        table_name="shop_catalog",
    )

    write_silver_table(
        df=sales_monthly,
        bucket=bucket,
        database=silver_database,
        silver_prefix=silver_prefix,
        table_name="sales_monthly",
        partition_cols=["date_block_num"],
    )

    write_silver_table(
        df=sales_monthly_enriched,
        bucket=bucket,
        database=silver_database,
        silver_prefix=silver_prefix,
        table_name="sales_monthly_enriched",
        partition_cols=["date_block_num"],
    )

    write_silver_table(
        df=forecast_input,
        bucket=bucket,
        database=silver_database,
        silver_prefix=silver_prefix,
        table_name="forecast_input",
        partition_cols=["date_block_num"],
    )

    LOGGER.info("Silver ETL completed successfully.")


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        run_silver_etl(
            bucket=args.bucket,
            bronze_database=args.bronze_database,
            silver_database=args.silver_database,
            silver_prefix=args.silver_prefix,
        )
    except Exception:
        LOGGER.exception("Silver ETL failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
