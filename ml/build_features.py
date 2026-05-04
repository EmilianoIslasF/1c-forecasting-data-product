# Construye features de ML a partir de tablas Silver y las guarda en la capa ML.
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import awswrangler as wr


LOGGER = logging.getLogger(__name__)

# Configuración base de la variable objetivo y lags.
TARGET_COL = "item_cnt_month"
LAGS = [1, 2, 3, 6, 12]
CLIP_MIN = 0
CLIP_MAX = 20


def configure_logging() -> None:
    # Configura logs para monitorear la ejecución del feature build.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    # Define argumentos para ejecutar el script desde terminal.
    parser = argparse.ArgumentParser(
        description="Build ML features using the original Task 01 feature logic."
    )

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--silver-database", default="forecasting_silver")
    parser.add_argument("--ml-database", default="forecasting_ml")
    parser.add_argument("--ml-prefix", default="forecasting/ml")

    return parser.parse_args()


def read_silver_table(database: str, table: str) -> pd.DataFrame:
    # Lee una tabla Silver desde Glue/S3.
    LOGGER.info("Reading Silver table: %s.%s", database, table)

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


def prepare_monthly(sales_monthly: pd.DataFrame) -> pd.DataFrame:
    # Prepara la tabla mensual con la variable objetivo.
    LOGGER.info("Preparing monthly target")

    required = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_cnt_month_clipped",
    ]

    missing = [col for col in required if col not in sales_monthly.columns]
    assert not missing, f"sales_monthly missing columns: {missing}"

    monthly = sales_monthly[required].copy()

    monthly["date_block_num"] = pd.to_numeric(
        monthly["date_block_num"],
        errors="coerce",
    ).astype("int16")

    monthly["shop_id"] = pd.to_numeric(
        monthly["shop_id"],
        errors="coerce",
    ).astype("int16")

    monthly["item_id"] = pd.to_numeric(
        monthly["item_id"],
        errors="coerce",
    ).astype("int32")

    monthly[TARGET_COL] = pd.to_numeric(
        monthly["item_cnt_month_clipped"],
        errors="coerce",
    ).fillna(0).clip(CLIP_MIN, CLIP_MAX).astype("float32")

    monthly = monthly[
        [
            "date_block_num",
            "shop_id",
            "item_id",
            TARGET_COL,
        ]
    ]

    monthly = (
        monthly.groupby(
            [
                "date_block_num",
                "shop_id",
                "item_id",
            ],
            as_index=False,
        )
        .agg(item_cnt_month=(TARGET_COL, "sum"))
    )

    monthly[TARGET_COL] = monthly[TARGET_COL].clip(CLIP_MIN, CLIP_MAX).astype("float32")

    LOGGER.info("monthly rows=%s", len(monthly))

    return monthly


def prepare_item_metadata(item_catalog: pd.DataFrame) -> pd.DataFrame:
    # Prepara metadata de productos y categorías.
    LOGGER.info("Preparing item metadata")

    required = [
        "item_id",
        "category_id",
    ]

    missing = [col for col in required if col not in item_catalog.columns]
    assert not missing, f"item_catalog missing columns: {missing}"

    cols = [
        col
        for col in [
            "item_id",
            "item_name",
            "category_id",
            "category_name",
        ]
        if col in item_catalog.columns
    ]

    items = item_catalog[cols].copy()

    items["item_id"] = pd.to_numeric(
        items["item_id"],
        errors="coerce",
    ).astype("int32")

    items["item_category_id"] = pd.to_numeric(
        items["category_id"],
        errors="coerce",
    ).fillna(-1).astype("int16")

    if "item_name" not in items.columns:
        items["item_name"] = ""

    if "category_name" not in items.columns:
        items["category_name"] = ""

    items = items[
        [
            "item_id",
            "item_name",
            "item_category_id",
            "category_name",
        ]
    ].drop_duplicates("item_id")

    return items


def build_train_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    # Construye la matriz mensual tienda-producto con ceros.
    LOGGER.info("Building original-style monthly grid with zeros")

    grid: list[pd.DataFrame] = []

    blocks = sorted(monthly["date_block_num"].unique().tolist())

    for block in blocks:
        cur = monthly[monthly["date_block_num"] == block]
        shops_in_month = cur["shop_id"].unique()
        items_in_month = cur["item_id"].unique()

        block_df = pd.DataFrame(
            [(block, shop_id, item_id) for shop_id in shops_in_month for item_id in items_in_month],
            columns=[
                "date_block_num",
                "shop_id",
                "item_id",
            ],
        )

        block_df["date_block_num"] = block_df["date_block_num"].astype("int16")
        block_df["shop_id"] = block_df["shop_id"].astype("int16")
        block_df["item_id"] = block_df["item_id"].astype("int32")

        grid.append(block_df)

    matrix = pd.concat(grid, ignore_index=True)

    matrix = matrix.merge(
        monthly,
        on=[
            "date_block_num",
            "shop_id",
            "item_id",
        ],
        how="left",
    )

    matrix[TARGET_COL] = matrix[TARGET_COL].fillna(0).clip(CLIP_MIN, CLIP_MAX).astype("float32")

    LOGGER.info("train matrix rows=%s", len(matrix))

    return matrix


def add_category_and_seasonality(
    matrix: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    # Agrega categoría, mes y año a la matriz histórica.
    LOGGER.info("Adding item_category_id, month and year")

    df = matrix.merge(
        items[
            [
                "item_id",
                "item_name",
                "item_category_id",
                "category_name",
            ]
        ],
        on="item_id",
        how="left",
    )

    df["item_category_id"] = df["item_category_id"].fillna(-1).astype("int16")
    df["category_id"] = df["item_category_id"].astype("int16")

    df["month"] = (df["date_block_num"] % 12).astype("int8")
    df["year"] = (df["date_block_num"] // 12).astype("int8")

    return df


def build_test_matrix(
    forecast_input: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    # Construye la matriz del mes de inferencia.
    LOGGER.info("Building test matrix for inference month")

    required = [
        "shop_id",
        "item_id",
        "date_block_num",
    ]

    missing = [col for col in required if col not in forecast_input.columns]
    assert not missing, f"forecast_input missing columns: {missing}"

    cols = [
        col
        for col in [
            "id",
            "shop_id",
            "item_id",
            "date_block_num",
            "shop_name",
        ]
        if col in forecast_input.columns
    ]

    test_matrix = forecast_input[cols].copy()

    if "id" not in test_matrix.columns:
        test_matrix["id"] = np.arange(len(test_matrix))

    if "shop_name" not in test_matrix.columns:
        test_matrix["shop_name"] = ""

    test_matrix["date_block_num"] = pd.to_numeric(
        test_matrix["date_block_num"],
        errors="coerce",
    ).astype("int16")

    test_matrix["shop_id"] = pd.to_numeric(
        test_matrix["shop_id"],
        errors="coerce",
    ).astype("int16")

    test_matrix["item_id"] = pd.to_numeric(
        test_matrix["item_id"],
        errors="coerce",
    ).astype("int32")

    test_matrix = test_matrix.merge(
        items[
            [
                "item_id",
                "item_name",
                "item_category_id",
                "category_name",
            ]
        ],
        on="item_id",
        how="left",
    )

    test_matrix["item_category_id"] = test_matrix["item_category_id"].fillna(-1).astype("int16")
    test_matrix["category_id"] = test_matrix["item_category_id"].astype("int16")

    test_matrix["month"] = (test_matrix["date_block_num"] % 12).astype("int8")
    test_matrix["year"] = (test_matrix["date_block_num"] // 12).astype("int8")
    test_matrix[TARGET_COL] = np.float32(0)

    LOGGER.info("test matrix rows=%s", len(test_matrix))

    return test_matrix


def add_target_lags(all_data: pd.DataFrame) -> pd.DataFrame:
    # Agrega lags de la variable objetivo.
    LOGGER.info("Adding original target lags: %s", LAGS)

    df = all_data.sort_values(
        [
            "shop_id",
            "item_id",
            "date_block_num",
        ]
    ).copy()

    group = df.groupby(
        [
            "shop_id",
            "item_id",
        ],
        sort=False,
    )[TARGET_COL]

    for lag in LAGS:
        col = f"{TARGET_COL}_lag_{lag}"
        df[col] = group.shift(lag).fillna(0).astype("float32")

    df["lag_1"] = df["item_cnt_month_lag_1"]
    df["lag_mean_3"] = df[
        [
            "item_cnt_month_lag_1",
            "item_cnt_month_lag_2",
            "item_cnt_month_lag_3",
        ]
    ].mean(axis=1)

    df["lag_mean_6"] = df[
        [
            "item_cnt_month_lag_1",
            "item_cnt_month_lag_2",
            "item_cnt_month_lag_3",
            "item_cnt_month_lag_6",
        ]
    ].mean(axis=1)

    df["had_sales_lag_1"] = (df["item_cnt_month_lag_1"] > 0).astype("int8")

    df["had_sales_lag_3"] = (
        df[
            [
                "item_cnt_month_lag_1",
                "item_cnt_month_lag_2",
                "item_cnt_month_lag_3",
            ]
        ].sum(axis=1)
        > 0
    ).astype("int8")

    LOGGER.info("Lags added")

    return df


def split_features(
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Separa features de entrenamiento, validación e inferencia.
    LOGGER.info("Splitting train, validation and inference")

    common_cols = [
        "id",
        "date_block_num",
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "category_id",
        "category_name",
        "month",
        "year",
        TARGET_COL,
    ]

    for col in common_cols:
        if col not in train_matrix.columns:
            train_matrix[col] = np.nan
        if col not in test_matrix.columns:
            test_matrix[col] = np.nan

    all_data = pd.concat(
        [
            train_matrix[common_cols],
            test_matrix[common_cols],
        ],
        ignore_index=True,
    )

    all_data["id"] = all_data["id"].fillna(-1).astype("int64")
    all_data["date_block_num"] = all_data["date_block_num"].astype("int16")
    all_data["shop_id"] = all_data["shop_id"].astype("int16")
    all_data["item_id"] = all_data["item_id"].astype("int32")
    all_data["item_category_id"] = all_data["item_category_id"].astype("int16")
    all_data["category_id"] = all_data["category_id"].astype("int16")
    all_data["month"] = all_data["month"].astype("int8")
    all_data["year"] = all_data["year"].astype("int8")
    all_data[TARGET_COL] = all_data[TARGET_COL].astype("float32")

    all_data = add_target_lags(all_data)

    historical_max_month = int(train_matrix["date_block_num"].max())
    inference_month = int(test_matrix["date_block_num"].max())

    train_features = all_data[
        all_data["date_block_num"] < historical_max_month
    ].copy()

    validation_features = all_data[
        all_data["date_block_num"] == historical_max_month
    ].copy()

    inference_features = all_data[
        all_data["date_block_num"] == inference_month
    ].copy()

    LOGGER.info("train rows=%s", len(train_features))
    LOGGER.info("validation rows=%s", len(validation_features))
    LOGGER.info("inference rows=%s", len(inference_features))

    assert not train_features.empty, "train_features is empty"
    assert not validation_features.empty, "validation_features is empty"
    assert not inference_features.empty, "inference_features is empty"

    return train_features, validation_features, inference_features


def write_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    prefix: str,
    table_name: str,
    partition_cols: list[str] | None = None,
) -> None:
    # Escribe una tabla de features en S3 y la registra en Glue.
    path = f"s3://{bucket}/{prefix}/{table_name}/"

    LOGGER.info("Deleting Glue table if exists: %s.%s", database, table_name)

    wr.catalog.delete_table_if_exists(
        database=database,
        table=table_name,
    )

    df = df.copy()

    # Evita errores de awswrangler/Athena cuando una columna object está 100% vacía.
    # Estas columnas son metadata para la app/dashboard; el modelo no las usa directamente.
    text_columns = [
        "shop_name",
        "item_name",
        "category_name",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    # Asegura tipos numéricos estables.
    int_columns = [
        "id",
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "category_id",
        "month",
        "year",
        "had_sales_lag_1",
        "had_sales_lag_3",
    ]

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")

    float_columns = [
        "item_cnt_month",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "item_cnt_month_lag_6",
        "item_cnt_month_lag_12",
        "lag_1",
        "lag_mean_3",
        "lag_mean_6",
    ]

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float64")

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

    LOGGER.info("Finished %s rows=%s", table_name, len(df))


def run(
    bucket: str,
    silver_database: str,
    ml_database: str,
    ml_prefix: str,
) -> None:
    # Orquesta la construcción completa de features.
    wr.catalog.create_database(
        name=ml_database,
        exist_ok=True,
    )

    sales_monthly = read_silver_table(silver_database, "sales_monthly")
    item_catalog = read_silver_table(silver_database, "item_catalog")
    forecast_input = read_silver_table(silver_database, "forecast_input")

    monthly = prepare_monthly(sales_monthly)
    items = prepare_item_metadata(item_catalog)

    train_matrix = build_train_matrix(monthly)
    train_matrix = add_category_and_seasonality(
        matrix=train_matrix,
        items=items,
    )

    test_matrix = build_test_matrix(
        forecast_input=forecast_input,
        items=items,
    )

    train_features, validation_features, inference_features = split_features(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
    )

    write_table(
        df=train_features,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="train_features",
        partition_cols=["date_block_num"],
    )

    write_table(
        df=validation_features,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="validation_features",
        partition_cols=["date_block_num"],
    )

    write_table(
        df=inference_features,
        bucket=bucket,
        database=ml_database,
        prefix=ml_prefix,
        table_name="inference_features",
        partition_cols=["date_block_num"],
    )

    LOGGER.info("ML feature build completed with original Task 01 features.")


def main() -> None:
    # Punto de entrada del script.
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