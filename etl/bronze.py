from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import boto3
import pandas as pd
import awswrangler as wr


LOGGER = logging.getLogger(__name__)


EXPECTED_KEYS = {
    "sales_train": [
        "date",
        "date_block_num",
        "shop_id",
        "item_id",
        "item_price",
        "item_cnt_day",
    ],
    "items": [
        "item_id",
        "category_id",
    ],
    "item_categories": [
        "item_category_id",
    ],
    "shops": [
        "shop_id",
    ],
    "test": [
        "id",
        "shop_id",
        "item_id",
    ],
    "sample_submission": [
        "id",
        "item_cnt_month",
    ],
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bronze ETL for forecasting-data-product."
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where raw and bronze datasets will be stored.",
    )

    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Local directory where Kaggle CSV files will be downloaded.",
    )

    parser.add_argument(
        "--kaggle-dataset",
        default="ndarshan2797/english-converted-datasets",
        help="Kaggle dataset slug to download.",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download and use existing local CSV files.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Kaggle download even if CSV files already exist.",
    )

    parser.add_argument(
        "--database",
        default="forecasting_bronze",
        help="Glue database name for Bronze tables.",
    )

    parser.add_argument(
        "--raw-prefix",
        default="forecasting/raw",
        help="S3 prefix where original CSV files will be uploaded.",
    )

    parser.add_argument(
        "--bronze-prefix",
        default="forecasting/bronze",
        help="S3 prefix where Bronze parquet datasets will be written.",
    )

    return parser.parse_args()


def normalize_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def table_name_from_file(path: Path) -> str:
    return normalize_name(path.stem)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(col) for col in df.columns]

    if len(df.columns) != len(set(df.columns)):
        raise ValueError(
            f"Duplicate column names after normalization: {list(df.columns)}"
        )

    return df


def validate_bucket_exists(bucket: str) -> None:
    s3 = boto3.client("s3")
    s3.head_bucket(Bucket=bucket)
    LOGGER.info("S3 bucket exists: s3://%s", bucket)


def csv_files_exist(data_dir: str) -> bool:
    root = Path(data_dir)
    return root.exists() and any(root.rglob("*.csv"))


def validate_kaggle_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    has_env_credentials = bool(
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )

    if kaggle_json.exists() or has_env_credentials:
        LOGGER.info("Kaggle credentials found.")
        return

    raise FileNotFoundError(
        "Kaggle credentials not found. Expected ~/.kaggle/kaggle.json "
        "or environment variables KAGGLE_USERNAME and KAGGLE_KEY."
    )


def download_kaggle_dataset(
    dataset: str,
    data_dir: str,
    force_download: bool,
) -> None:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    if csv_files_exist(data_dir) and not force_download:
        LOGGER.info(
            "CSV files already exist in %s. Skipping Kaggle download. "
            "Use --force-download to download again.",
            data_dir,
        )
        return

    validate_kaggle_credentials()

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        data_dir,
        "--unzip",
    ]

    if force_download:
        command.append("--force")

    LOGGER.info("Downloading Kaggle dataset: %s", dataset)
    LOGGER.info("Command: %s", " ".join(command))

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI was not found. Install it with: uv add kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Kaggle stdout:\n%s", exc.stdout)
        LOGGER.error("Kaggle stderr:\n%s", exc.stderr)
        raise RuntimeError("Kaggle dataset download failed.") from exc

    if result.stdout:
        LOGGER.info("Kaggle stdout:\n%s", result.stdout)

    if result.stderr:
        LOGGER.warning("Kaggle stderr:\n%s", result.stderr)

    LOGGER.info("Kaggle dataset downloaded into: %s", data_dir)


def list_csv_files(data_dir: str) -> list[Path]:
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(root.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found inside: {data_dir}")

    LOGGER.info("Found %s CSV files in %s", len(files), data_dir)

    for file in files:
        LOGGER.info("Found CSV: %s", file)

    return files


def read_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Reading local CSV: %s", path)

    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    LOGGER.info(
        "Loaded %s rows and %s columns from %s",
        len(df),
        len(df.columns),
        path.name,
    )

    return df


def validate_dataframe(df: pd.DataFrame, table_name: str) -> None:
    assert not df.empty, f"{table_name} is empty"
    assert len(df.columns) > 0, f"{table_name} has no columns"

    expected_cols = EXPECTED_KEYS.get(table_name)

    if expected_cols is not None:
        missing = [col for col in expected_cols if col not in df.columns]
        assert not missing, f"{table_name} is missing expected columns: {missing}"

        key_cols = [
            col
            for col in expected_cols
            if col.endswith("_id") or col in ["id", "date", "date_block_num"]
        ]

        for col in key_cols:
            assert df[col].notna().all(), f"{table_name}.{col} contains null values"

    LOGGER.info("Validation passed for table: %s", table_name)


def upload_raw_csv(path: Path, bucket: str, raw_prefix: str) -> str:
    s3_key = f"{raw_prefix}/{path.name}"
    s3_uri = f"s3://{bucket}/{s3_key}"

    LOGGER.info("Uploading raw CSV to %s", s3_uri)

    s3 = boto3.client("s3")
    s3.upload_file(str(path), bucket, s3_key)

    return s3_uri


def write_bronze_table(
    df: pd.DataFrame,
    bucket: str,
    database: str,
    bronze_prefix: str,
    table_name: str,
) -> str:
    path = f"s3://{bucket}/{bronze_prefix}/{table_name}/"

    LOGGER.info("Deleting Glue table if it already exists: %s.%s", database, table_name)

    wr.catalog.delete_table_if_exists(
        database=database,
        table=table_name,
    )

    LOGGER.info("Writing Bronze parquet table to %s", path)

    wr.s3.to_parquet(
        df=df,
        path=path,
        dataset=True,
        database=database,
        table=table_name,
        mode="overwrite",
        compression="snappy",
        index=False,
        sanitize_columns=True,
    )

    return path


def run_bronze_etl(
    bucket: str,
    data_dir: str,
    kaggle_dataset: str,
    skip_download: bool,
    force_download: bool,
    database: str,
    raw_prefix: str,
    bronze_prefix: str,
) -> None:
    validate_bucket_exists(bucket)

    if skip_download:
        LOGGER.info("Skipping Kaggle download by user request.")
    else:
        download_kaggle_dataset(
            dataset=kaggle_dataset,
            data_dir=data_dir,
            force_download=force_download,
        )

    LOGGER.info("Creating Glue database if needed: %s", database)

    wr.catalog.create_database(
        name=database,
        exist_ok=True,
    )

    csv_files = list_csv_files(data_dir)

    for csv_file in csv_files:
        table_name = table_name_from_file(csv_file)

        LOGGER.info(
            "Starting Bronze load for file=%s table=%s",
            csv_file.name,
            table_name,
        )

        df = read_csv(csv_file)
        validate_dataframe(df, table_name)

        raw_uri = upload_raw_csv(
            path=csv_file,
            bucket=bucket,
            raw_prefix=raw_prefix,
        )

        bronze_uri = write_bronze_table(
            df=df,
            bucket=bucket,
            database=database,
            bronze_prefix=bronze_prefix,
            table_name=table_name,
        )

        LOGGER.info(
            "Finished table=%s rows=%s raw=%s bronze=%s",
            table_name,
            len(df),
            raw_uri,
            bronze_uri,
        )

    LOGGER.info("Bronze ETL completed successfully.")


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        run_bronze_etl(
            bucket=args.bucket,
            data_dir=args.data_dir,
            kaggle_dataset=args.kaggle_dataset,
            skip_download=args.skip_download,
            force_download=args.force_download,
            database=args.database,
            raw_prefix=args.raw_prefix,
            bronze_prefix=args.bronze_prefix,
        )
    except Exception:
        LOGGER.exception("Bronze ETL failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
