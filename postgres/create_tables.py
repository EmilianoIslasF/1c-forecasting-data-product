from __future__ import annotations

import argparse
import json
import logging
import sys
from urllib.parse import quote_plus

import boto3
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    func,
    text,
)


LOGGER = logging.getLogger(__name__)
metadata = MetaData()


forecast_jobs = Table(
    "forecast_jobs",
    metadata,
    Column("job_id", String(100), primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("requested_by", String(100), nullable=True),
    Column("job_type", String(50), nullable=False),
    Column("scope_type", String(50), nullable=False),
    Column("scope_value", String(200), nullable=True),
    Column("status", String(50), nullable=False, default="created"),
    Column("s3_output_uri", Text, nullable=True),
    Column("message", Text, nullable=True),
    Column("metadata_json", JSON, nullable=True),
)


business_feedback = Table(
    "business_feedback",
    metadata,
    Column("feedback_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("created_by", String(100), nullable=True),
    Column("shop_id", Integer, nullable=True),
    Column("item_id", Integer, nullable=True),
    Column("category_id", Integer, nullable=True),
    Column("forecast_month", Integer, nullable=True),
    Column("severity", String(30), nullable=False, default="medium"),
    Column("status", String(30), nullable=False, default="open"),
    Column("feedback_text", Text, nullable=False),
)


flagged_products = Table(
    "flagged_products",
    metadata,
    Column("flag_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("created_by", String(100), nullable=True),
    Column("shop_id", Integer, nullable=True),
    Column("item_id", Integer, nullable=False),
    Column("category_id", Integer, nullable=True),
    Column("reason", String(200), nullable=False),
    Column("priority", String(30), nullable=False, default="medium"),
    Column("status", String(30), nullable=False, default="open"),
    Column("notes", Text, nullable=True),
)


app_metrics = Table(
    "app_metrics",
    metadata,
    Column("metric_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("metric_name", String(100), nullable=False),
    Column("metric_value", Numeric, nullable=True),
    Column("metric_unit", String(50), nullable=True),
    Column("metadata_json", JSON, nullable=True),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PostgreSQL operational tables for forecasting-data-product."
    )

    parser.add_argument(
        "--host",
        required=True,
        help="RDS endpoint from CloudFormation Outputs.",
    )

    parser.add_argument(
        "--secret-name",
        default="itam/rds/forecasting/credentials",
        help="Secrets Manager secret name with DB credentials.",
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region.",
    )

    return parser.parse_args()


def get_secret(secret_name: str, region: str) -> dict:
    LOGGER.info("Reading secret from Secrets Manager: %s", secret_name)

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)

    return json.loads(response["SecretString"])


def build_connection_url(host: str, creds: dict) -> str:
    username = quote_plus(creds["username"])
    password = quote_plus(creds["password"])
    dbname = creds["dbname"]
    port = creds.get("port", "5432")

    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        creds = get_secret(args.secret_name, args.region)
        connection_url = build_connection_url(args.host, creds)

        LOGGER.info("Connecting to PostgreSQL host=%s db=%s", args.host, creds["dbname"])

        engine = create_engine(connection_url, pool_pre_ping=True)

        LOGGER.info("Creating tables if they do not exist")
        metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT current_database(), current_user"))
            row = result.fetchone()
            LOGGER.info("Connected successfully: database=%s user=%s", row[0], row[1])

        LOGGER.info("RDS operational schema created successfully.")

    except Exception:
        LOGGER.exception("Failed to create RDS tables.")
        sys.exit(1)


if __name__ == "__main__":
    main()
