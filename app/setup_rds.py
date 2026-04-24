"""
setup_rds.py — Crea las tablas en RDS y carga datos base.
Córrelo UNA VEZ desde SageMaker Studio Jupyter Lab después de crear el stack CFN.

Uso:
    python setup_rds.py \
        --host <RdsEndpoint del Output CFN> \
        --secret-arn <SecretArn del Output CFN>
"""

import argparse
import json
import logging
import sys

import boto3
import pandas as pd
import psycopg2
from psycopg2 import sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def obtener_credenciales(secret_arn: str) -> dict:
    """Lee user/pass desde Secrets Manager."""
    cliente = boto3.client("secretsmanager", region_name="us-east-1")
    respuesta = cliente.get_secret_value(SecretId=secret_arn)
    return json.loads(respuesta["SecretString"])


def conectar(host: str, creds: dict) -> psycopg2.extensions.connection:
    """Crea conexión a PostgreSQL."""
    return psycopg2.connect(
        host=host,
        port=5432,
        dbname=creds.get("dbname", "forecasting"),
        user=creds["username"],
        password=creds["password"],
    )


DDL = """
-- Tabla de tiendas
CREATE TABLE IF NOT EXISTS tiendas (
    shop_id   INTEGER PRIMARY KEY,
    nombre    VARCHAR(200) NOT NULL,
    ciudad    VARCHAR(100)
);

-- Tabla de productos (catálogo)
CREATE TABLE IF NOT EXISTS productos (
    item_id           INTEGER PRIMARY KEY,
    nombre            VARCHAR(500) NOT NULL,
    categoria_id      INTEGER,
    categoria_nombre  VARCHAR(200),
    shop_id           INTEGER REFERENCES tiendas(shop_id)
);

-- Tabla de predicciones (la más importante)
CREATE TABLE IF NOT EXISTS predicciones (
    id               SERIAL PRIMARY KEY,
    item_id          INTEGER NOT NULL REFERENCES productos(item_id),
    shop_id          INTEGER NOT NULL REFERENCES tiendas(shop_id),
    fecha_mes        DATE NOT NULL,
    valor_predicho   FLOAT NOT NULL,
    valor_real       FLOAT,           -- NULL hasta tener ground truth
    rmse             FLOAT,
    modelo_version   VARCHAR(50) DEFAULT 'v1.0',
    creado_en        TIMESTAMP DEFAULT NOW()
);

-- Tabla de feedback del negocio
CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    item_id     INTEGER NOT NULL REFERENCES productos(item_id),
    shop_id     INTEGER NOT NULL REFERENCES tiendas(shop_id),
    usuario     VARCHAR(200) NOT NULL,
    comentario  TEXT NOT NULL,
    creado_en   TIMESTAMP DEFAULT NOW()
);

-- Tabla de productos con problemas identificados
CREATE TABLE IF NOT EXISTS productos_problema (
    id           SERIAL PRIMARY KEY,
    item_id      INTEGER NOT NULL REFERENCES productos(item_id),
    shop_id      INTEGER NOT NULL REFERENCES tiendas(shop_id),
    razon        TEXT,
    estado       VARCHAR(50) DEFAULT 'pendiente',
    reportado_en TIMESTAMP DEFAULT NOW(),
    UNIQUE(item_id, shop_id)
);
"""


def crear_tablas(conn):
    """Ejecuta el DDL para crear todas las tablas."""
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    logger.info("Tablas creadas correctamente.")


def cargar_datos_base(conn, ruta_shops: str, ruta_items: str):
    """
    Carga tiendas y productos desde los CSVs originales de la tarea 01.
    Ajusta las rutas según donde tengas tus CSVs.
    """
    # Tiendas
    logger.info("Cargando tiendas...")
    shops = pd.read_csv(ruta_shops)
    shops = shops.rename(columns={"shop_id": "shop_id", "shop_name": "nombre"})
    shops["ciudad"] = shops["nombre"].str.split(" ").str[0]

    with conn.cursor() as cur:
        for _, row in shops.iterrows():
            cur.execute(
                """
                INSERT INTO tiendas (shop_id, nombre, ciudad)
                VALUES (%s, %s, %s)
                ON CONFLICT (shop_id) DO NOTHING
                """,
                (int(row["shop_id"]), str(row["nombre"]), str(row["ciudad"])),
            )
    conn.commit()
    logger.info(f"Tiendas cargadas: {len(shops):,}")

    # Productos
    logger.info("Cargando productos...")
    items = pd.read_csv(ruta_items)
    # Ajusta el nombre de las columnas según tu CSV (tarea 01)
    # columns esperadas: item_id, item_name, item_category_id

    with conn.cursor() as cur:
        for _, row in items.iterrows():
            cur.execute(
                """
                INSERT INTO productos (item_id, nombre, categoria_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (item_id) DO NOTHING
                """,
                (
                    int(row["item_id"]),
                    str(row["item_name"]),
                    int(row["item_category_id"]),
                ),
            )
    conn.commit()
    logger.info(f"Productos cargados: {len(items):,}")


def main():
    parser = argparse.ArgumentParser(description="Setup de RDS para el parcial")
    parser.add_argument("--host", required=True, help="RdsEndpoint del Output CFN")
    parser.add_argument("--secret-arn", required=True, help="SecretArn del Output CFN")
    parser.add_argument("--shops-csv", default="data/raw/shops.csv")
    parser.add_argument("--items-csv", default="data/raw/items.csv")
    args = parser.parse_args()

    logger.info(f"Conectando a {args.host}...")
    try:
        creds = obtener_credenciales(args.secret_arn)
        conn = conectar(args.host, creds)
        logger.info("Conexión exitosa.")
    except Exception as e:
        logger.exception(f"No se pudo conectar: {e}")
        sys.exit(1)

    crear_tablas(conn)
    cargar_datos_base(conn, args.shops_csv, args.items_csv)

    logger.info("Setup completo. RDS lista para recibir predicciones.")
    conn.close()


if __name__ == "__main__":
    main()
