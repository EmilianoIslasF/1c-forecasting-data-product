"""
cargar_predicciones.py — Corre tu modelo y carga predicciones en RDS.
Este es el script de "pre-cómputo" — se corre UNA VEZ (o cuando hay datos nuevos).
La app Streamlit SOLO LEE, nunca corre el modelo en tiempo real.

Uso desde SageMaker Studio:
    python cargar_predicciones.py \
        --host <RdsEndpoint> \
        --secret-arn <SecretArn> \
        --modelo-path artifacts/model.joblib \
        --datos-path data/prep/test.csv
"""

import argparse
import json
import logging
import sys
from datetime import date

import boto3
import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

FECHA_PRONOSTICO = date(2026, 1, 1)   # El mes que pronosticas — ajusta según tus datos


def obtener_credenciales(secret_arn: str) -> dict:
    cliente = boto3.client("secretsmanager", region_name="us-east-1")
    respuesta = cliente.get_secret_value(SecretId=secret_arn)
    return json.loads(respuesta["SecretString"])


def conectar(host: str, creds: dict):
    return psycopg2.connect(
        host=host, port=5432,
        dbname=creds.get("dbname", "forecasting"),
        user=creds["username"], password=creds["password"],
    )


def cargar_modelo(ruta: str):
    logger.info(f"Cargando modelo desde {ruta}...")
    try:
        modelo = joblib.load(ruta)
        logger.info("Modelo cargado OK.")
        return modelo
    except FileNotFoundError:
        logger.error(f"Modelo no encontrado en {ruta}")
        sys.exit(1)


def generar_predicciones(modelo, datos_path: str) -> pd.DataFrame:
    """
    Lee el CSV de inferencia, genera predicciones y calcula RMSE.
    Ajusta las columnas según tu feature engineering de tarea 01.
    """
    logger.info(f"Leyendo datos desde {datos_path}...")
    df = pd.read_csv(datos_path)
    logger.info(f"Datos leídos: {len(df):,} registros")

    # Ajusta estas columnas a las que tiene tu CSV de inferencia
    columnas_features = [c for c in df.columns if c not in ["item_id", "shop_id", "item_cnt_month"]]

    logger.info("Generando predicciones...")
    df["valor_predicho"] = modelo.predict(df[columnas_features])
    df["valor_predicho"] = df["valor_predicho"].clip(lower=0)  # No hay ventas negativas

    # Si tienes ground truth, calculas RMSE; si no, lo dejas en None
    if "item_cnt_month" in df.columns:
        df["valor_real"] = df["item_cnt_month"]
        rmse_global = np.sqrt(mean_squared_error(df["valor_real"], df["valor_predicho"]))
        df["rmse"] = np.sqrt((df["valor_predicho"] - df["valor_real"]) ** 2)
        logger.info(f"RMSE global: {rmse_global:.4f}")
    else:
        df["valor_real"] = None
        df["rmse"] = None

    logger.info(f"Predicciones generadas: {len(df):,}")
    return df[["item_id", "shop_id", "valor_predicho", "valor_real", "rmse"]]


def insertar_predicciones(conn, df: pd.DataFrame, fecha_mes: date):
    """Inserta las predicciones en la tabla RDS."""
    logger.info(f"Insertando {len(df):,} predicciones para el mes {fecha_mes}...")

    registros = [
        (
            int(row["item_id"]),
            int(row["shop_id"]),
            fecha_mes,
            float(row["valor_predicho"]),
            float(row["valor_real"]) if row["valor_real"] is not None else None,
            float(row["rmse"]) if row["rmse"] is not None else None,
            "v1.0",
        )
        for _, row in df.iterrows()
    ]

    with conn.cursor() as cur:
        # Borramos predicciones anteriores del mismo mes para idempotencia
        cur.execute("DELETE FROM predicciones WHERE fecha_mes = %s", (fecha_mes,))
        logger.info(f"Predicciones anteriores del mes {fecha_mes} eliminadas.")

        cur.executemany(
            """
            INSERT INTO predicciones
                (item_id, shop_id, fecha_mes, valor_predicho, valor_real, rmse, modelo_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            registros,
        )
    conn.commit()
    logger.info("Predicciones insertadas correctamente.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--secret-arn", required=True)
    parser.add_argument("--modelo-path", default="artifacts/model.joblib")
    parser.add_argument("--datos-path", default="data/prep/test.csv")
    parser.add_argument("--fecha-mes", default=str(FECHA_PRONOSTICO),
                        help="Fecha del mes a pronosticar (YYYY-MM-DD)")
    args = parser.parse_args()

    fecha = date.fromisoformat(args.fecha_mes)

    logger.info(f"Conectando a RDS en {args.host}...")
    creds = obtener_credenciales(args.secret_arn)
    conn = conectar(args.host, creds)

    modelo = cargar_modelo(args.modelo_path)
    df_pred = generar_predicciones(modelo, args.datos_path)
    insertar_predicciones(conn, df_pred, fecha)

    conn.close()
    logger.info("Proceso completado. La app Streamlit ya puede leer las predicciones.")


if __name__ == "__main__":
    main()
