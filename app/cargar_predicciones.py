"""
cargar_predicciones.py — Corre tu modelo y carga predicciones en RDS.
Este es el script de "pre-cómputo" — se corre UNA VEZ (o cuando hay datos nuevos).
La app Streamlit SOLO LEE, nunca corre el modelo en tiempo real.

Uso desde SageMaker Studio:
    python cargar_predicciones.py \
        --host <RdsEndpoint> \
        --secret-arn <SecretArn> \
        --modelo-path artifacts/model.joblib \
        --x-path data/prep/X_test.csv \
        --y-path data/prep/y_valid.csv
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

FECHA_PRONOSTICO = date(2026, 1, 1)  # El mes que pronosticas — ajusta según tus datos


# ─── Helpers de conexión ──────────────────────────────────────────────────────

def obtener_credenciales(secret_arn: str) -> dict:
    """Lee user/pass desde AWS Secrets Manager."""
    cliente = boto3.client("secretsmanager", region_name="us-east-1")
    try:
        respuesta = cliente.get_secret_value(SecretId=secret_arn)
        return json.loads(respuesta["SecretString"])
    except Exception as e:
        logger.exception(f"Error al leer Secrets Manager: {e}")
        sys.exit(1)


def conectar(host: str, creds: dict) -> psycopg2.extensions.connection:
    """Crea conexión a PostgreSQL."""
    return psycopg2.connect(
        host=host,
        port=5432,
        dbname=creds.get("dbname", "forecasting"),
        user=creds["username"],
        password=creds["password"],
    )


# ─── Lógica principal ─────────────────────────────────────────────────────────

def cargar_modelo(ruta: str):
    """Carga el modelo serializado desde disco."""
    logger.info(f"Cargando modelo desde {ruta}...")
    try:
        modelo = joblib.load(ruta)
        logger.info("Modelo cargado OK.")
        return modelo
    except FileNotFoundError:
        logger.error(f"Modelo no encontrado en {ruta}")
        sys.exit(1)


def generar_predicciones(modelo, x_path: str, y_path: str = None) -> pd.DataFrame:
    """
    Lee los features (X_test.csv) y genera predicciones.
    Si se pasa y_path (y_valid.csv), calcula RMSE contra el ground truth.

    Parámetros
    ----------
    modelo  : modelo entrenado (joblib)
    x_path  : ruta a X_test.csv  — features del set de inferencia
    y_path  : ruta a y_valid.csv — ground truth opcional para evaluar
    """
    logger.info(f"Leyendo features desde {x_path}...")
    X = pd.read_csv(x_path)
    logger.info(f"Features leídos: {len(X):,} registros")

    # Guardamos item_id y shop_id antes de predecir
    # (no son features del modelo, son identificadores)
    columnas_id = [c for c in ["item_id", "shop_id"] if c in X.columns]
    df_ids = X[columnas_id].copy()
    columnas_features = [c for c in X.columns if c not in columnas_id]

    logger.info(f"Features usados para predecir: {columnas_features}")
    logger.info("Generando predicciones...")

    predicciones = modelo.predict(X[columnas_features])
    df_ids["valor_predicho"] = predicciones
    df_ids["valor_predicho"] = df_ids["valor_predicho"].clip(lower=0)  # sin ventas negativas

    # Ground truth opcional (y_valid.csv)
    if y_path:
        logger.info(f"Leyendo ground truth desde {y_path}...")
        try:
            y = pd.read_csv(y_path)
            # Tomamos la primera columna numérica como el target
            col_target = y.select_dtypes(include=[np.number]).columns[0]
            df_ids["valor_real"] = y[col_target].values
            df_ids["rmse"] = np.sqrt(
                (df_ids["valor_predicho"] - df_ids["valor_real"]) ** 2
            )
            rmse_global = np.sqrt(
                mean_squared_error(df_ids["valor_real"], df_ids["valor_predicho"])
            )
            logger.info(f"RMSE global del modelo: {rmse_global:.4f}")
        except FileNotFoundError:
            logger.warning(f"No se encontró {y_path}. Se continúa sin ground truth.")
            df_ids["valor_real"] = None
            df_ids["rmse"] = None
    else:
        logger.info("No se proporcionó y_path. Predicciones sin ground truth.")
        df_ids["valor_real"] = None
        df_ids["rmse"] = None

    logger.info(f"Predicciones generadas: {len(df_ids):,} registros")
    return df_ids[["item_id", "shop_id", "valor_predicho", "valor_real", "rmse"]]


def insertar_predicciones(conn, df: pd.DataFrame, fecha_mes: date):
    """Inserta las predicciones en la tabla RDS. Es idempotente — borra el mes antes."""
    logger.info(f"Insertando {len(df):,} predicciones para el mes {fecha_mes}...")

    registros = []
    for _, row in df.iterrows():
        valor_real = row["valor_real"]
        rmse = row["rmse"]
        registros.append((
            int(row["item_id"]),
            int(row["shop_id"]),
            fecha_mes,
            float(row["valor_predicho"]),
            float(valor_real) if valor_real is not None and not np.isnan(float(valor_real)) else None,
            float(rmse) if rmse is not None and not np.isnan(float(rmse)) else None,
            "v1.0",
        ))

    with conn.cursor() as cur:
        # Idempotencia: borramos predicciones anteriores del mismo mes
        cur.execute("DELETE FROM predicciones WHERE fecha_mes = %s", (fecha_mes,))
        deleted = cur.rowcount
        logger.info(f"Predicciones anteriores eliminadas: {deleted}")

        cur.executemany(
            """
            INSERT INTO predicciones
                (item_id, shop_id, fecha_mes, valor_predicho, valor_real, rmse, modelo_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            registros,
        )
    conn.commit()
    logger.info("Predicciones insertadas correctamente en RDS.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-computa predicciones y las carga en RDS"
    )
    parser.add_argument("--host", required=True,
                        help="RdsEndpoint del Output de CloudFormation")
    parser.add_argument("--secret-arn", required=True,
                        help="SecretArn del Output de CloudFormation")
    parser.add_argument("--modelo-path", default="artifacts/model.joblib",
                        help="Ruta al modelo .joblib entrenado")
    parser.add_argument("--x-path", default="data/prep/X_test.csv",
                        help="Ruta a X_test.csv (features de inferencia)")
    parser.add_argument("--y-path", default=None,
                        help="Ruta a y_valid.csv (ground truth, opcional)")
    parser.add_argument("--fecha-mes", default=str(FECHA_PRONOSTICO),
                        help="Mes del pronóstico en formato YYYY-MM-DD")
    args = parser.parse_args()

    fecha = date.fromisoformat(args.fecha_mes)
    logger.info(f"Iniciando carga de predicciones para el mes {fecha}...")

    # Conexión
    logger.info(f"Conectando a RDS en {args.host}...")
    try:
        creds = obtener_credenciales(args.secret_arn)
        conn = conectar(args.host, creds)
        logger.info("Conexión a RDS exitosa.")
    except Exception as e:
        logger.exception(f"No se pudo conectar a RDS: {e}")
        sys.exit(1)

    # Pipeline: cargar modelo → predecir → insertar
    modelo = cargar_modelo(args.modelo_path)
    df_pred = generar_predicciones(modelo, args.x_path, args.y_path)
    insertar_predicciones(conn, df_pred, fecha)

    conn.close()
    logger.info("Proceso completado. La app Streamlit ya puede leer las predicciones.")


if __name__ == "__main__":
    main()
