"""
app.py — Producto de datos de pronóstico de ventas para 1C Company
Streamlit app con 5 vistas: predicciones, batch, evaluación, feedback y problemas.
Lee todo de RDS (PostgreSQL). Credenciales vía AWS Secrets Manager.
"""

import json
import os

import boto3
import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from botocore.exceptions import ClientError

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="1C Company — Pronóstico de Ventas",
    page_icon="📦",
    layout="wide",
)

# ─── Conexión a RDS via Secrets Manager ───────────────────────────────────────

def obtener_credenciales(secret_arn: str) -> dict:
    """Lee credenciales de RDS desde AWS Secrets Manager."""
    cliente = boto3.client("secretsmanager", region_name="us-east-1")
    try:
        respuesta = cliente.get_secret_value(SecretId=secret_arn)
        return json.loads(respuesta["SecretString"])
    except ClientError as e:
        st.error(f"Error al leer Secrets Manager: {e}")
        st.stop()


@st.cache_resource
def obtener_conexion():
    """
    Crea y cachea la conexión a PostgreSQL.
    Lee SECRET_ARN y RDS_HOST de variables de entorno (seteadas por ECS).
    """
    secret_arn = os.environ.get("SECRET_ARN", "")
    rds_host = os.environ.get("RDS_HOST", "localhost")

    # Si hay secret_arn, usamos Secrets Manager (producción en ECS)
    if secret_arn:
        creds = obtener_credenciales(secret_arn)
        usuario = creds["username"]
        password = creds["password"]
        base_datos = creds.get("dbname", "forecasting")
    else:
        # Fallback para desarrollo local con .env o valores hardcodeados
        usuario = os.environ.get("DB_USER", "itam")
        password = os.environ.get("DB_PASSWORD", "password")
        base_datos = os.environ.get("DB_NAME", "forecasting")

    conexion = psycopg2.connect(
        host=rds_host,
        port=5432,
        dbname=base_datos,
        user=usuario,
        password=password,
    )
    return conexion


@st.cache_data(ttl=300)
def ejecutar_query(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta un SELECT y devuelve un DataFrame. Cachea 5 minutos."""
    conn = obtener_conexion()
    return pd.read_sql(sql, conn, params=params)


def ejecutar_insert(sql: str, params: tuple):
    """Ejecuta un INSERT/UPDATE sin caché."""
    conn = obtener_conexion()
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


# ─── Sidebar de navegación ────────────────────────────────────────────────────
st.sidebar.title("📦 1C Company")
st.sidebar.caption("Producto de datos — Pronóstico de Ventas")

vista = st.sidebar.radio(
    "Navegar",
    [
        "Predicciones",
        "Batch / CFO",
        "Evaluación del modelo",
        "Feedback de negocio",
        "Productos con problemas",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Demo para consejo directivo · Abril 2026")


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — PREDICCIONES INDIVIDUALES
# ═══════════════════════════════════════════════════════════════════════════════
if vista == "Predicciones":
    st.title("📊 Predicciones de Ventas")
    st.caption("Filtra por tienda o categoría y visualiza el pronóstico del próximo mes")

    col1, col2 = st.columns(2)

    with col1:
        tiendas_df = ejecutar_query("SELECT shop_id, nombre FROM tiendas ORDER BY nombre")
        tienda_opciones = {"Todas": None} | dict(
            zip(tiendas_df["nombre"], tiendas_df["shop_id"])
        )
        tienda_sel = st.selectbox("Tienda", list(tienda_opciones.keys()))

    with col2:
        cats_df = ejecutar_query(
            "SELECT DISTINCT categoria_nombre FROM productos ORDER BY categoria_nombre"
        )
        cat_opciones = ["Todas"] + cats_df["categoria_nombre"].tolist()
        cat_sel = st.selectbox("Categoría", cat_opciones)

    # Construir query dinámico según filtros
    where_parts = []
    params = []

    if tienda_opciones[tienda_sel] is not None:
        where_parts.append("p.shop_id = %s")
        params.append(tienda_opciones[tienda_sel])

    if cat_sel != "Todas":
        where_parts.append("prod.categoria_nombre = %s")
        params.append(cat_sel)

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    sql = f"""
        SELECT
            prod.nombre AS producto,
            t.nombre    AS tienda,
            pred.fecha_mes,
            pred.valor_predicho,
            pred.valor_real,
            pred.rmse
        FROM predicciones pred
        JOIN productos prod ON pred.item_id = prod.item_id
        JOIN tiendas t      ON pred.shop_id = t.shop_id
        {where_clause}
        ORDER BY pred.fecha_mes DESC
        LIMIT 500
    """
    df = ejecutar_query(sql, params or None)

    if df.empty:
        st.warning("No hay predicciones para el filtro seleccionado.")
    else:
        # KPIs rápidos
        k1, k2, k3 = st.columns(3)
        k1.metric("Productos", f"{df['producto'].nunique():,}")
        k2.metric("RMSE promedio", f"{df['rmse'].mean():.2f}")
        k3.metric("Meses cubiertos", f"{df['fecha_mes'].nunique()}")

        st.divider()

        # Gráfica predicho vs real
        fig = px.line(
            df.sort_values("fecha_mes"),
            x="fecha_mes",
            y=["valor_predicho", "valor_real"],
            color_discrete_sequence=["#2563EB", "#10B981"],
            labels={"value": "Unidades", "fecha_mes": "Mes", "variable": "Serie"},
            title="Predicho vs Real — serie de tiempo",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla detalle
        with st.expander("Ver tabla completa"):
            st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 2 — BATCH / CFO
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "Batch / CFO":
    st.title("📥 Descarga Batch — Reporte CFO")
    st.caption("Genera el archivo de pronósticos para el mes siguiente y descárgalo como CSV")

    col1, col2 = st.columns(2)
    with col1:
        cats_df = ejecutar_query(
            "SELECT DISTINCT categoria_nombre FROM productos ORDER BY categoria_nombre"
        )
        cat_opciones = ["Todo el catálogo"] + cats_df["categoria_nombre"].tolist()
        cat_batch = st.selectbox("Selecciona categoría (o todo el catálogo)", cat_opciones)

    with col2:
        meses_df = ejecutar_query(
            "SELECT DISTINCT fecha_mes FROM predicciones ORDER BY fecha_mes DESC LIMIT 12"
        )
        mes_sel = st.selectbox("Mes del pronóstico", meses_df["fecha_mes"].tolist())

    if st.button("Generar archivo", type="primary"):
        where_cat = (
            "AND prod.categoria_nombre = %s" if cat_batch != "Todo el catálogo" else ""
        )
        params_batch = [mes_sel]
        if cat_batch != "Todo el catálogo":
            params_batch.insert(0, cat_batch)

        sql_batch = f"""
            SELECT
                prod.item_id,
                prod.nombre                 AS producto,
                prod.categoria_nombre       AS categoria,
                t.nombre                    AS tienda,
                pred.fecha_mes,
                pred.valor_predicho         AS pronostico_unidades,
                pred.rmse                   AS error_estimado
            FROM predicciones pred
            JOIN productos prod ON pred.item_id = prod.item_id
            JOIN tiendas t      ON pred.shop_id = t.shop_id
            WHERE pred.fecha_mes = %s
            {where_cat}
            ORDER BY prod.categoria_nombre, prod.nombre
        """
        df_batch = ejecutar_query(sql_batch, params_batch)

        if df_batch.empty:
            st.error("Sin datos para ese filtro y mes.")
        else:
            st.success(f"Archivo generado: {len(df_batch):,} registros")
            csv = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"pronostico_{mes_sel}_{cat_batch.replace(' ', '_')}.csv",
                mime="text/csv",
            )
            st.dataframe(df_batch.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 3 — EVALUACIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "Evaluación del modelo":
    st.title("🎯 Evaluación del Modelo")
    st.caption(
        "Métricas de error por grupo de productos y comparación contra baseline naive"
    )

    sql_eval = """
        SELECT
            prod.categoria_nombre AS categoria,
            COUNT(*)              AS num_predicciones,
            AVG(pred.rmse)        AS rmse_modelo,
            AVG(ABS(pred.valor_predicho - pred.valor_real)) AS mae_modelo
        FROM predicciones pred
        JOIN productos prod ON pred.item_id = prod.item_id
        WHERE pred.valor_real IS NOT NULL
        GROUP BY prod.categoria_nombre
        ORDER BY rmse_modelo DESC
    """
    df_eval = ejecutar_query(sql_eval)

    if df_eval.empty:
        st.info("Aún no hay datos de evaluación (necesitas cargar valor_real en las predicciones).")
    else:
        # KPIs globales
        k1, k2, k3 = st.columns(3)
        k1.metric("RMSE global", f"{df_eval['rmse_modelo'].mean():.2f}")
        k2.metric("MAE global", f"{df_eval['mae_modelo'].mean():.2f}")
        k3.metric("Categorías evaluadas", len(df_eval))

        st.divider()

        # Gráfica RMSE por categoría
        fig_rmse = px.bar(
            df_eval.sort_values("rmse_modelo", ascending=True),
            x="rmse_modelo",
            y="categoria",
            orientation="h",
            color="rmse_modelo",
            color_continuous_scale="RdYlGn_r",
            title="RMSE por categoría de producto",
            labels={"rmse_modelo": "RMSE", "categoria": "Categoría"},
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        # Scatter predicho vs real por producto (muestra)
        sql_scatter = """
            SELECT
                pred.valor_predicho,
                pred.valor_real,
                prod.categoria_nombre AS categoria
            FROM predicciones pred
            JOIN productos prod ON pred.item_id = prod.item_id
            WHERE pred.valor_real IS NOT NULL
            LIMIT 2000
        """
        df_scatter = ejecutar_query(sql_scatter)
        if not df_scatter.empty:
            fig_scatter = px.scatter(
                df_scatter,
                x="valor_real",
                y="valor_predicho",
                color="categoria",
                opacity=0.5,
                title="Predicho vs Real (muestra de 2000 puntos)",
                labels={"valor_real": "Real", "valor_predicho": "Predicho"},
            )
            # Línea diagonal perfecta
            max_val = max(df_scatter[["valor_real", "valor_predicho"]].max())
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("Ver tabla de métricas por categoría"):
            st.dataframe(df_eval, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 4 — FEEDBACK DE NEGOCIO
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "Feedback de negocio":
    st.title("💬 Feedback de Negocio")
    st.caption(
        "Captura observaciones sobre productos con predicciones incorrectas"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dejar un comentario")

        productos_df = ejecutar_query(
            "SELECT item_id, nombre FROM productos ORDER BY nombre LIMIT 200"
        )
        prod_opciones = dict(zip(productos_df["nombre"], productos_df["item_id"]))
        prod_sel = st.selectbox("Producto", list(prod_opciones.keys()))

        tiendas_df2 = ejecutar_query(
            "SELECT shop_id, nombre FROM tiendas ORDER BY nombre"
        )
        tienda_opciones2 = dict(
            zip(tiendas_df2["nombre"], tiendas_df2["shop_id"])
        )
        tienda_sel2 = st.selectbox("Tienda", list(tienda_opciones2.keys()), key="fb_tienda")

        usuario = st.text_input("Tu nombre / área", placeholder="ej. Planeación de demanda")
        comentario = st.text_area(
            "Comentario",
            placeholder="ej. Las predicciones de este producto no capturan la promo de diciembre",
        )

        marcar_problema = st.checkbox("Marcar como producto con problema")

        if st.button("Guardar comentario", type="primary"):
            if not comentario or not usuario:
                st.error("Completa nombre y comentario antes de guardar.")
            else:
                ejecutar_insert(
                    """
                    INSERT INTO feedback (item_id, shop_id, usuario, comentario)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        prod_opciones[prod_sel],
                        tienda_opciones2[tienda_sel2],
                        usuario,
                        comentario,
                    ),
                )
                if marcar_problema:
                    ejecutar_insert(
                        """
                        INSERT INTO productos_problema (item_id, shop_id, razon, estado)
                        VALUES (%s, %s, %s, 'pendiente')
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            prod_opciones[prod_sel],
                            tienda_opciones2[tienda_sel2],
                            comentario[:200],
                        ),
                    )
                st.success("Comentario guardado.")
                st.cache_data.clear()

    with col2:
        st.subheader("Comentarios recientes")
        df_fb = ejecutar_query(
            """
            SELECT
                prod.nombre     AS producto,
                t.nombre        AS tienda,
                fb.usuario,
                fb.comentario,
                fb.creado_en
            FROM feedback fb
            JOIN productos prod ON fb.item_id = prod.item_id
            JOIN tiendas t      ON fb.shop_id = t.shop_id
            ORDER BY fb.creado_en DESC
            LIMIT 50
            """
        )
        if df_fb.empty:
            st.info("Aún no hay comentarios.")
        else:
            for _, row in df_fb.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['producto']}** · {row['tienda']}")
                    st.markdown(f"_{row['comentario']}_")
                    st.caption(f"{row['usuario']} · {row['creado_en']}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 5 — PRODUCTOS CON PROBLEMAS
# ═══════════════════════════════════════════════════════════════════════════════
elif vista == "Productos con problemas":
    st.title("⚠️ Productos con Problemas")
    st.caption(
        "Lista de productos identificados por el equipo de negocio con predicciones incorrectas"
    )

    df_prob = ejecutar_query(
        """
        SELECT
            pp.id,
            prod.nombre         AS producto,
            t.nombre            AS tienda,
            prod.categoria_nombre AS categoria,
            pp.razon,
            pp.estado,
            pp.reportado_en,
            pred.rmse
        FROM productos_problema pp
        JOIN productos prod ON pp.item_id = prod.item_id
        JOIN tiendas t      ON pp.shop_id = t.shop_id
        LEFT JOIN predicciones pred
            ON pred.item_id = pp.item_id AND pred.shop_id = pp.shop_id
        ORDER BY pp.reportado_en DESC
        """
    )

    if df_prob.empty:
        st.success("No hay productos marcados con problemas.")
    else:
        # Filtro por estado
        estados = ["Todos"] + df_prob["estado"].unique().tolist()
        estado_fil = st.selectbox("Filtrar por estado", estados)
        if estado_fil != "Todos":
            df_prob = df_prob[df_prob["estado"] == estado_fil]

        st.metric("Productos con problemas", len(df_prob))
        st.divider()

        # Tabla interactiva con color por RMSE
        st.dataframe(
            df_prob.drop(columns=["id"]),
            use_container_width=True,
            column_config={
                "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "reportado_en": st.column_config.DatetimeColumn("Reportado"),
            },
        )

        # Distribución por categoría
        if len(df_prob) > 1:
            fig_pie = px.pie(
                df_prob,
                names="categoria",
                title="Distribución de problemas por categoría",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
