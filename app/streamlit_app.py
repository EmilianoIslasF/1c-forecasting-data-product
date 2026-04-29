from __future__ import annotations

import json
import os
from urllib.parse import quote_plus

import boto3
import awswrangler as wr
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GOLD_DATABASE = os.getenv("GOLD_DATABASE", "forecasting_gold")
RDS_ENDPOINT = os.getenv("RDS_ENDPOINT")
RDS_SECRET_NAME = os.getenv("RDS_SECRET_NAME", "itam/rds/forecasting/credentials")


st.set_page_config(
    page_title="Forecasting Data Product",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=600)
def read_gold_table(table_name: str) -> pd.DataFrame:
    return wr.s3.read_parquet_table(
        database=GOLD_DATABASE,
        table=table_name,
    )


@st.cache_resource
def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


@st.cache_resource
def get_engine():
    if not RDS_ENDPOINT:
        return None

    creds = get_secret(RDS_SECRET_NAME)

    username = quote_plus(creds["username"])
    password = quote_plus(creds["password"])
    dbname = creds["dbname"]
    port = creds.get("port", "5432")

    url = f"postgresql+psycopg2://{username}:{password}@{RDS_ENDPOINT}:{port}/{dbname}"

    return create_engine(url, pool_pre_ping=True)


def insert_feedback(
    created_by: str,
    shop_id: int | None,
    item_id: int | None,
    category_id: int | None,
    forecast_month: int | None,
    severity: str,
    feedback_text: str,
) -> None:
    engine = get_engine()

    if engine is None:
        raise RuntimeError("RDS_ENDPOINT is not configured.")

    query = text(
        """
        INSERT INTO business_feedback (
            created_by,
            shop_id,
            item_id,
            category_id,
            forecast_month,
            severity,
            status,
            feedback_text
        )
        VALUES (
            :created_by,
            :shop_id,
            :item_id,
            :category_id,
            :forecast_month,
            :severity,
            'open',
            :feedback_text
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "created_by": created_by,
                "shop_id": shop_id,
                "item_id": item_id,
                "category_id": category_id,
                "forecast_month": forecast_month,
                "severity": severity,
                "feedback_text": feedback_text,
            },
        )


def insert_flagged_product(
    created_by: str,
    shop_id: int | None,
    item_id: int,
    category_id: int | None,
    reason: str,
    priority: str,
    notes: str | None,
) -> None:
    engine = get_engine()

    if engine is None:
        raise RuntimeError("RDS_ENDPOINT is not configured.")

    query = text(
        """
        INSERT INTO flagged_products (
            created_by,
            shop_id,
            item_id,
            category_id,
            reason,
            priority,
            status,
            notes
        )
        VALUES (
            :created_by,
            :shop_id,
            :item_id,
            :category_id,
            :reason,
            :priority,
            'open',
            :notes
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "created_by": created_by,
                "shop_id": shop_id,
                "item_id": item_id,
                "category_id": category_id,
                "reason": reason,
                "priority": priority,
                "notes": notes,
            },
        )


def read_feedback() -> pd.DataFrame:
    engine = get_engine()

    if engine is None:
        return pd.DataFrame()

    query = text(
        """
        SELECT
            feedback_id,
            created_at,
            created_by,
            shop_id,
            item_id,
            category_id,
            forecast_month,
            severity,
            status,
            feedback_text
        FROM business_feedback
        ORDER BY created_at DESC
        LIMIT 200
        """
    )

    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def read_flagged_products() -> pd.DataFrame:
    engine = get_engine()

    if engine is None:
        return pd.DataFrame()

    query = text(
        """
        SELECT
            flag_id,
            created_at,
            created_by,
            shop_id,
            item_id,
            category_id,
            reason,
            priority,
            status,
            notes
        FROM flagged_products
        ORDER BY created_at DESC
        LIMIT 200
        """
    )

    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def sidebar_filters(forecast_df: pd.DataFrame) -> tuple[list[str], list[int], list[int]]:
    st.sidebar.header("Filtros")

    categories = sorted(
        forecast_df["category_name"].dropna().astype(str).unique().tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Categoría",
        options=categories,
        default=categories[:5] if len(categories) > 5 else categories,
    )

    filtered = forecast_df[
        forecast_df["category_name"].astype(str).isin(selected_categories)
    ]

    shops = sorted(filtered["shop_id"].dropna().astype(int).unique().tolist())
    selected_shops = st.sidebar.multiselect(
        "Tienda",
        options=shops,
        default=shops[:5] if len(shops) > 5 else shops,
    )

    items = sorted(filtered["item_id"].dropna().astype(int).unique().tolist())
    selected_items = st.sidebar.multiselect(
        "Producto",
        options=items,
        default=items[:10] if len(items) > 10 else items,
    )

    return selected_categories, selected_shops, selected_items


def page_overview() -> None:
    st.title("📈 Forecasting Data Product")
    st.caption("MVP para planeación de demanda, finanzas y operaciones.")

    metrics_global = read_gold_table("baseline_metrics_global")
    category_metrics = read_gold_table("baseline_metrics_by_category")
    category_monthly = read_gold_table("category_monthly")
    product_kpis = read_gold_table("product_kpis")

    metric_row = metrics_global.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Shop-item pairs evaluados", f"{metric_row['n_shop_item_pairs']:,.0f}")
    c2.metric("MAE baseline", f"{metric_row['mae']:.3f}")
    c3.metric("RMSE baseline", f"{metric_row['rmse']:.3f}")
    c4.metric("Bias baseline", f"{metric_row['bias']:.3f}")

    st.subheader("Demanda mensual por categoría")

    top_categories = (
        category_monthly.groupby("category_name", as_index=False)
        .agg(total_sales=("total_item_cnt_month_clipped", "sum"))
        .sort_values("total_sales", ascending=False)
        .head(10)
    )

    selected_categories = top_categories["category_name"].tolist()

    chart_data = category_monthly[
        category_monthly["category_name"].isin(selected_categories)
    ].copy()

    fig = px.line(
        chart_data,
        x="year_month",
        y="total_item_cnt_month_clipped",
        color="category_name",
        markers=True,
        title="Top 10 categorías por demanda histórica",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorías con mayor error del baseline")

    category_metrics_sorted = category_metrics.sort_values("rmse", ascending=False)

    st.dataframe(
        category_metrics_sorted[
            [
                "category_id",
                "category_name",
                "n_shop_item_pairs",
                "actual_total",
                "prediction_total",
                "mae",
                "rmse",
                "bias",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Productos principales")

    st.dataframe(
        product_kpis.sort_values("total_item_cnt_month_clipped", ascending=False)
        .head(50)[
            [
                "item_id",
                "item_name",
                "category_name",
                "total_item_cnt_month_clipped",
                "total_revenue_month",
                "active_months",
                "active_shops",
                "avg_sales_per_active_month",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def page_forecast() -> None:
    st.title("🔮 Pronóstico siguiente mes")
    st.caption("Forecast baseline precalculado en Gold para el mes siguiente al entrenamiento.")

    forecast = read_gold_table("baseline_forecast_next_month")

    selected_categories, selected_shops, selected_items = sidebar_filters(forecast)

    filtered = forecast.copy()

    if selected_categories:
        filtered = filtered[
            filtered["category_name"].astype(str).isin(selected_categories)
        ]

    if selected_shops:
        filtered = filtered[filtered["shop_id"].isin(selected_shops)]

    if selected_items:
        filtered = filtered[filtered["item_id"].isin(selected_items)]

    c1, c2, c3 = st.columns(3)

    c1.metric("Filas filtradas", f"{len(filtered):,.0f}")
    c2.metric("Predicción total", f"{filtered['baseline_prediction'].sum():,.0f}")
    c3.metric("Productos únicos", f"{filtered['item_id'].nunique():,.0f}")

    st.subheader("Tabla de pronósticos")

    st.dataframe(
        filtered[
            [
                "id",
                "prediction_month",
                "shop_id",
                "shop_name",
                "item_id",
                "item_name",
                "category_id",
                "category_name",
                "baseline_last_month",
                "baseline_3_month_avg",
                "baseline_prediction",
            ]
        ].sort_values("baseline_prediction", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar forecast filtrado CSV",
        data=csv,
        file_name="forecast_next_month.csv",
        mime="text/csv",
    )


def page_evaluation() -> None:
    st.title("📊 Evaluación del baseline")
    st.caption("Comparación entre último mes real y predicción naive del mes anterior.")

    evaluation = read_gold_table("baseline_evaluation")
    category_metrics = read_gold_table("baseline_metrics_by_category")

    c1, c2, c3 = st.columns(3)

    c1.metric("Filas evaluación", f"{len(evaluation):,.0f}")
    c2.metric("MAE promedio", f"{evaluation['absolute_error'].mean():.3f}")
    c3.metric("RMSE", f"{(evaluation['squared_error'].mean() ** 0.5):.3f}")

    st.subheader("Actual vs predicción por categoría")

    fig = px.scatter(
        category_metrics,
        x="actual_total",
        y="prediction_total",
        size="n_shop_item_pairs",
        color="category_name",
        hover_name="category_name",
        title="Actual total vs predicción total por categoría",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detalle de evaluación")

    st.dataframe(
        evaluation[
            [
                "shop_id",
                "shop_name",
                "item_id",
                "item_name",
                "category_id",
                "category_name",
                "actual_item_cnt_month",
                "baseline_prediction",
                "error",
                "absolute_error",
                "squared_error",
            ]
        ].sort_values("absolute_error", ascending=False).head(500),
        use_container_width=True,
        hide_index=True,
    )


def page_feedback() -> None:
    st.title("📝 Feedback de negocio")
    st.caption("Captura observaciones del equipo de negocio sobre productos o categorías.")

    with st.form("feedback_form"):
        created_by = st.text_input("Nombre / usuario", value="analista_negocio")
        shop_id = st.number_input("shop_id opcional", min_value=0, step=1, value=0)
        item_id = st.number_input("item_id opcional", min_value=0, step=1, value=0)
        category_id = st.number_input("category_id opcional", min_value=0, step=1, value=0)
        forecast_month = st.number_input("forecast_month opcional", min_value=0, step=1, value=34)

        severity = st.selectbox(
            "Severidad",
            options=["low", "medium", "high", "critical"],
            index=1,
        )

        feedback_text = st.text_area(
            "Observación",
            placeholder="Ej. La predicción se ve muy baja para esta categoría por temporada alta.",
        )

        submitted = st.form_submit_button("Guardar feedback")

    if submitted:
        if not feedback_text.strip():
            st.error("Escribe una observación antes de guardar.")
        else:
            try:
                insert_feedback(
                    created_by=created_by,
                    shop_id=shop_id if shop_id != 0 else None,
                    item_id=item_id if item_id != 0 else None,
                    category_id=category_id if category_id != 0 else None,
                    forecast_month=forecast_month if forecast_month != 0 else None,
                    severity=severity,
                    feedback_text=feedback_text,
                )
                st.success("Feedback guardado en RDS.")
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"No se pudo guardar feedback: {exc}")

    st.subheader("Feedback reciente")

    feedback = read_feedback()

    if feedback.empty:
        st.info("Todavía no hay feedback o RDS_ENDPOINT no está configurado.")
    else:
        st.dataframe(feedback, use_container_width=True, hide_index=True)


def page_flagged_products() -> None:
    st.title("🚩 Productos marcados con problemas")
    st.caption("Lista operacional para que el equipo de ML investigue casos problemáticos.")

    with st.form("flag_form"):
        created_by = st.text_input("Nombre / usuario", value="analista_negocio")
        shop_id = st.number_input("shop_id opcional", min_value=0, step=1, value=0)
        item_id = st.number_input("item_id", min_value=0, step=1)
        category_id = st.number_input("category_id opcional", min_value=0, step=1, value=0)

        reason = st.selectbox(
            "Razón",
            options=[
                "forecast_too_high",
                "forecast_too_low",
                "missing_product_context",
                "seasonality_not_captured",
                "other",
            ],
        )

        priority = st.selectbox(
            "Prioridad",
            options=["low", "medium", "high", "critical"],
            index=1,
        )

        notes = st.text_area("Notas")

        submitted = st.form_submit_button("Marcar producto")

    if submitted:
        try:
            insert_flagged_product(
                created_by=created_by,
                shop_id=shop_id if shop_id != 0 else None,
                item_id=int(item_id),
                category_id=category_id if category_id != 0 else None,
                reason=reason,
                priority=priority,
                notes=notes,
            )
            st.success("Producto marcado en RDS.")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"No se pudo marcar el producto: {exc}")

    st.subheader("Productos marcados recientemente")

    flagged = read_flagged_products()

    if flagged.empty:
        st.info("Todavía no hay productos marcados o RDS_ENDPOINT no está configurado.")
    else:
        st.dataframe(flagged, use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.title("Forecasting MVP")

    page = st.sidebar.radio(
        "Navegación",
        options=[
            "Resumen ejecutivo",
            "Pronóstico",
            "Evaluación",
            "Feedback",
            "Productos marcados",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Gold database: {GOLD_DATABASE}")

    if RDS_ENDPOINT:
        st.sidebar.success("RDS configurado")
    else:
        st.sidebar.warning("RDS_ENDPOINT no configurado")

    if page == "Resumen ejecutivo":
        page_overview()
    elif page == "Pronóstico":
        page_forecast()
    elif page == "Evaluación":
        page_evaluation()
    elif page == "Feedback":
        page_feedback()
    elif page == "Productos marcados":
        page_flagged_products()


if __name__ == "__main__":
    main()
