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
    st.title("Forecasting Data Product")
    st.caption("MVP de planeación de demanda mensual para retail.")

    model_metrics_global = read_gold_table("model_metrics_global")
    model_metrics_by_category = read_gold_table("model_metrics_by_category")
    category_monthly = read_gold_table("category_monthly")
    product_kpis = read_gold_table("product_kpis")
    forecast = read_gold_table("model_forecast_next_month")

    metric_row = model_metrics_global.iloc[0]

    model_name = str(metric_row.get("model_name", "GradientBoostingRegressor"))
    feature_set = str(metric_row.get("feature_set", "task_01_original_features"))

    actual_total = float(metric_row["actual_total"])
    predicted_total = float(metric_row["model_prediction_total"])
    total_gap = predicted_total - actual_total
    total_gap_pct = (total_gap / actual_total) * 100 if actual_total != 0 else 0

    st.markdown(
        """
        ### ¿Qué resuelve este producto?

        Este MVP convierte ventas históricas en **pronósticos mensuales de demanda**
        a nivel **tienda-producto**. La aplicación permite que un usuario de negocio
        consulte demanda esperada, revise desempeño del modelo y deje feedback operativo
        desde una URL pública.

        Las predicciones se calculan fuera de la app y se guardan en la capa **Gold**.
        Streamlit solo consulta resultados ya preparados, lo que hace que el dashboard
        sea rápido y estable.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE modelo", f"{metric_row['model_mae']:.3f}")
    c2.metric("RMSE modelo", f"{metric_row['model_rmse']:.3f}")
    c3.metric("R² modelo", f"{metric_row['model_r2']:.3f}")
    c4.metric("Filas evaluación", f"{metric_row['n_rows']:,.0f}")

    c5, c6, c7 = st.columns(3)

    c5.metric("Demanda real validación", f"{actual_total:,.0f}")
    c6.metric("Demanda predicha validación", f"{predicted_total:,.0f}")
    c7.metric("Diferencia agregada", f"{total_gap:,.0f}", f"{total_gap_pct:.1f}%")

    st.info(
        f"Modelo final: **{model_name}**. "
        f"Feature set: **{feature_set}**. "
        "Las métricas se calculan usando el último mes histórico como validación."
    )

    st.markdown(
        """
        ### Hallazgos principales

        - La demanda es muy dispersa: muchas combinaciones tienda-producto tienen demanda esperada menor a 1 unidad mensual.
        - Por eso, la vista más útil para planeación no es solo la fila granular, sino la demanda agregada por categoría, tienda o producto.
        - El modelo permite identificar dónde se concentra la demanda esperada y dónde conviene revisar errores o productos problemáticos.
        """
    )

    st.subheader("Demanda esperada del siguiente mes por categoría")

    forecast_by_category = (
        forecast.groupby(["category_id", "category_name"], as_index=False)
        .agg(
            predicted_demand=("model_prediction", "sum"),
            avg_prediction=("model_prediction", "mean"),
            shop_item_pairs=("item_id", "size"),
            unique_items=("item_id", "nunique"),
            unique_shops=("shop_id", "nunique"),
        )
        .sort_values("predicted_demand", ascending=False)
        .head(15)
    )

    fig_forecast = px.bar(
        forecast_by_category,
        x="predicted_demand",
        y="category_name",
        orientation="h",
        title="Top 15 categorías por demanda esperada",
        labels={
            "predicted_demand": "Demanda esperada",
            "category_name": "Categoría",
        },
    )
    fig_forecast.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Demanda histórica mensual de categorías principales")

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

    fig_history = px.line(
        chart_data,
        x="year_month",
        y="total_item_cnt_month_clipped",
        color="category_name",
        markers=True,
        title="Demanda histórica mensual: top 10 categorías",
        labels={
            "year_month": "Mes",
            "total_item_cnt_month_clipped": "Unidades vendidas",
            "category_name": "Categoría",
        },
    )

    st.plotly_chart(fig_history, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Categorías con mayor demanda esperada")
        st.dataframe(
            forecast_by_category[
                [
                    "category_id",
                    "category_name",
                    "predicted_demand",
                    "avg_prediction",
                    "shop_item_pairs",
                    "unique_items",
                    "unique_shops",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Productos históricamente más relevantes")
        st.dataframe(
            product_kpis.sort_values("total_item_cnt_month_clipped", ascending=False)
            .head(20)[
                [
                    "item_id",
                    "item_name",
                    "category_name",
                    "total_item_cnt_month_clipped",
                    "total_revenue_month",
                    "active_months",
                    "active_shops",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

def page_forecast() -> None:
    st.title("Pronóstico siguiente mes")
    st.caption("Predicción mensual de demanda generada por el modelo final.")

    forecast = read_gold_table("model_forecast_next_month")

    st.info(
        "Las predicciones representan demanda esperada mensual por combinación tienda-producto. "
        "En productos de baja rotación es normal observar valores menores a 1. "
        "Para planeación, las vistas agregadas por categoría, tienda o producto son más útiles "
        "que una fila individual."
    )

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

    total_prediction = filtered["model_prediction"].sum()
    avg_prediction = filtered["model_prediction"].mean() if len(filtered) else 0
    pairs_over_one = (filtered["model_prediction"] >= 1).sum()
    pairs_over_half = (filtered["model_prediction"] >= 0.5).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Demanda esperada total", f"{total_prediction:,.1f}")
    c2.metric("Promedio tienda-producto", f"{avg_prediction:.3f}")
    c3.metric("Combinaciones ≥ 1 unidad", f"{pairs_over_one:,.0f}")
    c4.metric("Combinaciones ≥ 0.5 unidad", f"{pairs_over_half:,.0f}")

    tab_category, tab_shop, tab_product, tab_detail = st.tabs(
        [
            "Por categoría",
            "Por tienda",
            "Top productos",
            "Detalle tienda-producto",
        ]
    )

    with tab_category:
        st.subheader("Pronóstico agregado por categoría")

        by_category = (
            filtered.groupby(["category_id", "category_name"], as_index=False)
            .agg(
                predicted_demand=("model_prediction", "sum"),
                avg_prediction=("model_prediction", "mean"),
                shop_item_pairs=("item_id", "size"),
                unique_items=("item_id", "nunique"),
                unique_shops=("shop_id", "nunique"),
            )
            .sort_values("predicted_demand", ascending=False)
        )

        fig = px.bar(
            by_category.head(20),
            x="predicted_demand",
            y="category_name",
            orientation="h",
            title="Top 20 categorías por demanda esperada",
            labels={
                "predicted_demand": "Demanda esperada",
                "category_name": "Categoría",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            by_category,
            use_container_width=True,
            hide_index=True,
        )

    with tab_shop:
        st.subheader("Pronóstico agregado por tienda")

        group_cols = ["shop_id"]

        if "shop_name" in filtered.columns and filtered["shop_name"].astype(str).str.len().sum() > 0:
            group_cols.append("shop_name")

        by_shop = (
            filtered.groupby(group_cols, as_index=False)
            .agg(
                predicted_demand=("model_prediction", "sum"),
                avg_prediction=("model_prediction", "mean"),
                shop_item_pairs=("item_id", "size"),
                unique_items=("item_id", "nunique"),
                unique_categories=("category_id", "nunique"),
            )
            .sort_values("predicted_demand", ascending=False)
        )

        label_col = "shop_name" if "shop_name" in by_shop.columns else "shop_id"

        fig = px.bar(
            by_shop.head(20),
            x="predicted_demand",
            y=label_col,
            orientation="h",
            title="Top tiendas por demanda esperada",
            labels={
                "predicted_demand": "Demanda esperada",
                label_col: "Tienda",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            by_shop,
            use_container_width=True,
            hide_index=True,
        )

    with tab_product:
        st.subheader("Top productos por demanda esperada")

        by_product = (
            filtered.groupby(
                [
                    "item_id",
                    "item_name",
                    "category_id",
                    "category_name",
                ],
                as_index=False,
            )
            .agg(
                predicted_demand=("model_prediction", "sum"),
                avg_prediction=("model_prediction", "mean"),
                active_shops=("shop_id", "nunique"),
                shop_item_pairs=("shop_id", "size"),
                last_month_demand=("lag_1", "sum"),
                recent_avg_demand=("lag_mean_3", "sum"),
            )
            .sort_values("predicted_demand", ascending=False)
        )

        st.dataframe(
            by_product.head(100),
            use_container_width=True,
            hide_index=True,
        )

    with tab_detail:
        st.subheader("Detalle granular tienda-producto")

        st.caption(
            "Esta tabla es para auditoría y drill-down. "
            "Un valor como 0.35 significa demanda esperada mensual, "
            "no una orden física redondeada."
        )

        detail = filtered.copy()

        for col in ["model_prediction", "lag_1", "lag_mean_3", "lag_mean_6"]:
            if col in detail.columns:
                detail[col] = detail[col].round(4)

        columns = [
            "prediction_month",
            "shop_id",
            "item_id",
            "item_name",
            "category_id",
            "category_name",
            "model_prediction",
            "lag_1",
            "lag_mean_3",
            "lag_mean_6",
        ]

        if "shop_name" in detail.columns and detail["shop_name"].astype(str).str.len().sum() > 0:
            columns.insert(2, "shop_name")

        st.dataframe(
            detail[columns]
            .sort_values("model_prediction", ascending=False)
            .head(1000),
            use_container_width=True,
            hide_index=True,
        )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar forecast filtrado CSV",
        data=csv,
        file_name="model_forecast_next_month.csv",
        mime="text/csv",
    )

def page_evaluation() -> None:
    st.title("Evaluación del modelo")
    st.caption("Validación del modelo sobre el último mes histórico disponible.")

    evaluation = read_gold_table("model_evaluation")
    category_metrics = read_gold_table("model_metrics_by_category")
    global_metrics = read_gold_table("model_metrics_global")

    metric_row = global_metrics.iloc[0]

    st.markdown(
        """
        Esta sección resume qué tan bien el modelo reproduce la demanda observada
        en el último mes histórico. La vista principal se concentra en el modelo final;
        la comparación contra baseline queda como referencia técnica.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Filas evaluación", f"{len(evaluation):,.0f}")
    c2.metric("MAE modelo", f"{metric_row['model_mae']:.3f}")
    c3.metric("RMSE modelo", f"{metric_row['model_rmse']:.3f}")
    c4.metric("R² modelo", f"{metric_row['model_r2']:.3f}")

    actual_total = float(metric_row["actual_total"])
    predicted_total = float(metric_row["model_prediction_total"])
    total_gap = predicted_total - actual_total
    total_gap_pct = (total_gap / actual_total) * 100 if actual_total != 0 else 0

    c5, c6, c7 = st.columns(3)

    c5.metric("Demanda real", f"{actual_total:,.0f}")
    c6.metric("Demanda predicha", f"{predicted_total:,.0f}")
    c7.metric("Diferencia agregada", f"{total_gap:,.0f}", f"{total_gap_pct:.1f}%")

    st.subheader("Demanda real vs demanda predicha por categoría")

    fig = px.scatter(
        category_metrics,
        x="actual_total",
        y="model_prediction_total",
        size="n_rows",
        color="category_name",
        hover_name="category_name",
        title="Demanda real vs predicción del modelo",
        labels={
            "actual_total": "Demanda real",
            "model_prediction_total": "Demanda predicha",
            "n_rows": "Combinaciones",
            "category_name": "Categoría",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorías con mayor error agregado")

    clean_metrics = category_metrics.copy()
    clean_metrics["absolute_total_error"] = (
        clean_metrics["model_prediction_total"] - clean_metrics["actual_total"]
    ).abs()

    st.dataframe(
        clean_metrics[
            [
                "category_id",
                "category_name",
                "n_rows",
                "actual_total",
                "model_prediction_total",
                "absolute_total_error",
                "model_mae",
                "model_rmse",
                "model_bias",
            ]
        ].sort_values("absolute_total_error", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Casos individuales con mayor error")

    st.dataframe(
        evaluation[
            [
                "date_block_num",
                "shop_id",
                "item_id",
                "item_name",
                "category_id",
                "category_name",
                "actual_item_cnt_month",
                "model_prediction",
                "model_error",
                "model_absolute_error",
            ]
        ].sort_values("model_absolute_error", ascending=False).head(500),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Ver comparación técnica contra baseline"):
        st.write(
            "Esta sección se conserva para auditoría técnica. "
            "La app ejecutiva usa el modelo final como resultado principal."
        )

        cols = [
            "validation_date_block_num",
            "model_mae",
            "baseline_mae",
            "model_rmse",
            "baseline_rmse",
            "model_r2",
            "baseline_r2",
            "actual_total",
            "model_prediction_total",
            "baseline_prediction_total",
            "mae_improvement",
            "rmse_improvement",
        ]

        available_cols = [col for col in cols if col in global_metrics.columns]

        st.dataframe(
            global_metrics[available_cols],
            use_container_width=True,
            hide_index=True,
        )

def page_feedback() -> None:
    st.title("Feedback de negocio")
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
    st.title("Productos marcados con problemas")
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
