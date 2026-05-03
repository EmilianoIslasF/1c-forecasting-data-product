# Forecasting Data Product — 1C Company

MVP end-to-end para planeación de demanda mensual en retail. El proyecto implementa una arquitectura Medallion en AWS, un pipeline de Machine Learning batch, una app pública en Streamlit y una base operacional en RDS PostgreSQL para capturar feedback de negocio.

---

## App pública

```text
http://forecasting-data-product-alb-1021855290.us-east-1.elb.amazonaws.com/
```

La app permite:

- Consultar un resumen ejecutivo del modelo.
- Explorar pronósticos por categoría, tienda y producto.
- Revisar la evaluación del modelo.
- Descargar predicciones filtradas en CSV.
- Capturar feedback de negocio.
- Marcar productos con problemas para revisión.

---

## Problema de negocio

El objetivo es apoyar la planeación de demanda mensual a nivel tienda-producto usando datos históricos de ventas.

El producto ayuda a reducir dos riesgos principales:

- **Sobrestock:** exceso de inventario y costos de almacenamiento.
- **Quiebres de stock:** falta de productos y pérdida de ventas.

---

## Arquitectura

La solución sigue un patrón Medallion:

```text
Bronze → Silver → Gold
```

Además, se agrega una rama de Machine Learning:

```text
Silver → Features ML → Entrenamiento → Outputs del modelo en Gold → Streamlit
```

### Diagrama de arquitectura

<img width="1450" height="787" alt="architecture" src="https://github.com/user-attachments/assets/97d5afbc-b8f9-42f0-b365-4336e43cb70e" />

---

## Servicios de AWS utilizados

| Servicio | Uso |
|---|---|
| S3 | Data lake para raw, bronze, silver, gold, features ML y artefactos |
| Glue Data Catalog | Catálogo de tablas |
| Athena | Consultas SQL sobre datos en S3 |
| RDS PostgreSQL | Feedback operacional |
| Secrets Manager | Credenciales de RDS |
| ECR | Imagen Docker de Streamlit |
| ECS Fargate | Ejecución pública de la app |
| Application Load Balancer | URL pública |
| CloudFormation | Infraestructura como código |

---

## Estructura del repositorio

```text
.
├── app/
│   └── streamlit_app.py
├── etl/
│   ├── bronze.py
│   ├── silver.py
│   └── gold.py
├── ml/
│   ├── build_features.py
│   └── train_model.py
├── postgres/
│   └── create_tables.py
├── infra/
│   ├── forecasting-data-product-foundation.yaml
│   ├── rds-forecasting.yaml
│   └── ecs-streamlit.yaml
├── docs/
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Pipeline de datos

### Bronze

Script:

```text
etl/bronze.py
```

Hace lo siguiente:

- Descarga el dataset desde Kaggle.
- Guarda CSVs en `data/raw`.
- Sube CSVs a S3 raw.
- Convierte los datos a Parquet.
- Registra tablas en Glue.

Base de Glue:

```text
forecasting_bronze
```

Tablas principales:

```text
sales_train
items
item_categories
shops
test
sample_submission
```

Ejecución:

```bash
uv run python etl/bronze.py --bucket forecasting-data-product
```

---

### Silver

Script:

```text
etl/silver.py
```

Hace lo siguiente:

- Limpia datos Bronze.
- Agrega ventas mensuales.
- Enriquece con catálogos.
- Construye el input de inferencia.

Base de Glue:

```text
forecasting_silver
```

Tablas principales:

```text
item_catalog
shop_catalog
sales_monthly
sales_monthly_enriched
forecast_input
```

Ejecución:

```bash
uv run python etl/silver.py --bucket forecasting-data-product
```

---

### Gold

Script:

```text
etl/gold.py
```

Hace lo siguiente:

- Construye tablas analíticas.
- Genera KPIs por categoría, tienda y producto.
- Calcula un baseline naive.

Base de Glue:

```text
forecasting_gold
```

Tablas principales:

```text
demand_history
category_monthly
shop_monthly
product_kpis
baseline_forecast_next_month
baseline_evaluation
baseline_metrics_global
baseline_metrics_by_category
```

Ejecución:

```bash
uv run python etl/gold.py --bucket forecasting-data-product
```

---

## Pipeline de Machine Learning

El modelo se ejecuta en batch. Streamlit no recalcula predicciones en vivo; solo lee resultados ya guardados en Gold.

### Ingeniería de features

Script:

```text
ml/build_features.py
```

Hace lo siguiente:

- Construye una matriz tienda-producto-mes.
- Rellena meses sin venta con 0.
- Crea variables temporales y lags.
- Genera train, validation e inference.

Base de Glue:

```text
forecasting_ml
```

Tablas:

```text
train_features
validation_features
inference_features
```

Ejecución:

```bash
uv run python ml/build_features.py --bucket forecasting-data-product
```

---

### Entrenamiento y predicción batch

Script:

```text
ml/train_model.py
```

Modelo:

```text
GradientBoostingRegressor
```

Features principales:

```text
date_block_num
shop_id
item_id
item_category_id
month
year
item_cnt_month_lag_1
item_cnt_month_lag_2
item_cnt_month_lag_3
item_cnt_month_lag_6
item_cnt_month_lag_12
```

Outputs:

```text
model.joblib
model_metrics.json
model_forecast_next_month
model_evaluation
model_metrics_global
model_metrics_by_category
```

Ejecución:

```bash
uv run python ml/train_model.py --bucket forecasting-data-product
```

---

## Métricas del modelo

La validación usa el último mes histórico disponible.

| Métrica | Valor |
|---|---:|
| Filas de evaluación | 238,172 |
| MAE | 0.346 |
| RMSE | 0.973 |
| R² | 0.267 |
| Demanda real | 61,583 |
| Demanda predicha | 62,792 |
---

## Aplicación Streamlit

Archivo principal:

```text
app/streamlit_app.py
```

La app lee resultados desde:

```text
forecasting_gold
```

Páginas principales:

- **Resumen ejecutivo:** métricas, hallazgos y demanda esperada.
- **Pronóstico:** demanda por categoría, tienda y producto.
- **Evaluación:** métricas del modelo y análisis de error.
- **Feedback:** captura feedback de negocio en RDS.
- **Productos marcados:** guarda productos problemáticos en RDS.

---

## Base operacional en RDS

RDS PostgreSQL guarda información generada desde la app.

Tablas:

```text
forecast_jobs
business_feedback
flagged_products
app_metrics
```

RDS no almacena las tablas analíticas. Los pronósticos y outputs del modelo viven en S3/Glue.

### Diagrama ERD

<img width="1030" height="879" alt="erd" src="https://github.com/user-attachments/assets/ed3ae58d-b9e7-4ab6-8485-d2a25ea1dd70" />

---

## Despliegue

La app se empaqueta como imagen Docker, se sube a ECR y se ejecuta en ECS Fargate detrás de un Application Load Balancer.

```text
Streamlit App
→ Docker Image
→ ECR
→ ECS Fargate
→ Application Load Balancer
→ URL pública
```

Construir y subir imagen:

```bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPOSITORY="forecasting-data-product-streamlit"
export IMAGE_TAG="streamlit-clean-$(date +%Y%m%d%H%M%S)"
export IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

aws ecr get-login-password --region $AWS_REGION \
  | docker login \
      --username AWS \
      --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build --network sagemaker -t $ECR_REPOSITORY:$IMAGE_TAG .
docker tag $ECR_REPOSITORY:$IMAGE_TAG $IMAGE_URI
docker push $IMAGE_URI
```

Después se actualiza el parámetro `ImageUri` en el stack de ECS de CloudFormation.

---

## Infraestructura

Templates de CloudFormation:

```text
infra/forecasting-data-product-foundation.yaml
infra/rds-forecasting.yaml
infra/ecs-streamlit.yaml
```

Stacks:

```text
forecasting-data-product-foundation
forecasting-data-product-rds
forecasting-data-product-ecs
```

---

## Ejecutar Streamlit localmente en SageMaker

```bash
export RDS_ENDPOINT="PEGAR_ENDPOINT_RDS"
export RDS_SECRET_NAME="itam/rds/forecasting/credentials"
export AWS_REGION="us-east-1"
export GOLD_DATABASE="forecasting_gold"

uv run streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

Abrir con proxy de SageMaker:

```text
https://<studio-domain>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/
```

---

## Screenshots

### App

#### Resumen ejecutivo

<img width="1919" height="1040" alt="app_resumen" src="https://github.com/user-attachments/assets/bf41319f-09e6-4d62-89b5-09da33ef3658" />

#### Pronóstico por categoría

<img width="1919" height="1199" alt="app_pronostico_categoria" src="https://github.com/user-attachments/assets/d37a3652-d368-4edc-9602-c9cea1785029" />

#### Evaluación

<img width="1910" height="1097" alt="app_evaluacion" src="https://github.com/user-attachments/assets/01257e51-c44f-4c50-a2a5-f2708e79ebeb" />

#### Feedback

<img width="1919" height="1015" alt="app_feedback" src="https://github.com/user-attachments/assets/85824e50-9470-4c3c-b5ae-a7a66a9e4744" />

#### Productos marcados

<img width="1919" height="1055" alt="app_productos_marcados" src="https://github.com/user-attachments/assets/a655aa21-cd7e-40eb-9c10-ada6806a7461" />

---

### Evidencia AWS

#### CloudFormation

<img width="1677" height="32" alt="cloudformation_foundation" src="https://github.com/user-attachments/assets/6c94bc8d-1de5-4ac5-88b9-d8227baeef7a" />

<img width="1670" height="31" alt="cloudformation_rds" src="https://github.com/user-attachments/assets/1d1827fe-a192-4312-a5c5-dbfc51594bd0" />

<img width="1250" height="36" alt="cloudformation_ecs" src="https://github.com/user-attachments/assets/03f5a40e-0caf-469b-a220-61cf881d4867" />

#### S3

<img width="1919" height="588" alt="s3_bucket" src="https://github.com/user-attachments/assets/87099898-446d-453c-b56f-709ce99063da" />

#### Glue

<img width="1529" height="107" alt="glue_databases" src="https://github.com/user-attachments/assets/8b477a0a-70bc-49b7-881d-dd2a4056b464" />

#### ECR

<img width="1919" height="951" alt="ecr_image" src="https://github.com/user-attachments/assets/3cbf4fdb-ed0b-47fd-b813-42f9ceac56d6" />

#### ECS

<img width="1908" height="689" alt="ecs_service_running" src="https://github.com/user-attachments/assets/42c53402-f6d6-42cb-9052-3a79f896da54" />

#### RDS

<img width="1919" height="752" alt="rds_available" src="https://github.com/user-attachments/assets/9ca93eb4-2d1e-44f4-8e02-73e3b1e4b263" />

---

