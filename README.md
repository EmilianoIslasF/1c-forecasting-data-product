# Forecasting Data Product — 1C Company

MVP end-to-end para planeación de demanda mensual en retail usando arquitectura Medallion en AWS, Machine Learning batch, una aplicación Streamlit pública y una base operacional en PostgreSQL/RDS para capturar feedback de negocio.

---

## 1. URL pública de la aplicación

**Aplicación Streamlit pública:**

```text
http://forecasting-data-product-alb-1021855290.us-east-1.elb.amazonaws.com/
```

La aplicación permite:

- Consultar un resumen ejecutivo del modelo y de la demanda esperada.
- Explorar pronósticos agregados por categoría, tienda y producto.
- Revisar evaluación del modelo sobre el último mes histórico.
- Descargar predicciones filtradas en CSV.
- Capturar feedback de negocio.
- Marcar productos con problemas para revisión operativa.

---

## 2. Problema de negocio

El caso está basado en datos históricos de ventas de 1C Company. El objetivo es construir un producto de datos que ayude a planear inventario y demanda mensual a nivel tienda-producto.

En retail, una mala planeación de demanda puede generar dos problemas opuestos:

1. **Sobrestock:** exceso de inventario, costos de almacenamiento y descuentos forzados.
2. **Stockouts:** falta de productos, pérdida de ventas y mala experiencia para clientes.

Este proyecto busca convertir datos históricos de ventas en un sistema operativo para consultar predicciones, evaluar el modelo y capturar feedback del negocio.

---

## 3. Objetivo del proyecto

Construir un flujo completo de datos y ML en AWS:

```text
Kaggle dataset
→ Bronze ETL
→ Silver ETL
→ Gold ETL
→ ML feature engineering
→ Model training + batch prediction
→ Streamlit dashboard
→ RDS feedback
→ Docker/ECR/ECS public deployment
```

El producto final es una aplicación Streamlit desplegada públicamente en ECS Fargate.

---

## 4. Arquitectura general

La arquitectura sigue un patrón Medallion:

```text
Bronze → Silver → Gold
```

Además, se agrega una rama de Machine Learning desde Silver:

```text
Silver → ML Features → Model Training → Gold Model Outputs → Streamlit
```

### Diagrama de arquitectura

> Insertar imagen aquí.

```markdown
![Architecture Diagram](docs/architecture.png)
```

### Flujo general

```text
Kaggle Dataset
        ↓
etl/bronze.py
        ↓
S3 Raw / S3 Bronze
        ↓
Glue DB: forecasting_bronze
        ↓
etl/silver.py
        ↓
S3 Silver
        ↓
Glue DB: forecasting_silver
        ├──→ etl/gold.py
        │       ↓
        │   Glue DB: forecasting_gold
        │
        └──→ ml/build_features.py
                ↓
            Glue DB: forecasting_ml
                ↓
            ml/train_model.py
                ↓
            model.joblib + model_metrics.json
                ↓
            Glue DB: forecasting_gold
                ↓
            Streamlit App
                ↓
            ECS Fargate + ALB + Public URL
```

---

## 5. Servicios de AWS utilizados

| Servicio | Uso |
|---|---|
| Amazon S3 | Data lake para raw, bronze, silver, gold, ML features y artefactos |
| AWS Glue Data Catalog | Catálogo de tablas para Bronze, Silver, Gold y ML |
| Amazon Athena | Consulta SQL sobre tablas registradas en Glue |
| Amazon RDS PostgreSQL | Base operacional para feedback y productos marcados |
| AWS Secrets Manager | Administración de credenciales de RDS |
| Amazon ECR | Repositorio de imagen Docker de Streamlit |
| Amazon ECS Fargate | Ejecución serverless del contenedor de Streamlit |
| Application Load Balancer | Exposición pública de la app |
| AWS CloudFormation | Infraestructura como código |
| CloudWatch Logs | Logs del servicio ECS |

---

## 6. Estructura del repositorio

```text
.
├── Dockerfile
├── README.md
├── app
│   └── streamlit_app.py
├── artifacts
│   └── models
│       ├── model.joblib
│       └── model_metrics.json
├── data
│   └── raw
│       ├── item_categories.csv
│       ├── items.csv
│       ├── sales_train.csv
│       ├── sample_submission.csv
│       ├── shops.csv
│       └── test.csv
├── docs
│   └── repo-tree.txt
├── etl
│   ├── bronze.py
│   ├── gold.py
│   └── silver.py
├── infra
│   ├── ecs-streamlit.yaml
│   ├── forecasting-data-product-foundation.yaml
│   └── rds-forecasting.yaml
├── main.py
├── ml
│   ├── build_features.py
│   └── train_model.py
├── postgres
│   └── create_tables.py
├── pyproject.toml
└── uv.lock
```

---

## 7. Dataset

Fuente de datos:

```text
Kaggle — Predict Future Sales / English converted dataset
```

Archivos principales:

| Archivo | Descripción |
|---|---|
| `sales_train.csv` | Ventas históricas diarias |
| `items.csv` | Catálogo de productos |
| `item_categories.csv` | Catálogo de categorías |
| `shops.csv` | Catálogo de tiendas |
| `test.csv` | Combinaciones tienda-producto para predicción |
| `sample_submission.csv` | Formato de submission |

Los datos crudos no se versionan en Git. Se descargan mediante `etl/bronze.py` y se guardan en S3.

---

## 8. Infraestructura con CloudFormation

### 8.1 Foundation stack

Template:

```text
infra/forecasting-data-product-foundation.yaml
```

Crea o configura:

- Bucket S3: `forecasting-data-product`
- Glue databases:
  - `forecasting_bronze`
  - `forecasting_silver`
  - `forecasting_gold`

Stack sugerido:

```text
forecasting-data-product-foundation
```

---

### 8.2 RDS stack

Template:

```text
infra/rds-forecasting.yaml
```

Crea:

- RDS PostgreSQL
- Secrets Manager secret
- Security Group
- Subnet Group

Secret utilizado:

```text
itam/rds/forecasting/credentials
```

Stack sugerido:

```text
forecasting-data-product-rds
```

---

### 8.3 ECS / Streamlit stack

Template:

```text
infra/ecs-streamlit.yaml
```

Crea:

- ECS Cluster
- ECS Service
- ECS Task Definition
- Application Load Balancer
- Target Group
- Security Groups
- IAM Roles
- CloudWatch Log Group

Stack sugerido:

```text
forecasting-data-product-ecs
```

---

## 9. Pipeline Medallion

---

### 9.1 Bronze ETL

Script:

```text
etl/bronze.py
```

Responsabilidades:

- Descarga dataset desde Kaggle.
- Guarda CSVs localmente en `data/raw`.
- Sube CSVs crudos a S3 raw.
- Convierte CSVs a Parquet.
- Escribe tablas Bronze en S3.
- Registra metadata en Glue.

Ubicaciones S3:

```text
s3://forecasting-data-product/forecasting/raw/
s3://forecasting-data-product/forecasting/bronze/
```

Glue database:

```text
forecasting_bronze
```

Tablas creadas:

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
uv run python etl/bronze.py \
  --bucket forecasting-data-product
```

Si los CSVs ya existen localmente y no se quiere descargar otra vez:

```bash
uv run python etl/bronze.py \
  --bucket forecasting-data-product \
  --data-dir data/raw \
  --skip-download
```

---

### 9.2 Silver ETL

Script:

```text
etl/silver.py
```

Responsabilidades:

- Lee tablas Bronze desde Glue/S3.
- Limpia datos.
- Agrega ventas diarias a ventas mensuales.
- Enriquece ventas con catálogo de productos y tiendas.
- Construye input de inferencia.

Ubicación S3:

```text
s3://forecasting-data-product/forecasting/silver/
```

Glue database:

```text
forecasting_silver
```

Tablas creadas:

```text
item_catalog
shop_catalog
sales_monthly
sales_monthly_enriched
forecast_input
```

Ejecución:

```bash
uv run python etl/silver.py \
  --bucket forecasting-data-product
```

---

### 9.3 Gold ETL

Script:

```text
etl/gold.py
```

Responsabilidades:

- Construye tablas listas para consumo analítico.
- Genera KPIs por categoría, tienda y producto.
- Genera baseline naive.
- Evalúa baseline.

Ubicación S3:

```text
s3://forecasting-data-product/forecasting/gold/
```

Glue database:

```text
forecasting_gold
```

Tablas creadas por Gold ETL:

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
uv run python etl/gold.py \
  --bucket forecasting-data-product
```

---

## 10. Machine Learning pipeline

El modelo no se ejecuta dentro de Streamlit. La inferencia se ejecuta en batch y los resultados se escriben en Gold.

Esto reduce latencia, evita recalcular predicciones en la app y permite que Streamlit solo consuma tablas ya preparadas.

---

### 10.1 Feature engineering

Script:

```text
ml/build_features.py
```

Responsabilidades:

- Lee datos limpios desde `forecasting_silver`.
- Construye una matriz tienda-producto-mes.
- Incluye meses sin venta como ceros.
- Genera variables temporales y lags.
- Divide datos en train, validation e inference.

Glue database:

```text
forecasting_ml
```

Tablas creadas:

```text
train_features
validation_features
inference_features
```

Ejecución:

```bash
uv run python ml/build_features.py \
  --bucket forecasting-data-product
```

---

### 10.2 Entrenamiento y predicción batch

Script:

```text
ml/train_model.py
```

Modelo final:

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

Target:

```text
item_cnt_month
```

El target se recorta al rango:

```text
[0, 20]
```

Artefactos generados:

```text
artifacts/models/model.joblib
artifacts/models/model_metrics.json
```

Artefactos en S3:

```text
s3://forecasting-data-product/forecasting/artifacts/models/model.joblib
s3://forecasting-data-product/forecasting/artifacts/models/model_metrics.json
```

Tablas Gold creadas por el modelo:

```text
model_forecast_next_month
model_evaluation
model_metrics_global
model_metrics_by_category
```

Ejecución:

```bash
uv run python ml/train_model.py \
  --bucket forecasting-data-product
```

---

## 11. Métricas del modelo

Métricas sobre el último mes histórico usado como validación:

| Métrica | Valor |
|---|---:|
| Filas de evaluación | 238,172 |
| MAE modelo | 0.346 |
| RMSE modelo | 0.973 |
| R² modelo | 0.267 |
| Demanda real | 61,583 |
| Demanda predicha | 62,792 |

Interpretación:

- El modelo predice demanda mensual a nivel tienda-producto.
- En este problema hay muchas combinaciones con demanda cero o muy baja.
- Por ello, es normal encontrar predicciones menores a 1 unidad para combinaciones tienda-producto de baja rotación.
- Para decisiones de negocio, las vistas agregadas por categoría, tienda o producto son más útiles que revisar únicamente filas granulares.

---

## 12. Aplicación Streamlit

Archivo principal:

```text
app/streamlit_app.py
```

La app lee resultados desde:

```text
forecasting_gold
```

Tablas principales consumidas:

```text
category_monthly
product_kpis
model_forecast_next_month
model_evaluation
model_metrics_global
model_metrics_by_category
```

### Páginas de la app

#### 1. Resumen ejecutivo

Incluye:

- Descripción del producto.
- Hallazgos principales.
- Métricas del modelo.
- Demanda real vs demanda predicha.
- Demanda esperada por categoría.
- Productos históricamente relevantes.

#### 2. Pronóstico

Incluye:

- Demanda esperada total.
- Promedio por tienda-producto.
- Combinaciones con demanda esperada mayor a 1.
- Vista agregada por categoría.
- Vista agregada por tienda.
- Top productos.
- Detalle granular tienda-producto.

#### 3. Evaluación

Incluye:

- Métricas del modelo final.
- Demanda real vs predicha por categoría.
- Categorías con mayor error agregado.
- Casos individuales con mayor error.
- Comparación técnica contra baseline en un expander.

#### 4. Feedback

Permite capturar feedback de negocio y guardarlo en RDS.

#### 5. Productos marcados

Permite marcar productos problemáticos y guardarlos en RDS.

---

## 13. RDS operacional

RDS PostgreSQL se usa como base operacional de la app. No almacena las tablas analíticas del modelo; esas viven en S3/Glue.

RDS guarda información generada por usuarios:

```text
business_feedback
flagged_products
forecast_jobs
app_metrics
```

### Diagrama ERD

> Insertar imagen aquí.

```markdown
![RDS ERD](docs/erd-rds.png)
```

### Tablas

#### `business_feedback`

| Columna | Descripción |
|---|---|
| `feedback_id` | PK |
| `created_at` | Timestamp |
| `created_by` | Usuario |
| `shop_id` | Referencia lógica a tienda |
| `item_id` | Referencia lógica a producto |
| `category_id` | Referencia lógica a categoría |
| `forecast_month` | Mes del forecast |
| `severity` | Severidad |
| `status` | Estado |
| `feedback_text` | Comentario de negocio |

#### `flagged_products`

| Columna | Descripción |
|---|---|
| `flag_id` | PK |
| `created_at` | Timestamp |
| `created_by` | Usuario |
| `shop_id` | Referencia lógica a tienda |
| `item_id` | Referencia lógica a producto |
| `category_id` | Referencia lógica a categoría |
| `reason` | Razón del flag |
| `priority` | Prioridad |
| `status` | Estado |
| `notes` | Notas |

#### `forecast_jobs`

Tabla para registrar ejecuciones o solicitudes operativas.

#### `app_metrics`

Tabla para registrar métricas operativas de la aplicación.

Nota: `shop_id`, `item_id` y `category_id` funcionan como claves de negocio y referencias lógicas hacia entidades analíticas en S3/Glue. No son foreign keys físicas dentro de RDS en este MVP.

---

## 14. Despliegue

La app se empaqueta como imagen Docker, se sube a ECR y se ejecuta en ECS Fargate detrás de un Application Load Balancer.

### Flujo de despliegue

```text
Streamlit App + Dockerfile
→ Docker image
→ ECR
→ ECS Fargate
→ Application Load Balancer
→ Public URL
```

### Build local/SageMaker

```bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPOSITORY="forecasting-data-product-streamlit"
export IMAGE_TAG="streamlit-clean-$(date +%Y%m%d%H%M%S)"
export IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
```

Login:

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login \
      --username AWS \
      --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

Build en SageMaker:

```bash
docker build --network sagemaker -t $ECR_REPOSITORY:$IMAGE_TAG .
```

Push:

```bash
docker tag $ECR_REPOSITORY:$IMAGE_TAG $IMAGE_URI
docker push $IMAGE_URI
```

Después se actualiza CloudFormation cambiando el parámetro:

```text
ImageUri = nuevo IMAGE_URI
```

---

## 15. Cómo correr localmente en SageMaker

### Variables de entorno

```bash
export RDS_ENDPOINT="PEGAR_ENDPOINT_RDS"
export RDS_SECRET_NAME="itam/rds/forecasting/credentials"
export AWS_REGION="us-east-1"
export GOLD_DATABASE="forecasting_gold"
```

### Ejecutar Streamlit

```bash
uv run streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

En SageMaker, abrir vía proxy:

```text
https://<studio-domain>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/
```

---

## 16. Validaciones útiles

### Glue tables

```bash
aws glue get-tables \
  --database-name forecasting_gold \
  --query "TableList[].Name" \
  --output table
```

### Modelo en S3

```bash
aws s3 ls s3://forecasting-data-product/forecasting/artifacts/models/
```

### ECS service

```bash
aws ecs describe-services \
  --cluster forecasting-data-product-cluster \
  --services forecasting-data-product-streamlit-service \
  --region us-east-1 \
  --query "services[0].{Status:status,Desired:desiredCount,Running:runningCount,Pending:pendingCount}" \
  --output table
```

### RDS feedback

```bash
uv run python - <<'PY'
import os
import json
import boto3
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

host = os.environ["RDS_ENDPOINT"]
secret_name = os.getenv("RDS_SECRET_NAME", "itam/rds/forecasting/credentials")
region = os.getenv("AWS_REGION", "us-east-1")

client = boto3.client("secretsmanager", region_name=region)
secret = client.get_secret_value(SecretId=secret_name)
creds = json.loads(secret["SecretString"])

user = quote_plus(creds["username"])
password = quote_plus(creds["password"])
dbname = creds["dbname"]
port = creds.get("port", "5432")

url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(url, pool_pre_ping=True)

with engine.connect() as conn:
    feedback = pd.read_sql(
        text("SELECT * FROM business_feedback ORDER BY created_at DESC LIMIT 5"),
        conn,
    )
    flagged = pd.read_sql(
        text("SELECT * FROM flagged_products ORDER BY created_at DESC LIMIT 5"),
        conn,
    )

print("\nbusiness_feedback")
print(feedback)

print("\nflagged_products")
print(flagged)
PY
```

---

## 17. Evidencias / Screenshots

> Agregar screenshots en `docs/screenshots/`.

Sugeridos:

### App

- `docs/screenshots/app_resumen.png`
- `docs/screenshots/app_pronostico_categoria.png`
- `docs/screenshots/app_pronostico_producto.png`
- `docs/screenshots/app_evaluacion.png`
- `docs/screenshots/app_feedback.png`
- `docs/screenshots/app_productos_marcados.png`

### AWS

- `docs/screenshots/cloudformation_foundation.png`
- `docs/screenshots/cloudformation_rds.png`
- `docs/screenshots/cloudformation_ecs.png`
- `docs/screenshots/s3_bucket.png`
- `docs/screenshots/glue_databases.png`
- `docs/screenshots/ecr_image.png`
- `docs/screenshots/ecs_service_running.png`
- `docs/screenshots/target_group_healthy.png`
- `docs/screenshots/rds_available.png`

---

## 18. Limitaciones

- El modelo predice demanda esperada mensual, no órdenes de inventario directamente.
- Muchas combinaciones tienda-producto tienen demanda muy baja o cero, por lo que valores menores a 1 son normales.
- RDS se usa solo para feedback operacional, no como warehouse analítico.
- No se implementó una réplica de RDS porque la carga esperada del MVP es baja.
- El modelo se ejecuta batch/offline; la app no recalcula predicciones en tiempo real.
- No se implementó autenticación de usuarios para la app pública.
- No se implementaron intervalos de confianza.

---

## 19. Próximos pasos

- Agregar autenticación a la app.
- Automatizar el pipeline con SageMaker Pipelines o Step Functions.
- Agregar monitoreo de drift de datos y performance del modelo.
- Incluir intervalos de confianza en predicciones.
- Agregar lógica de recomendación de inventario basada en reglas de negocio.
- Incorporar promociones, precios y eventos externos.
- Agregar CI/CD para build y despliegue automático.
- Mejorar esquema operacional de RDS con usuarios, auditoría y relaciones normalizadas.

---

## 20. Costos y apagado

Para evitar costos innecesarios después de la revisión:

- Detener o eliminar stack ECS.
- Eliminar o pausar RDS si ya no se necesita.
- Revisar imágenes en ECR.
- Revisar objetos grandes en S3.
- Mantener solo evidencia necesaria para entrega.

Stacks principales:

```text
forecasting-data-product-foundation
forecasting-data-product-rds
forecasting-data-product-ecs
```

---

## 21. Uso de herramientas de IA

Durante el desarrollo se utilizó asistencia de IA para acelerar:

- Diseño de arquitectura.
- Generación inicial de scripts ETL.
- Debugging de errores de AWS, Docker y Glue.
- Redacción de documentación.
- Iteración del dashboard Streamlit.
- Organización del reporte técnico.

Todas las decisiones finales de arquitectura, validación, ejecución y despliegue fueron verificadas mediante pruebas en AWS, outputs de terminal, Glue/Athena, RDS y la aplicación desplegada.

---

## 22. Estado final del proyecto

El proyecto quedó desplegado como un producto de datos funcional:

```text
✅ ETL Bronze / Silver / Gold
✅ Feature engineering ML
✅ Entrenamiento batch
✅ Predicciones en Gold
✅ Dashboard Streamlit
✅ Feedback en RDS
✅ Docker image en ECR
✅ App corriendo en ECS Fargate
✅ URL pública mediante ALB
```
