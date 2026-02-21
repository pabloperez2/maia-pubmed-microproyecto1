Microproyecto 1

Sequential Sentence Classification in Medical Abstracts


Entrega 2 – Entrenamiento con Transformers y Tracking con MLflow

1. Contexto del Proyecto

En la investigación médica, los artículos científicos generalmente se presentan en resúmenes estructurados, donde cada oración cumple un rol específico dentro del razonamiento del estudio: Background, Objective, Methods, Results o Conclusions.

Sin embargo, en muchos repositorios estos abstracts se encuentran en texto plano, sin segmentación explícita. Esto dificulta:

La revisión rápida de información científica.

La búsqueda focalizada dentro de un abstract.

El desarrollo de herramientas automáticas de apoyo a investigadores.

Este proyecto aborda el problema como una tarea de clasificación supervisada a nivel de oración, utilizando modelos basados en transformers para identificar automáticamente el rol retórico de cada oración en un abstract médico.

La pregunta que guía el proyecto es:

¿Puede un sistema basado en procesamiento de lenguaje natural clasificar automáticamente las oraciones de resúmenes médicos en categorías retóricas que faciliten su análisis y comprensión?

2. Dataset

Se utiliza el dataset PubMed RCT 20k, disponible públicamente en Hugging Face:

https://huggingface.co/datasets/armanc/pubmed-rct20k

Características relevantes:

Total registros: 235,892

Entrenamiento: 176,642

Validación: 29,672

Prueba: 29,578

Clases:

Background

Objective

Methods

Results

Conclusions

Cada registro corresponde a una oración individual asociada a un abstract, identificado por abstract_id y su posición sentence_id.

3. Modelo Utilizado

En el notebook microproyecto3.ipynb se utiliza explícitamente el modelo:

allenai/scibert_scivocab_uncased

Este modelo es cargado mediante:

AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

El modelo se entrena para clasificación multiclase con 5 etiquetas correspondientes a las categorías retóricas del dataset.

4. Variantes Entrenadas

Se entrenan dos variantes del modelo SciBERT:

Variante 1 – Baseline (sin downsampling)

learning_rate = 2e-5

batch_size = 32

num_epochs = 3

max_length = 128

use_downsampling = False

Variante 2 – Con Downsampling

Mismos hiperparámetros, pero:

use_downsampling = True

Esto permite evaluar el impacto del desbalance de clases observado en el dataset.

5. Arquitectura del Proyecto

El entrenamiento y el tracking están desacoplados.

Componentes

Entrenamiento: Google Colab (GPU)

Tracking de experimentos: MLflow Server en AWS EC2

Backend store: file-based

Versionamiento: Git

Gestión de datos: DVC + S3 (según Entrega 1)

Flujo

Colab (SciBERT training)
→ MLflow Tracking Server (EC2)
→ UI Web para visualización de runs

La instancia EC2 funciona únicamente como tracking server y no realiza entrenamiento.

6. Configuración del MLflow Tracking Server

El servidor MLflow se ejecuta en una instancia EC2 Ubuntu t3.micro.

Comando utilizado para levantar el servidor:

nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*' \
  --backend-store-uri file:/home/ubuntu/dvc-proj/mlruns \
  --default-artifact-root file:/home/ubuntu/dvc-proj/mlruns \
  > ~/mlflow_5000.log 2>&1 &

echo $! | tee ~/mlflow_5000.pid

URL del tracking server:

http://13.218.110.62:5000


7. Configuración en Google Colab

Antes de entrenar, cada integrante debe configurar el tracking URI (Ejecutar este primer fragmento en colab):

import os
import mlflow

MLFLOW_TRACKING_URI = "http://54.205.108.123:5000"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("pubmed-rct-classification")

print("Tracking URI:", mlflow.get_tracking_uri())

El nombre del experimento utilizado en el notebook es:

pubmed-rct-classification

8. Métricas Registradas

Durante el entrenamiento se registran en MLflow:

train_loss

train_runtime

train_samples_per_second

val_macro_f1

val_micro_f1

test_macro_f1

test_micro_f1

Adicionalmente se registran como artifacts:

classification_report.txt

confusion_matrix.png

Modelo entrenado

9. Reproducibilidad

El proyecto integra:

Versionamiento de código con Git

Versionamiento de datos con DVC

Almacenamiento remoto en S3

Tracking centralizado de experimentos con MLflow

Esto permite:

Comparación estructurada entre variantes

Trazabilidad de hiperparámetros

Auditoría de métricas

Trabajo colaborativo del equipo