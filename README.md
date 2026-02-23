# Microproyecto 1: Sequential Sentence Classification in Medical Abstracts

## Entrega 2 – Entrenamiento con Transformers y Tracking con MLflow 🚀

### 1. Contexto del Proyecto
En la investigación médica, los artículos científicos suelen presentarse en resúmenes estructurados donde cada oración cumple un rol específico: **Background, Objective, Methods, Results o Conclusions**.

Sin embargo, muchos repositorios almacenan estos abstracts en texto plano sin segmentación explícita, lo que dificulta:
* La revisión rápida de información científica crítica.
* La búsqueda focalizada dentro de un abstract.
* El desarrollo de herramientas automáticas de apoyo a investigadores.

Este proyecto aborda el problema como una tarea de **clasificación supervisada a nivel de oración**, utilizando modelos basados en **Transformers** para identificar automáticamente el rol retórico de cada segmento en un abstract médico.

**Pregunta de investigación:**
> ¿Puede un sistema basado en procesamiento de lenguaje natural (NLP) clasificar automáticamente las oraciones de resúmenes médicos en categorías retóricas que faciliten su análisis y comprensión?

---

### 2. Dataset
Se utiliza el dataset **PubMed RCT 20k**, un estándar en la industria disponible en Hugging Face:
🔗 [armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k)

**Características relevantes:**
* **Total de registros:** 235,892 oraciones.
* **Partición:**
    * Entrenamiento: 176,642
    * Validación: 29,672
    * Prueba: 29,578
* **Clases (Labels):** `Background`, `Objective`, `Methods`, `Results`, `Conclusions`.

Cada registro incluye un `abstract_id` y un `sentence_id` para preservar el contexto secuencial de la publicación original.

---

### 3. Modelos utilizados

En los notebooks del proyecto (microproyecto3.ipynb, microproyecto3_pablo.ipynb, microproyecto3_nata_scibert-comparison.ipynb) se probaron tres modelos de lenguaje preentrenados para clasificación de secuencias:

- **allenai/scibert_scivocab_uncased** — SciBERT con vocabulario científico, sin distinguir mayúsculas/minúsculas.
- **allenai/scibert_scivocab_cased** — SciBERT con vocabulario científico, preservando capitalización.
- **microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext** — PubMedBERT, preentrenado en abstracts y full-text de PubMed.

Cada modelo se carga mediante:

```python
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
```

Los modelos se entrenan para clasificación multiclase con 5 etiquetas correspondientes a las categorías retóricas del dataset (background, objective, methods, results, conclusions).

### 4. Variantes Entrenadas

Para cada modelo (SciBERT y PubMedBERT) se entrenan dos variantes:

**Variante 1 – Baseline (sin downsampling)**

- learning_rate = 2e-5  
- batch_size = 16 o 32  
- num_epochs = 3  
- max_length = 128  
- use_downsampling = False  

**Variante 2 – Con downsampling**

- Mismos hiperparámetros, pero use_downsampling = True (dataset balanceado por clase).

Esto permite evaluar el impacto del desbalance de clases y comparar el rendimiento entre modelos científicos (SciBERT) y biomédicos (PubMedBERT).

---

### 5. Arquitectura del Proyecto
El flujo de trabajo sigue principios de **MLOps**

**Componentes:**
* **Entrenamiento:** Google Colab (utilizando aceleración por GPU).
* **Tracking de Experimentos:** Servidor MLflow desplegado en **AWS EC2**.
* **Backend Store:** Sistema de archivos local en la instancia.
* **Versionamiento:** Git para código y **DVC** para datos.
* **Almacenamiento de Datos:** Amazon S3 (según la arquitectura de la Entrega 1).

**Flujo de Trabajo:**
`Google Colab (SciBERT Training)` ➔ `AWS EC2 (MLflow Tracking Server)` ➔ `UI Web (Visualización de Runs)`

> **Nota:** La instancia EC2 funciona exclusivamente como servidor de tracking y almacenamiento de metadatos; no realiza el procesamiento del entrenamiento.

---

### 6. Configuración del MLflow Tracking Server

El servidor MLflow se ejecuta en una instancia EC2 Ubuntu (por ejemplo t3.micro). En la máquina virtual:

```bash
cd ~/dvc-proj
source .venv/bin/activate

```bash
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*' \
  --cors-allowed-origins '*' \
  --backend-store-uri file:/home/ubuntu/dvc-proj/mlruns \
  --default-artifact-root file:/home/ubuntu/dvc-proj/mlruns \
  > ~/mlflow_5000.log 2>&1 &

# Guardar el PID para gestión del proceso
echo $! | tee ~/mlflow_5000.pid

URL del tracking server: http://54.205.108.123:5000
```

### 7. Configuración en Google Colab

Antes de entrenar, cada integrante debe configurar el tracking URI y el experimento (ejecutar en Colab):

```python
import mlflow

# Configuración de conexión remota (Reemplazar VMIP por la IP pública de la instancia)
MLFLOW_TRACKING_URI = "http://54.205.108.123:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("microproyecto-entrega-Nata")

print("Tracking URI:", mlflow.get_tracking_uri())
```

El nombre del experimento donde se registran los runs del equipo es **microproyecto-entrega-Nata**. La URL del tracking server (IP y puerto) puede variar según la instancia EC2; en los notebooks se usa el valor configurado en `MLFLOW_TRACKING_URI`.

### 8. Métricas Registradas

Durante el entrenamiento se registran en MLflow las métricas:

- train_loss, train_runtime, train_samples_per_second  
- val_macro_f1, val_micro_f1  
- test_macro_f1, test_micro_f1  

Y como parámetros: model_name, learning_rate, batch_size, num_epochs, max_length, use_downsampling, train_size.

Adicionalmente se registran como artifacts:

- classification_report.txt  
- confusion_matrix.png  
- Modelo entrenado (carpeta del modelo guardado)

### 9. Reproducibilidad

El proyecto integra:

Versionamiento de código con Git

Versionamiento de datos con DVC

Almacenamiento remoto en S3

### 8. Métricas y Artefactos Registrados 📊
Durante el proceso de entrenamiento y evaluación, se registran automáticamente en el servidor de **MLflow** los siguientes parámetros y archivos para asegurar la trazabilidad del experimento:

* **Métricas de Rendimiento:** `train_loss`, `train_runtime`, `train_samples_per_second`.
* **Métricas de Validación y Test:** `val_macro_f1`, `val_micro_f1`, `test_macro_f1`, `test_micro_f1`.
* **Artefactos (Files):**
    * `classification_report.txt`: Reporte detallado con precisión, recall y puntuación F1 por cada clase.
    * `confusion_matrix.png`: Matriz de confusión visual para identificar errores sistemáticos del modelo.
    * **Modelo Entrenado:** Registro de los pesos y la configuración del Transformer para facilitar su despliegue futuro.

---

### 9. Reproducibilidad y Colaboración
Este proyecto integra herramientas estándar de la industria para garantizar un ciclo de vida de **MLOps** robusto y transparente:

1.  **Git:** Control de versiones exhaustivo del código fuente y notebooks.
2.  **DVC + S3:** Gestión y versionamiento de datasets de gran volumen, permitiendo que cualquier integrante del equipo recupere la versión exacta de los datos utilizados.
3.  **MLflow:** Centralización de resultados, lo que permite la comparación objetiva entre variantes y la auditoría de hiperparámetros.

Esta estructura asegura la **trazabilidad completa** de los experimentos, elimina el problema de "funciona en mi máquina" y facilita el trabajo colaborativo dentro del equipo de investigación.
