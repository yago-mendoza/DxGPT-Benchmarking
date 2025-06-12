# DxGPT Latitude Bench 🏥

Un sistema de evaluación para modelos de diagnóstico médico que integra análisis semántico con BERT y evaluación de severidad mediante LLMs. Este proyecto permite comparar el rendimiento de diferentes modelos de IA en tareas de diagnóstico diferencial, proporcionando métricas detalladas y visualizaciones interactivas.

## 🌟 Características Principales

- **Evaluación Dual**: Análisis semántico (SapBERT) + evaluación de severidad (LLM-as-judge)
- **Pipeline Automatizado**: Desde generación de diagnósticos hasta visualización de resultados
- **Módulos Reutilizables**: Herramientas extraíbles para BERT, ICD-10 y abstracción de LLMs
- **Dashboard Interactivo**: Visualización y comparación de experimentos en tiempo real
- **Multi-proveedor**: Soporte para Azure OpenAI y Hugging Face

## 🚀 Instalación Rápida

### 1. Crear y activar entorno virtual

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.\.venv\Scripts\activate      

# Activar (Linux/Mac)
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
# Para CPU (recomendado para desarrollo)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar proyecto en modo editable
pip install -e .
```

> 💡 **¿Qué es el modo editable?** Con `pip install -e .`, pip crea enlaces simbólicos a tu código en lugar de copiarlo. Esto significa que cualquier cambio que hagas se refleja inmediatamente sin necesidad de reinstalar.

## 🔑 Configuración de Variables de Entorno

El proyecto utiliza variables de entorno para configurar servicios externos. Crea un archivo `.env` en la raíz del proyecto:

```env
# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_API_KEY=tu-api-key-aqui
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# === Hugging Face ===
HF_TOKEN=hf_tu_token_aqui

# SapBERT para análisis semántico
SAPBERT_API_URL=https://tu-endpoint.huggingface.cloud

# Modelos médicos especializados
JONSNOW_ENDPOINT_URL=https://jonsnow-deployment.hf.space
MEDGEMMA_ENDPOINT_URL=https://medgemma-deployment.hf.space
SAKURA_ENDPOINT_URL=https://sakura-deployment.hf.space
OPENBIO_ENDPOINT_URL=https://openbio-deployment.hf.space
```

> 📝 **Nota**: El archivo `.env` es automáticamente cargado por `python-dotenv`. Nunca subas este archivo a control de versiones.

## 🏗️ Estructura del Proyecto

```
dxgpt-latitude-bench-test/
├── .env                    # Variables de entorno (no subir a git)
├── pyproject.toml          # Configuración del proyecto
├── README.md               # Este archivo
│
├── utils/                  # Módulos reutilizables
│   ├── __init__.py        # Hace que utils sea un paquete Python
│   ├── bert/              # Análisis de similitud semántica
│   ├── icd10/             # Herramientas para taxonomía médica
│   └── llm/               # Abstracción para múltiples LLMs
│
├── bench/                  # Sistema de evaluación
│   ├── candidate-prompts/  # Prompts para generar diagnósticos
│   ├── datasets/          # Datasets médicos procesados
│   └── pipeline/          # Pipeline de evaluación
│       ├── run.py         # Script principal
│       ├── config.yaml    # Configuración de experimentos
│       └── results/       # Resultados y visualizaciones
│
├── data29/                # Datos y ETL
│   ├── data-repos/        # Datos crudos y procesados
│   └── health-checker/    # Validador de calidad (futuro)
│
└── tests/                 # Tests unitarios y de integración
```

## 🔧 Uso Básico

### 1. Ejecutar un experimento de evaluación

```bash
cd bench/pipeline
python run.py
```

### 2. Visualizar resultados

```bash
cd bench/pipeline/results/dashboard
python serve_dashboard.py
# Abrir http://localhost:8000 en el navegador
```

### 3. Usar módulos individuales

```python
# Análisis semántico con BERT
from utils.bert import calculate_semantic_similarity
similarity = calculate_semantic_similarity("diabetes", "high blood sugar")

# Trabajar con códigos ICD-10
from utils.icd10 import ICD10Taxonomy
taxonomy = ICD10Taxonomy()
covid_info = taxonomy.find("U07.1")

# Generar con LLMs
from utils.llm import quick_generate
response = quick_generate("Explica qué es la hipertensión")
```

## 📚 Documentación Detallada

- [Utils - Módulos reutilizables](utils/README.md)
  - [BERT - Análisis semántico](utils/bert/README.md)
  - [ICD-10 - Taxonomía médica](utils/icd10/README.md)
  - [LLM - Abstracción multi-proveedor](utils/llm/README.md)
- [Bench - Sistema de evaluación](bench/README.md)
  - [Pipeline - Metodología](bench/pipeline/README.md)
  - [Dashboard - Visualizaciones](bench/pipeline/results/dashboard/README.md)
- [Data29 - Gestión de datos](data29/README.md)
- [Tests - Pruebas del sistema](tests/README.md)

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
