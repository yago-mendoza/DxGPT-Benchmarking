# Utils - Módulos de Utilidades

Colección de módulos de utilidades para análisis de datos médicos, procesamiento de texto y servicios de LLM.

## 🏗️ Instalación del Proyecto

### Setup completo
```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Instalar proyecto en modo desarrollo
py -m pip install -e .

# 3. Instalar dependencias adicionales
pip install python-dotenv requests numpy openai pyyaml
```

## 📦 Módulos Disponibles

### 🤖 Azure LLM Service
**Ubicación**: `utils/llm/`  
**Propósito**: Wrapper elegante para Azure OpenAI API con configuración automática y salida estructurada.

```python
from utils.llm import Azure, quick_generate

# Uso básico con configuración automática
llm = Azure("gpt-4o")
response = llm.generate("Explica qué es la inteligencia artificial")

# Una línea para casos simples
response = quick_generate("Traduce 'Hello' al español")
```

**[📖 Ver documentación completa →](llm/README.md)**

---

### 🧠 BERT Similarity Service  
**Ubicación**: `utils/bert/`  
**Propósito**: Calcular similaridad semántica entre términos médicos usando embeddings SapBERT.

```python
from utils.bert import calculate_semantic_similarity

# Comparación simple
result = calculate_semantic_similarity("heart attack", "myocardial infarction")

# Comparación múltiple
result = calculate_semantic_similarity(
    ["fever", "headache"], 
    ["flu", "migraine", "covid-19"]
)
```

**[📖 Ver documentación completa →](bert/README.md)**

---

### 🏥 ICD10 Taxonomy
**Ubicación**: `utils/icd10/`  
**Propósito**: Trabajar con códigos médicos ICD-10 y su taxonomía jerárquica.

```python
from utils.icd10.taxonomy import ICD10Taxonomy

taxonomy = ICD10Taxonomy()
result = taxonomy.find("S62")  # Buscar por código
result = taxonomy.find("Cholera")  # Buscar por nombre
```

**[📖 Ver documentación completa →](icd10/README.md)**

## 🎯 Filosofía de Diseño

Siguiendo las mejores prácticas del proyecto:

> **"Write the code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live."**

- **Lógica upstream**: Configuración y dependencias centralizadas
- **Imports simples**: Una línea para importar, sin setup complejo
- **Funciones claras**: Una función principal por módulo
- **Documentación zen**: Examples que funcionan de inmediato

## 🚀 Quick Start

### Para LLM con Azure OpenAI:
```python
from utils.llm import quick_generate

# Requiere configuración de .env con Azure OpenAI credentials
response = quick_generate("Resume este texto en 3 puntos")
print(response)
```

### Para similaridad semántica:
```python
from utils.bert import calculate_semantic_similarity

# Requiere configuración de .env con HuggingFace API
similarity = calculate_semantic_similarity("pneumonia", "lung infection")
score = similarity["pneumonia"]["lung infection"]
print(f"Similaridad: {score:.3f}")
```

### Para análisis ICD-10:
```python
from utils.icd10.taxonomy import ICD10Taxonomy

taxonomy = ICD10Taxonomy()
covid_info = taxonomy.find("U07.1")  # COVID-19
print(f"Encontrado: {covid_info['current']['name']}")
```

## 📁 Estructura

```
utils/
├── README.md                    # Este archivo
├── llm/
│   ├── README.md               # Documentación Azure LLM
│   ├── azure.py                # Módulo principal
│   └── __init__.py             # Exports principales
├── bert/
│   ├── README.md               # Documentación BERT
│   ├── bert_similarity.py      # Módulo principal
│   └── __init__.py             # Exports principales
├── icd10/
│   ├── README.md               # Documentación ICD10
│   ├── taxonomy.py             # Módulo principal
│   └── data/                   # Datos ICD10
└── __init__.py                 # Package initialization
```

## 🛠️ Desarrollo

### Agregar nuevo módulo:
1. Crear directorio en `utils/`
2. Implementar con función principal clara
3. Crear README con ejemplos
4. Seguir el patrón: importar → usar (sin setup)

### Principios:
- **Una responsabilidad** por módulo
- **Una función principal** exportada
- **Configuración upstream** (no en el código de uso)
- **Ejemplos funcionales** en README 