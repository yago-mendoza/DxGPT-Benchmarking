# Utils - Módulos de Utilidades

Colección de módulos de utilidades para análisis de datos médicos y procesamiento de texto.

## 🏗️ Instalación del Proyecto

### Setup completo
```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Instalar proyecto en modo desarrollo
py -m pip install -e .

# 3. Instalar dependencias adicionales (si es necesario)
pip install python-dotenv requests numpy
```

## 📦 Módulos Disponibles

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

---

### 🧠 BERT Similarity Service  
**Ubicación**: `utils/services/`  
**Propósito**: Calcular similaridad semántica entre términos médicos usando SapBERT.

```python
from utils.services.bert_similarity import calculate_semantic_similarity

# Comparación simple
result = calculate_semantic_similarity("heart attack", "myocardial infarction")

# Comparación múltiple
result = calculate_semantic_similarity(
    ["fever", "headache"], 
    ["flu", "migraine", "covid-19"]
)
```

**[📖 Ver documentación completa →](services/README.md)**

## 🎯 Filosofía de Diseño

Siguiendo las mejores prácticas del proyecto:

> **"Write the code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live."**

- **Lógica upstream**: Configuración y dependencias centralizadas
- **Imports simples**: Una línea para importar, sin setup complejo
- **Funciones claras**: Una función principal por módulo
- **Documentación zen**: Examples que funcionan de inmediato

## 🚀 Quick Start

### Para análisis ICD-10:
```python
from utils.icd10.taxonomy import ICD10Taxonomy

taxonomy = ICD10Taxonomy()
covid_info = taxonomy.find("U07.1")  # COVID-19
print(f"Encontrado: {covid_info['current']['name']}")
```

### Para similaridad semántica:
```python
from utils.services.bert_similarity import calculate_semantic_similarity

# Requiere configuración de .env con API keys
similarity = calculate_semantic_similarity("pneumonia", "lung infection")
score = similarity["pneumonia"]["lung infection"]
print(f"Similaridad: {score:.3f}")
```

## 📁 Estructura

```
utils/
├── README.md                    # Este archivo
├── icd10/
│   ├── README.md               # Documentación ICD10
│   ├── taxonomy.py             # Módulo principal
│   └── data/                   # Datos ICD10
└── services/
    ├── README.md               # Documentación BERT
    └── bert_similarity.py      # Módulo principal
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