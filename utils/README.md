# Utils - Módulos de Utilidades

Este paquete contiene una colección de módulos reutilizables diseñados para ser completamente extraíbles. Cada módulo está pensado para funcionar de forma independiente, pero todos siguen la misma filosofía: configuración automática, API simple y código limpio. La magia está en que puedes tomar cualquier carpeta de utils, copiarla a otro proyecto, y funcionará inmediatamente respetando su `__init__.py`.

## 🎯 Filosofía del Paquete

La carpeta `utils` es especial porque está diseñada como un **módulo Python totalmente portable**. Esto significa que:

1. **Es un paquete Python**: Contiene un archivo `__init__.py` que lo convierte en un módulo importable
2. **Instalación con pip**: Al hacer `pip install -e .`, pip encuentra `utils` y lo registra como un paquete disponible globalmente en tu entorno
3. **Importación desde cualquier lugar**: Una vez instalado, puedes hacer `from utils.bert import calculate_semantic_similarity` desde cualquier script en tu proyecto

## 🔍 ¿Cómo Funciona el `__init__.py`?

El archivo `__init__.py` es la puerta de entrada a un paquete Python. Veamos cómo funciona en nuestro proyecto:

### utils/__init__.py
```python
"""
Utils Package

Este archivo convierte la carpeta utils/ en un paquete Python.
Cuando Python ve un __init__.py, sabe que puede importar desde aquí.
"""

__version__ = '0.1.0'

# Este archivo puede estar vacío, pero su presencia es crucial
```

### utils/bert/__init__.py
```python
"""
BERT similarity service
"""

# Importamos las funciones principales del módulo
from .bert_similarity import *

# Definimos qué se exporta cuando alguien hace "from utils.bert import *"
__all__ = ['calculate_semantic_similarity', 'warm_up_endpoint']
```

### ¿Qué hace cada parte?

1. **El archivo en sí**: Su mera presencia convierte una carpeta en un paquete
2. **Los imports relativos** (`from .bert_similarity import *`): Exponen funcionalidad del módulo interno
3. **La variable `__all__`**: Define qué se exporta públicamente

### Ejemplo práctico de uso:

```python
# Opción 1: Import específico (recomendado)
from utils.bert import calculate_semantic_similarity

# Opción 2: Import del módulo completo
import utils.bert
similarity = utils.bert.calculate_semantic_similarity("diabetes", "sugar")

# Opción 3: Import con alias
from utils.bert import calculate_semantic_similarity as calc_similarity

# Esto es posible gracias al __init__.py
```

## 🏗️ Instalación del Proyecto

### Setup completo
```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Instalar proyecto en modo desarrollo
py -m pip install -e .

# 3. Las dependencias se instalan automáticamente desde pyproject.toml
```

## 📦 Módulos Disponibles

### 🧠 BERT Similarity Service  
**Ubicación**: `utils/bert/`  
**Propósito**: Servicio de análisis semántico que actúa como **juez semántico** en el pipeline de evaluación.

En el contexto del proyecto, BERT (específicamente SapBERT) se utiliza para:
- **Evaluar similitud semántica** entre diagnósticos generados (DDX) y diagnósticos de referencia (GDX)
- **Calcular scores de matching** que determinan qué tan cerca está un modelo de acertar el diagnóstico correcto
- **Proporcionar métricas objetivas** basadas en embeddings médicos especializados

```python
from utils.bert import calculate_semantic_similarity

# Comparación simple
result = calculate_semantic_similarity("heart attack", "myocardial infarction")
# Output: {'heart attack': {'myocardial infarction': 0.95}}

# En el pipeline de evaluación
ddx_list = ["pneumonia", "bronchitis", "covid-19"]
gdx_list = ["bacterial pneumonia", "viral pneumonia"]
scores = calculate_semantic_similarity(ddx_list, gdx_list)
```

**[📖 Ver documentación completa →](bert/README.md)**

---

### 🏥 ICD10 Taxonomy
**Ubicación**: `utils/icd10/`  
**Propósito**: Herramienta para navegar y analizar la taxonomía jerárquica de códigos médicos ICD-10.

Este módulo es fundamental para:
- **Validar códigos médicos** en los datasets
- **Navegar jerarquías** (capítulos → categorías → subcategorías)
- **Buscar diagnósticos** por código o nombre
- **Futuro**: Será la base del health-checker en `data29/`

```python
from utils.icd10 import ICD10Taxonomy

taxonomy = ICD10Taxonomy()
# Buscar COVID-19
covid = taxonomy.get("U07.1")
# Output: {'code': 'U07.1', 'name': 'COVID-19', 'type': 'subcategory'}

# Buscar todos los códigos de fracturas
fractures = taxonomy.match("fracture", type="category")
```

**[📖 Ver documentación completa →](icd10/README.md)**

---

### 🤖 LLM Service (Multi-provider)
**Ubicación**: `utils/llm/`  
**Propósito**: Abstracción unificada para trabajar con múltiples proveedores de LLM (Azure OpenAI, Hugging Face).

En el proyecto se usa para:
- **Generar diagnósticos diferenciales** (DDX) como DxGPT simulado
- **Asignar severidades** a diagnósticos usando LLM-as-judge
- **Procesar en batch** grandes volúmenes de casos médicos

```python
from utils.llm import get_llm, quick_generate

# Automáticamente intenta Azure, luego Hugging Face
llm = get_llm("gpt-4o")  # o "jonsnow", "medgemma", etc.

# Generar diagnósticos con esquema estructurado
schema = {
    "type": "object",
    "properties": {
        "diagnoses": {"type": "array", "items": {"type": "string"}}
    }
}
response = llm.generate(prompt, schema=schema)

# Una línea para casos simples
response = quick_generate("Lista 5 síntomas de diabetes")
```

**[📖 Ver documentación completa →](llm/README.md)**

## 🎯 Portabilidad Total

El verdadero poder de este paquete está en su **portabilidad completa**. Puedes:

1. **Copiar cualquier subcarpeta** (bert/, icd10/, llm/) a otro proyecto
2. **Instalar las dependencias** listadas en su README
3. **Usarlo inmediatamente** - cada módulo es autocontenido

### Ejemplo: Extrayendo el módulo BERT

```bash
# En tu nuevo proyecto
cp -r /path/to/utils/bert ./mi_proyecto/bert

# Instalar dependencias
pip install requests numpy python-dotenv

# Usar inmediatamente
from bert import calculate_semantic_similarity
```

## 🚀 Quick Start

### Para análisis semántico (BERT):
```python
from utils.bert import calculate_semantic_similarity

# Comparar términos médicos
similarity = calculate_semantic_similarity("pneumonia", "lung infection")
score = similarity["pneumonia"]["lung infection"]
print(f"Similaridad: {score:.3f}")  # ~0.87
```

### Para trabajar con ICD-10:
```python
from utils.icd10 import ICD10Taxonomy

taxonomy = ICD10Taxonomy()
# Buscar por código o nombre
covid = taxonomy.get("U07.1")  # o taxonomy.get("COVID-19")
print(f"{covid['code']}: {covid['name']}")

# Navegar jerarquía
children = taxonomy.children("J00-J99", type="category")
print(f"Enfermedades respiratorias: {len(children)} categorías")
```

### Para generar con LLMs:
```python
from utils.llm import get_llm

# Auto-detecta el mejor proveedor disponible
llm = get_llm("gpt-4o")  # Intenta Azure, luego HuggingFace

# Con template para reutilizar
analyzer = llm.template(
    "Analiza síntomas: {symptoms}\nPosibles diagnósticos:",
    temperature=0.3
)

result = analyzer(symptoms="Fiebre, tos seca, fatiga")
```

## 📁 Estructura del Paquete

```
utils/
├── __init__.py                 # Convierte utils/ en paquete Python
├── README.md                   # Este archivo
│
├── bert/                       # Análisis semántico
│   ├── __init__.py            # Exporta: calculate_semantic_similarity, warm_up_endpoint
│   ├── bert_similarity.py     # Implementación con SapBERT
│   └── README.md              # Documentación detallada
│
├── icd10/                      # Taxonomía médica
│   ├── __init__.py            # Exporta: ICD10Taxonomy
│   ├── taxonomy.py            # API para navegar ICD-10
│   ├── data/                  # JSON con taxonomía completa
│   │   └── icd10-taxonomy-complete.json
│   └── README.md              # Documentación y ejemplos
│
└── llm/                        # Abstracción multi-proveedor
    ├── __init__.py            # Exporta: get_llm, Azure, HuggingLLM, etc.
    ├── base.py                # Clase base para todos los LLMs
    ├── azure.py               # Implementación Azure OpenAI
    ├── hugging.py             # Implementación Hugging Face
    └── README.md              # Guía completa de uso
```

## 🛠️ Desarrollo

### Agregar un nuevo módulo:

1. **Crear estructura**:
   ```bash
   mkdir utils/mi_modulo
   touch utils/mi_modulo/__init__.py
   touch utils/mi_modulo/mi_modulo.py
   touch utils/mi_modulo/README.md
   ```

2. **Implementar con API clara**:
   ```python
   # mi_modulo.py
   def funcion_principal(param1, param2):
       """Una función, una responsabilidad."""
       return resultado
   ```

3. **Configurar exports en __init__.py**:
   ```python
   from .mi_modulo import funcion_principal
   __all__ = ['funcion_principal']
   ```

4. **Documentar con ejemplos reales** en README.md

### Principios de diseño:

- **Una responsabilidad** por módulo
- **Configuración automática** desde variables de entorno
- **Cero configuración** para el usuario final
- **Ejemplos que funcionan** copy-paste
- **Totalmente portable** - cada módulo es independiente 