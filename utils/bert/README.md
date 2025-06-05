# BERT Similarity Service

Módulo para calcular similaridad semántica entre términos médicos usando embeddings SapBERT.

## 🚀 Instalación

### 1. Crear entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Instalar proyecto en modo desarrollo
```bash
py -m pip install -e .
```

### 3. Instalar dependencias específicas
```bash
pip install python-dotenv requests numpy
```

## 🔧 Configuración

### Crear archivo `.env`
Crear un archivo `.env` en la raíz del proyecto con:

```env
SAPBERT_API_URL=https://your-endpoint-id.huggingface.cloud
HF_TOKEN=hf_your_token_with_permissions
```

> **Nota**: Necesitas una API key de HuggingFace y acceso a un endpoint SapBERT.

## 📦 Importación

```python
from utils.bert import calculate_semantic_similarity
s```

## 🔧 Uso Principal

### Función Principal: `calculate_semantic_similarity()`
La única función que necesitas para calcular similaridad semántica.

```python
# Comparar dos términos
result = calculate_semantic_similarity("heart attack", "myocardial infarction")

# Comparar un término con múltiples
result = calculate_semantic_similarity("covid-19", ["sars-cov-2", "influenza"])

# Comparar múltiples con múltiples
result = calculate_semantic_similarity(
    ["heart attack", "stroke"], 
    ["myocardial infarction", "cerebrovascular accident"]
)
```

## 📝 Ejemplos Rápidos

### Ejemplo 1: Comparación simple
```python
from utils.services.bert_similarity import calculate_semantic_similarity

# Comparar dos términos médicos
result = calculate_semantic_similarity("diabetes", "high blood sugar")

print(result)
# {'diabetes': {'high blood sugar': 0.872}}

score = result['diabetes']['high blood sugar']
print(f"Similaridad: {score:.3f}")
```

### Ejemplo 2: Uno vs múltiples
```python
# Comparar un síntoma con múltiples condiciones
symptom = "chest pain"
conditions = ["heart attack", "anxiety", "pneumonia"]

result = calculate_semantic_similarity(symptom, conditions)

print("Similaridades:")
for condition, score in result[symptom].items():
    if score is not None:
        print(f"  {condition}: {score:.3f}")
    else:
        print(f"  {condition}: Error")
```

### Ejemplo 3: Matriz completa
```python
# Comparación cruzada completa
symptoms = ["fever", "headache"]
diseases = ["flu", "migraine", "covid-19"]

result = calculate_semantic_similarity(symptoms, diseases)

# Mostrar matriz de similaridad
print("Matriz de Similaridad:")
print("Síntoma → Enfermedad")
for symptom in symptoms:
    print(f"\n{symptom}:")
    for disease in diseases:
        score = result[symptom][disease]
        if score is not None:
            print(f"  → {disease}: {score:.3f}")
        else:
            print(f"  → {disease}: Error")
```

### Ejemplo 4: Análisis con visualización
```python
from utils.services.bert_similarity import (
    calculate_semantic_similarity,
    SimilarityVisualizer
)

# Calcular similaridades
result = calculate_semantic_similarity(
    "acute respiratory distress",
    ["ARDS", "pneumonia", "broken leg"]
)

# Usar visualizador para mejor presentación
visualizer = SimilarityVisualizer()

for term_a, comparisons in result.items():
    print(f"Comparaciones para '{term_a}':")
    for term_b, score in comparisons.items():
        bar = visualizer.create_progress_bar(score)
        level = visualizer.get_similarity_level(score)
        score_str = visualizer.format_score(score)
        
        print(f"  vs '{term_b}': {score_str} {bar} {level}")
```

## 📊 Estructura de Respuesta

La función devuelve un diccionario anidado:

```python
{
    "término_A1": {
        "término_B1": 0.95,  # Score de 0.0 a 1.0
        "término_B2": 0.32,
        "término_B3": None   # Error en cálculo
    },
    "término_A2": {
        "término_B1": 0.28,
        "término_B2": 0.87,
        "término_B3": 0.45
    }
}
```

## 🎯 Interpretación de Scores

| Rango | Nivel | Interpretación |
|-------|-------|----------------|
| 0.90+ | 🟡 MUY ALTA | Términos prácticamente sinónimos |
| 0.75+ | 🟢 ALTA | Fuerte relación semántica |
| 0.50+ | 🟠 MEDIA | Relación moderada |
| 0.30+ | 🔴 BAJA | Relación débil |
| 0.15+ | ⚫ MUY BAJA | Relación mínima |
| 0.00+ | ⚫⚫ BAJÍSIMA | Sin relación aparente |

## ⚡ Características

- **Comparación cruzada**: Muchos-a-muchos automáticamente
- **Manejo de errores**: Devuelve `None` si falla el cálculo
- **Optimización**: Reutiliza embeddings para términos repetidos
- **Visualización**: Herramientas incluidas para mostrar resultados

## 🎯 Casos de Uso

- ✅ Análisis de similaridad entre síntomas y enfermedades
- ✅ Búsqueda semántica en bases de datos médicas  
- ✅ Validación de mapeos de códigos médicos
- ✅ Agrupación de términos médicos similares
- ✅ Análisis de equivalencias en diferentes idiomas

## 🔴 Manejo de Errores

```python
result = calculate_semantic_similarity("term1", "term2")

score = result["term1"]["term2"]
if score is None:
    print("Error: No se pudo calcular la similaridad")
    # Posibles causas:
    # - Problema de conectividad
    # - API key inválida
    # - Endpoint no disponible
else:
    print(f"Similaridad: {score:.3f}")
``` 