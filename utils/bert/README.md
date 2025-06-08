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
from utils.bert import calculate_semantic_similarity, warm_up_endpoint
```

## 🔥 Warm-up del Endpoint (IMPORTANTE)

Para procesamiento en lote o múltiples llamadas, es **altamente recomendado** calentar el endpoint primero:

```python
from utils.bert import warm_up_endpoint, calculate_semantic_similarity

# Calentar el endpoint UNA VEZ antes de procesar
if warm_up_endpoint():
    print("✅ Endpoint listo para procesar")
    # Ahora puedes hacer múltiples llamadas sin esperas
    results = calculate_semantic_similarity(terms_a, terms_b)
else:
    print("❌ Error al inicializar el endpoint")
```

### ¿Por qué es importante?
- El endpoint de HuggingFace puede estar "dormido" si no se ha usado recientemente
- Sin warm-up, la primera llamada puede tardar 30-60 segundos
- Con warm-up, todas las llamadas son rápidas (~2-3 segundos)

## 🔧 Uso Principal

### Función Principal: `calculate_semantic_similarity()`
La función principal para calcular similaridad semántica.

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
from utils.bert import calculate_semantic_similarity

# Comparar dos términos médicos
result = calculate_semantic_similarity("diabetes", "high blood sugar")

print(result)
# {'diabetes': {'high blood sugar': 0.872}}

score = result['diabetes']['high blood sugar']
print(f"Similaridad: {score:.3f}")
```

### Ejemplo 2: Procesamiento en lote con warm-up
```python
from utils.bert import warm_up_endpoint, calculate_semantic_similarity

# IMPORTANTE: Calentar endpoint antes de procesamiento masivo
if not warm_up_endpoint():
    print("⚠️  Advertencia: El endpoint podría estar lento")

# Procesar múltiples casos
cases = [
    ("fever", ["flu", "covid-19", "cold"]),
    ("chest pain", ["heart attack", "anxiety", "pneumonia"]),
    ("headache", ["migraine", "tension", "tumor"])
]

for symptom, conditions in cases:
    result = calculate_semantic_similarity(symptom, conditions)
    print(f"\n{symptom}:")
    for condition, score in result[symptom].items():
        if score is not None:
            print(f"  → {condition}: {score:.3f}")
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
from utils.bert import calculate_semantic_similarity

# Calcular similaridades
result = calculate_semantic_similarity(
    "acute respiratory distress",
    ["ARDS", "pneumonia", "broken leg"]
)

# Visualizar resultados
for term_a, comparisons in result.items():
    print(f"\nComparaciones para '{term_a}':")
    for term_b, score in comparisons.items():
        if score is not None:
            # Crear barra visual
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  vs '{term_b}': {score:.3f} [{bar}]")
        else:
            print(f"  vs '{term_b}': Error")
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

- **Warm-up automático**: Prepara el endpoint para procesamiento rápido
- **Comparación cruzada**: Muchos-a-muchos automáticamente
- **Manejo de errores**: Devuelve `None` si falla el cálculo
- **Optimización**: Reutiliza embeddings para términos repetidos
- **Logs limpios**: Mensajes concisos con emojis para claridad

## 🎯 Casos de Uso

- ✅ Análisis de similaridad entre síntomas y enfermedades
- ✅ Búsqueda semántica en bases de datos médicas  
- ✅ Validación de mapeos de códigos médicos
- ✅ Agrupación de términos médicos similares
- ✅ Análisis de equivalencias en diferentes idiomas

## 🔴 Manejo de Errores

```python
# Siempre verificar warm-up para procesamiento masivo
if not warm_up_endpoint():
    print("⚠️  El endpoint podría no estar disponible")
    # Considerar implementar lógica de fallback

result = calculate_semantic_similarity("term1", "term2")

score = result["term1"]["term2"]
if score is None:
    print("Error: No se pudo calcular la similaridad")
    # Posibles causas:
    # - Endpoint no calentado (usar warm_up_endpoint())
    # - Problema de conectividad
    # - API key inválida
    # - Límite de rate excedido
else:
    print(f"Similaridad: {score:.3f}")
```

## 🚀 Mejores Prácticas

1. **Siempre usar warm-up** para procesamiento en lote
2. **Agrupar llamadas** cuando sea posible (la función acepta listas)
3. **Verificar None** en los resultados antes de usar scores
4. **Monitorear logs** - los emojis indican el estado:
   - 🔄 = Procesando
   - ✅ = Éxito
   - ⚠️ = Advertencia
   - ❌ = Error
   - ⏳ = Esperando
   - ⏱️ = Timeout
   - 🌐 = Error de red 