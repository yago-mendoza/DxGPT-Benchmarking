# Pipeline de Evaluación DxGPT - Metodología Detallada 🔬

Este directorio contiene el motor de evaluación que implementa la metodología científica para medir el rendimiento de modelos de IA en diagnóstico médico. El pipeline está diseñado para ser reproducible, extensible y estadísticamente riguroso.

## 🎯 Visión General

El pipeline evalúa modelos de diagnóstico mediante un proceso de 5 etapas que replica cómo un médico abordaría un caso clínico:

1. **Generación de Diagnósticos Diferenciales (DDX)**
2. **Evaluación Semántica** (¿Acertó el diagnóstico?)
3. **Asignación de Severidades** (¿Qué tan graves son?)
4. **Evaluación de Severidad** (¿Estimó bien la gravedad?)
5. **Agregación de Resultados** (Métricas finales)

## 📊 Metodología de Evaluación

### 1. Score Semántico (SapBERT como Juez)

El score semántico mide qué tan cerca está el modelo de identificar el diagnóstico correcto, usando embeddings médicos especializados.

**Principio clave**: El objetivo es que el diagnóstico correcto esté presente, sin importar su posición en la lista.

**Proceso**:
```python
# Para cada caso
DDX = ["Neumonía", "Bronquitis", "COVID-19", "Gripe", "Asma"]
GDX = ["Neumonía bacteriana", "Neumonía viral"]

# SapBERT calcula similitud entre cada par DDX-GDX
similitudes = {
    "Neumonía": {
        "Neumonía bacteriana": 0.92,
        "Neumonía viral": 0.89
    },
    "COVID-19": {
        "Neumonía bacteriana": 0.45,
        "Neumonía viral": 0.67
    }
    # ...
}

# Score final = mejor match encontrado
best_match = 0.92  # Neumonía ↔ Neumonía bacteriana
```

**Interpretación**:
- En casos con comorbilidad, basta con acertar UNA de las condiciones
- Priorizamos cobertura sobre ranking exacto
- Un score > 0.8 indica identificación correcta del diagnóstico

### 2. Score de Severidad (LLM como Juez)

El score de severidad evalúa si el modelo comprende la gravedad real de las condiciones diagnosticadas.

**Principio clave**: Consideramos TODOS los diagnósticos generados porque aparecen simultáneamente al paciente.

**Metodología Nueva (Optimista vs Pesimista)**:

```python
# Severidades van de S0 (mínima) a S10 (máxima)
GDX_severity = "S8"  # Diagnóstico real es grave
DDX_severities = ["S9", "S7", "S6", "S3", "S2"]  # Lo que predijo el modelo

# Calcular distancia máxima posible
if S_GDX <= 5:
    max_distance = 10 - S_GDX  # Puede sobrestimar hasta S10
else:
    max_distance = S_GDX  # Puede subestimar hasta S0

# Para cada DDX
for ddx_severity in DDX_severities:
    distance = abs(S_GDX - S_DDX)
    normalized = distance / max_distance
    
    # Categorizar
    if S_DDX < S_GDX:
        optimista.append(normalized)  # Subestimó gravedad
    elif S_DDX > S_GDX:
        pesimista.append(normalized)  # Sobreestimó gravedad
```

**Métricas resultantes**:
- **Score general**: Promedio de distancias normalizadas [0-1]
- **Score optimista**: Promedio cuando subestima gravedad
- **Score pesimista**: Promedio cuando sobreestima gravedad
- **0 = perfecto**, 1 = máximo error posible

**¿Por qué esta metodología?**

1. **Distancia normalizada**: Penaliza proporcionalmente según qué tan lejos está la estimación
2. **Asimetría considerada**: Un S2→S8 es diferente a S8→S2 en implicaciones clínicas
3. **Análisis optimista/pesimista**: Revela si el modelo tiende a minimizar o exagerar

## 🔧 Implementación Técnica

### Archivo Principal: `run.py`

El script orquesta todo el pipeline con estas funciones clave:

#### Etapa 1: Generación de DDX
```python
def generate_candidate_diagnoses(cases, llm, prompt_template, schema, logger):
    """
    Genera 5 diagnósticos diferenciales por caso.
    
    - Usa el modelo configurado (GPT-4, MedGemma, etc.)
    - Aplica el prompt de candidate_prompt.txt
    - Fuerza exactamente 5 DDX por caso
    """
```

#### Etapa 2: Evaluación Semántica
```python
def process_semantic_evaluation_parallel(cases, candidate_responses, logger):
    """
    Evalúa similitud DDX vs GDX en paralelo.
    
    - Warm-up del endpoint SapBERT primero
    - Procesamiento batch para eficiencia
    - ThreadPoolExecutor para paralelismo
    - Retorna best_match por caso
    """
```

#### Etapa 3: Asignación de Severidades
```python
def assign_severities_batch(llm, prompt_template, unique_ddxs, schema, logger):
    """
    Asigna severidad S0-S10 a diagnósticos únicos.
    
    - Procesamiento en batches de 50
    - Un solo LLM call por batch
    - Manejo robusto de formatos de respuesta
    """
```

#### Etapa 4: Evaluación de Severidad
```python
def evaluate_case_severity(case_id, ddxs, gdxs, best_gdx_name, 
                          severity_assignments, logger):
    """
    Calcula scores de severidad con metodología optimista/pesimista.
    
    - Usa el mejor GDX del match semántico
    - Normaliza por distancia máxima posible
    - Separa en categorías optimista/pesimista
    - Retorna estructura detallada con todas las métricas
    """
```

### Configuración: `config.yaml`

```yaml
experiment_name: "Nombre descriptivo del experimento"
dataset_path: "bench/datasets/RAMEDIS.json"

llm_configs:
  # Modelo que genera diagnósticos
  candidate_dx_gpt:
    model: "gpt-4o"  # o "jonsnow", "medgemma", etc.
    prompt: "../candidate-prompts/candidate_prompt.txt"
    output_schema: "../candidate-prompts/candidate_output_schema.json"
    params:
      temperature: 0.7
      max_tokens: 500

  # Modelo que asigna severidades
  severity_assigner_llm:
    model: "gpt-4o"
    prompt: "eval-prompts/severity_assignment_batch_prompt.txt"
    output_schema: "eval-prompts/severity_assignment_batch_schema.json"
    params:
      temperature: 0.1  # Baja para consistencia
      max_tokens: 2000
```

## 📈 Outputs y Resultados

### Estructura de Salida

Cada experimento genera:

```
results/experiment_{modelo}_{timestamp}/
├── config.yaml                  # Configuración usada
├── execution.log               # Log detallado de ejecución
├── candidate_responses.json    # DDX generados
├── semantic_evaluation.json    # Scores semánticos
├── severity_assignments.json   # Severidades asignadas
├── severity_evaluation.json    # Evaluación final
├── summary.json               # Métricas agregadas
└── plots/                     # Visualizaciones (si se generan)
```

### Formato de Resultados

#### severity_evaluation.json
```json
{
  "evaluations": [
    {
      "id": "CASE_001",
      "final_score": 0.234,
      "optimist": {
        "n": 3,
        "score": 0.187
      },
      "pessimist": {
        "n": 2,
        "score": 0.298
      },
      "gdx": {
        "disease": "Myocardial infarction",
        "severity": "S9"
      },
      "ddx_list": [
        {
          "disease": "Heart attack",
          "severity": "S9",
          "distance": 0,
          "score": 0.0
        },
        // ... más DDX
      ]
    }
  ]
}
```

## 🚀 Ejecutar Experimentos

### Experimento Básico

```bash
cd bench/pipeline
python run.py
```

### Experimento con Dataset Pequeño

```yaml
# En config.yaml
dataset_path: "bench/datasets/ramedis-5.json"
```

### Comparar Múltiples Modelos

```bash
# Ejecutar con diferentes modelos
for model in gpt-4o jonsnow medgemma; do
    # Modificar config.yaml programáticamente
    sed -i "s/model: .*/model: $model/" config.yaml
    python run.py
done
```

## 📊 Interpretación de Resultados

### Métricas Clave en summary.json

```json
{
  "semantic_evaluation": {
    "mean_score": 0.823,        // Promedio de best matches
    "standard_deviation": 0.145,
    "range": {
      "min": 0.234,
      "max": 0.981
    }
  },
  "severity_evaluation": {
    "mean_score": 0.267,        // Promedio de errores de severidad
    "standard_deviation": 0.189,
    "range": {
      "min": 0.0,
      "max": 0.764
    }
  }
}
```

### Interpretación:

**Score Semántico** (0-1, mayor es mejor):
- > 0.85: Excelente capacidad diagnóstica
- 0.70-0.85: Buena capacidad
- 0.55-0.70: Capacidad moderada
- < 0.55: Capacidad pobre

**Score de Severidad** (0-1, menor es mejor):
- < 0.20: Excelente estimación de gravedad
- 0.20-0.35: Buena estimación
- 0.35-0.50: Estimación moderada
- > 0.50: Estimación pobre

## 🧪 Validación y Testing

### Tests de Integridad

El pipeline incluye validaciones automáticas:
- Exactamente 5 DDX por caso
- Severidades en rango S0-S10
- Scores en rango [0,1]
- JSONs bien formados

### Modo Debug

```python
# En run.py, cambiar nivel de logging
logging.basicConfig(level=logging.DEBUG)
```

### Validación Manual

Revisar casos específicos en los JSONs de salida para verificar:
- Diagnósticos tienen sentido médico
- Severidades son razonables
- Best matches son correctos

## 🎯 Mejores Prácticas

1. **Consistencia**: Usar mismo dataset y prompts para comparaciones justas
2. **Repetibilidad**: Fijar seeds y temperatura para reproducibilidad
3. **Documentación**: Anotar cambios y observaciones en config.yaml
4. **Versionado**: Guardar configs exitosas para referencia
5. **Validación**: Revisar manualmente subset de resultados

## 🚨 Limitaciones Conocidas

1. **Sesgo del juez**: LLM asignando severidades puede tener sus propios sesgos
2. **Cobertura ICD-10**: No todos los diagnósticos tienen código ICD-10
3. **Variabilidad**: Resultados pueden variar entre ejecuciones
4. **Escalabilidad**: Datasets muy grandes pueden ser costosos

## 🔗 Referencias

- [Documentación de SapBERT](https://github.com/cambridgeltl/sapbert)
- [Escala de Severidad S0-S10](../docs/severity_scale.md)
- [Dashboard de Visualización](results/dashboard/README.md)