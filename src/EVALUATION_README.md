# DxGPT Evaluation Framework

Sistema modular y automatizado para evaluar el rendimiento de DxGPT con múltiples criterios y datasets.

## Estructura

```
src/
├── evaluate.py           # CLI principal
├── evaluators/          # Evaluadores disponibles
│   ├── semantic.py      # Evaluador de similaridad semántica
│   └── severity.py      # Evaluador de severidad (placeholder)
├── core/                # Componentes principales
│   ├── session.py       # Gestión de sesiones de evaluación
│   ├── registry.py      # Registro de evaluadores
│   └── config.py        # Gestión de configuración
├── configs/             # Archivos de configuración
│   └── default_config.yaml
├── datasets/            # Datasets para evaluación
├── results/             # Resultados de evaluaciones
├── visualizations/      # Módulo de visualización
└── dxgpt-prompt/        # Prompts para DxGPT
```

## Uso Básico

### 1. Listar evaluadores disponibles
```bash
python src/evaluate.py --list-evaluators
```

### 2. Listar datasets disponibles
```bash
python src/evaluate.py --list-datasets
```

### 3. Ejecutar evaluación semántica
```bash
python src/evaluate.py --evaluators semantic --datasets src/golden-dataset/mini-test.csv
```

### 4. Ejecutar con configuración personalizada
```bash
python src/evaluate.py --config mi_config.yaml --evaluators semantic --datasets test.csv
```

### 5. Modificar parámetros en línea de comandos
```bash
python src/evaluate.py --evaluators semantic --datasets test.csv \
  --set evaluators.semantic.config.batch_size 10 \
  --set evaluators.semantic.config.n_diagnoses 3
```

## Formato de Datasets

### Evaluador Semántico
Columnas requeridas:
- `id`: Identificador único
- `case`: Descripción del caso/síntomas
- `diagnosis`: Diagnósticos dorados separados por ";"

### Evaluador de Severidad (cuando se implemente)
Columnas requeridas:
- `id`: Identificador único
- `case`: Descripción del caso/síntomas
- `diagnosis`: Diagnósticos dorados separados por ";"
- `severity`: Nivel de severidad

## Configuración

El archivo `configs/default_config.yaml` contiene la configuración por defecto:

```yaml
evaluators:
  semantic:
    enabled: true
    config:
      batch_size: 5          # Casos por lote
      n_diagnoses: 5         # Diagnósticos a generar
      llm_model: "gpt-4o"    # Modelo a usar
      temperature: 0.3       # Temperatura del LLM
```

## Resultados

Cada evaluación genera:
1. **session_metadata.json**: Metadatos de la sesión (commit hash, configuración, etc.)
2. **<evaluator>_<dataset>_details.json**: Resultados detallados
3. **visualizations/**: Gráficos y visualizaciones

### Estructura de resultados (semántico):
```json
{
  "uid": "0",
  "score": 0.85,
  "diagnosis": "Diagnosis1; Diagnosis2",
  "details": {
    "ddx": ["Predicted1", "Predicted2", ...],
    "similarity_matrix": [[0.85, 0.72], ...]
  }
}
```

## Visualizaciones

Se generan automáticamente:
- **score_distribution**: Distribución de puntuaciones
- **confusion_matrix**: Matriz de confusión
- **top_errors**: Casos con menor similaridad

Desactivar visualizaciones:
```bash
python src/evaluate.py --evaluators semantic --datasets test.csv --no-viz
```

## Extensibilidad

### Crear nuevo evaluador:

1. Crear archivo en `src/evaluators/mi_evaluador.py`
2. Heredar de `BaseEvaluator`
3. Implementar métodos requeridos:

```python
from core.registry import BaseEvaluator

class MiEvaluator(BaseEvaluator):
    EXPECTED_COLUMNS = ['id', 'case', 'mi_columna']
    
    def get_name(self):
        return "mi_evaluador"
    
    def evaluate(self, dataset_path, config):
        # Implementar lógica
        return results
```

### Agregar visualizaciones:

1. Crear función en `visualizations/`
2. Registrar en configuración:

```yaml
visualizations:
  mi_evaluador:
    - type: "mi_visualizacion"
      enabled: true
```

## Sesiones y Trazabilidad

Cada evaluación crea una sesión única con:
- ID único con timestamp
- Información completa del último commit Git:
  - Hash del commit
  - Mensaje del commit
  - Autor y fecha
  - Branch actual
  - Estado del repositorio (limpio/modificado)
  - Archivos no commiteados
- Configuración utilizada
- Hash de archivos de evaluadores y datasets
- Tiempos de ejecución
- **Logs completos** guardados en `evaluation.log`

Esto permite reproducibilidad completa de experimentos.

## Ejemplos Avanzados

### Evaluar múltiples datasets
```bash
python src/evaluate.py --evaluators semantic \
  --datasets dataset1.csv dataset2.csv dataset3.csv
```

### Cambiar directorio de salida
```bash
python src/evaluate.py --evaluators semantic --datasets test.csv \
  --output-dir mis_resultados/experimento1
```

### Configuración compleja
```bash
python src/evaluate.py \
  --config base_config.yaml \
  --evaluators semantic severity \
  --datasets test1.csv test2.csv \
  --set evaluators.semantic.config.batch_size 10 \
  --set visualizations.enabled true \
  --log-level DEBUG
```