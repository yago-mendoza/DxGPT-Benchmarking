# Results - Almacén de Experimentos y Resultados 📊

Este directorio contiene todos los resultados de experimentos ejecutados con el pipeline de evaluación DxGPT. Cada experimento se almacena en su propia carpeta con nomenclatura estandarizada, facilitando la comparación y el análisis histórico.

## 🎯 Estructura de Resultados

### Nomenclatura de Carpetas

Las carpetas siguen el patrón: `experiment_{modelo}_{timestamp}`

Ejemplos:
- `experiment_gpt_4o_20250609011432/`
- `experiment_jonsnow_20250610123831/`
- `experiment_medgemma_20250610153330/`

El timestamp usa formato: `YYYYMMDDHHMMSS` para ordenación cronológica natural.

### Contenido de Cada Experimento

```
experiment_{modelo}_{timestamp}/
├── config.yaml                  # Configuración exacta usada
├── execution.log               # Log completo de la ejecución
├── candidate_responses.json    # DDX generados por el modelo
├── semantic_evaluation.json    # Scores de similitud semántica
├── severity_assignments.json   # Severidades asignadas a DDX únicos
├── severity_evaluation.json    # Evaluación de severidad por caso
├── summary.json               # Métricas agregadas finales
└── plots/                     # Gráficos generados (opcional)
    ├── semantic_distribution.png
    ├── severity_distribution.png
    └── comparison_scatter.png
```

## 📄 Descripción de Archivos

### 1. config.yaml
Copia exacta de la configuración usada. Crucial para reproducibilidad.

```yaml
experiment_name: "Baseline GPT-4o en RAMEDIS"
dataset_path: "bench/datasets/RAMEDIS.json"
llm_configs:
  candidate_dx_gpt:
    model: "gpt-4o"
    # ... resto de configuración
```

### 2. execution.log
Registro detallado de toda la ejecución:
- Timestamps de cada etapa
- Mensajes de progreso
- Errores y warnings
- Estadísticas de rendimiento

### 3. candidate_responses.json
Los diagnósticos diferenciales generados:

```json
{
  "metadata": {
    "experiment_name": "Baseline GPT-4o",
    "model_used": "gpt-4o",
    "timestamp": "2025-06-09T01:14:32"
  },
  "responses": [
    {
      "case_id": "RAMEDIS_001",
      "ddxs": [
        "Myocardial infarction",
        "Unstable angina",
        "Pulmonary embolism",
        "Aortic dissection",
        "Panic attack"
      ]
    }
    // ... más casos
  ]
}
```

### 4. semantic_evaluation.json
Evaluación de qué tan bien el modelo identificó los diagnósticos:

```json
{
  "metadata": {
    "experiment_name": "Baseline GPT-4o",
    "timestamp": "2025-06-09T01:16:45"
  },
  "evaluations": [
    {
      "case_id": "RAMEDIS_001",
      "gdx_set": ["Acute myocardial infarction", "STEMI"],
      "best_match": {
        "gdx": "Acute myocardial infarction",
        "ddx": "Myocardial infarction",
        "score": 0.924
      },
      "ddx_semantic_scores": {
        "Myocardial infarction": [0.924, 0.856],
        "Unstable angina": [0.678, 0.623],
        // ... scores para cada DDX vs cada GDX
      }
    }
  ]
}
```

### 5. severity_assignments.json
Severidades asignadas por el LLM juez a todos los diagnósticos únicos:

```json
{
  "metadata": {
    "model_used": "gpt-4o",
    "total_unique_diagnoses": 487,
    "timestamp": "2025-06-09T01:18:22"
  },
  "assigned_severities": [
    {
      "ddx_unique_name": "Myocardial infarction",
      "inferred_severity": "S9"
    },
    {
      "ddx_unique_name": "Common cold",
      "inferred_severity": "S1"
    }
    // ... todas las severidades únicas
  ]
}
```

### 6. severity_evaluation.json
Evaluación final de severidad con metodología optimista/pesimista:

```json
{
  "metadata": {
    "experiment_name": "Baseline GPT-4o",
    "timestamp": "2025-06-09T01:20:15"
  },
  "evaluations": [
    {
      "id": "RAMEDIS_001",
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
        "disease": "Acute myocardial infarction",
        "severity": "S9"
      },
      "ddx_list": [
        {
          "disease": "Myocardial infarction",
          "severity": "S9",
          "distance": 0,
          "score": 0.0
        },
        {
          "disease": "Unstable angina",
          "severity": "S7",
          "distance": 2,
          "score": 0.2
        }
        // ... resto de DDX con sus scores
      ]
    }
  ]
}
```

### 7. summary.json
Resumen ejecutivo con métricas agregadas:

```json
{
  "metadata": {
    "experiment_name": "Baseline GPT-4o",
    "llm_configs": {
      "candidate_dx_gpt": {
        "model": "gpt-4o",
        "prompt": "../candidate-prompts/candidate_prompt.txt"
      }
    }
  },
  "semantic_evaluation": {
    "mean_score": 0.823,
    "standard_deviation": 0.145,
    "range": {
      "min": 0.234,
      "max": 0.981
    }
  },
  "severity_evaluation": {
    "mean_score": 0.267,
    "standard_deviation": 0.189,
    "range": {
      "min": 0.0,
      "max": 0.764
    }
  }
}
```

## 📈 Dashboard de Visualización

El subdirectorio `dashboard/` contiene una aplicación web interactiva para explorar y comparar resultados:

```
dashboard/
├── README.md               # Documentación del dashboard
├── serve_dashboard.py      # Script servidor Python
└── scripts/               # Aplicación web
    ├── index.html         # Interfaz principal
    ├── script.js          # Lógica de visualización
    └── style.css          # Estilos
```

### Ejecutar el Dashboard

```bash
cd bench/pipeline/results/dashboard
python serve_dashboard.py
# Abrir http://localhost:8000
```

## 🔍 Análisis de Resultados

### Comparación Rápida entre Experimentos

Para comparar modelos, revisar `summary.json` de cada experimento:

```python
import json
import glob

# Cargar todos los summaries
summaries = {}
for exp_dir in glob.glob("experiment_*/"):
    with open(f"{exp_dir}/summary.json") as f:
        data = json.load(f)
        model = data['metadata']['llm_configs']['candidate_dx_gpt']['model']
        summaries[model] = {
            'semantic': data['semantic_evaluation']['mean_score'],
            'severity': data['severity_evaluation']['mean_score']
        }

# Mostrar comparación
for model, scores in summaries.items():
    print(f"{model}: Semantic={scores['semantic']:.3f}, Severity={scores['severity']:.3f}")
```

### Análisis Detallado de Casos

Para investigar casos específicos problemáticos:

```python
# Cargar evaluación de severidad
with open("experiment_gpt_4o_20250609011432/severity_evaluation.json") as f:
    data = json.load(f)

# Encontrar casos con peor score de severidad
worst_cases = sorted(
    data['evaluations'], 
    key=lambda x: x['final_score'], 
    reverse=True
)[:10]

for case in worst_cases:
    print(f"Case {case['id']}: Score={case['final_score']:.3f}")
    print(f"  GDX: {case['gdx']['disease']} ({case['gdx']['severity']})")
    print(f"  Optimistas: {case['optimist']['n']}, Pesimistas: {case['pessimist']['n']}")
```

## 🗂️ Gestión de Experimentos

### Limpieza de Experimentos Antiguos

Para mantener el directorio manejable:

```bash
# Archivar experimentos de más de 30 días
find . -name "experiment_*" -mtime +30 -exec tar -czf {}.tar.gz {} \;
find . -name "experiment_*" -mtime +30 -type d -exec rm -rf {} \;
```

### Respaldo de Resultados Importantes

```bash
# Crear backup de experimentos clave
mkdir -p backups/2025-06
cp -r experiment_gpt_4o_* backups/2025-06/
```

## 📊 Interpretación de Métricas

### Score Semántico (0-1, mayor es mejor)
- **0.85+**: El modelo identifica correctamente la mayoría de diagnósticos
- **0.70-0.85**: Buen rendimiento con algunos errores
- **0.55-0.70**: Rendimiento moderado, mejoras necesarias
- **<0.55**: Rendimiento pobre, revisar prompts o modelo

### Score de Severidad (0-1, menor es mejor)
- **<0.20**: Excelente estimación de gravedad
- **0.20-0.35**: Buena estimación con desviaciones menores
- **0.35-0.50**: Estimación moderada, tiende a sobre/subestimar
- **>0.50**: Estimación pobre de gravedad

### Análisis Optimista vs Pesimista
- **Alto score optimista**: El modelo subestima gravedad (peligroso)
- **Alto score pesimista**: El modelo sobreestima gravedad (causa ansiedad)
- **Balance**: Idealmente ambos scores deberían ser bajos y similares

## 🚀 Mejores Prácticas

1. **Documentar cada experimento**: Añadir notas en config.yaml sobre cambios
2. **Guardar configuraciones exitosas**: Para referencia futura
3. **Comparar con baseline**: Siempre tener un experimento de referencia
4. **Múltiples ejecuciones**: Para manejar variabilidad estocástica
5. **Versionado de prompts**: Mantener histórico de prompts usados

## 🔗 Herramientas Útiles

### Script para Generar Reporte

```python
# generate_report.py
import json
import pandas as pd
from pathlib import Path

def generate_experiment_report(exp_dir):
    """Genera reporte markdown de un experimento."""
    
    # Cargar datos
    with open(f"{exp_dir}/summary.json") as f:
        summary = json.load(f)
    
    # Generar reporte
    report = f"""
# Reporte: {exp_dir}

## Configuración
- Modelo: {summary['metadata']['llm_configs']['candidate_dx_gpt']['model']}
- Dataset: {summary['metadata']['llm_configs']['candidate_dx_gpt'].get('dataset', 'N/A')}

## Resultados
### Evaluación Semántica
- Score promedio: {summary['semantic_evaluation']['mean_score']:.3f}
- Desviación estándar: {summary['semantic_evaluation']['standard_deviation']:.3f}
- Rango: [{summary['semantic_evaluation']['range']['min']:.3f}, {summary['semantic_evaluation']['range']['max']:.3f}]

### Evaluación de Severidad
- Score promedio: {summary['severity_evaluation']['mean_score']:.3f}
- Desviación estándar: {summary['severity_evaluation']['standard_deviation']:.3f}
- Rango: [{summary['severity_evaluation']['range']['min']:.3f}, {summary['severity_evaluation']['range']['max']:.3f}]
"""
    
    with open(f"{exp_dir}/REPORT.md", "w") as f:
        f.write(report)
    
    print(f"Reporte generado: {exp_dir}/REPORT.md")

# Generar para todos los experimentos
for exp in Path(".").glob("experiment_*/"):
    generate_experiment_report(exp)
```

## 🔗 Referencias

- [Pipeline - Metodología completa](../README.md)
- [Dashboard - Guía de visualización](dashboard/README.md)
- [Configuración - Referencia](../config.yaml)