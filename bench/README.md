# Bench - Sistema de Evaluación de Modelos de Diagnóstico 🏆

El directorio `bench` contiene todo el sistema de benchmarking para evaluar modelos de IA médica. Este es el núcleo del proyecto DxGPT Latitude Bench, donde se ejecutan experimentos controlados para medir el rendimiento de diferentes modelos en tareas de diagnóstico diferencial.

## 🎯 Propósito

El sistema de bench está diseñado para responder una pregunta fundamental: **¿Qué tan bien diagnostican los modelos de IA comparados con médicos expertos?**

Para responder esto, evaluamos dos dimensiones críticas:

1. **Precisión Semántica**: ¿El modelo identificó el diagnóstico correcto?
2. **Evaluación de Severidad**: ¿El modelo estimó correctamente la gravedad?

## 🏗️ Arquitectura del Sistema

```
bench/
├── README.md                # Este archivo
├── candidate-prompts/       # Prompts para generar diagnósticos
│   ├── candidate_prompt.txt         # Prompt principal para DDX
│   └── candidate_output_schema.json # Esquema de respuesta esperada
├── datasets/               # Datasets médicos procesados
│   ├── RAMEDIS.json       # Dataset completo (1000+ casos)
│   ├── ramedis-45.json    # Subset mediano para pruebas
│   └── ramedis-5.json     # Mini subset para desarrollo
└── pipeline/              # Motor de evaluación
    ├── run.py            # Script principal de ejecución
    ├── config.yaml       # Configuración de experimentos
    ├── eval-prompts/     # Prompts para evaluación
    └── results/          # Resultados de experimentos
```

## 🔄 Flujo de Evaluación

El proceso completo sigue estos pasos:

### 1. Generación de Diagnósticos (DDX)
```python
# El modelo recibe un caso clínico
caso = "Paciente de 45 años con dolor torácico..."

# Genera 5 diagnósticos diferenciales
ddx = ["Infarto", "Angina", "Reflujo", "Ansiedad", "Costocondritis"]
```

### 2. Evaluación Semántica
```python
# Comparamos DDX con diagnósticos correctos (GDX)
gdx = ["Infarto agudo de miocardio", "Síndrome coronario agudo"]

# SapBERT calcula similitud semántica
scores = {
    "Infarto": {"Infarto agudo de miocardio": 0.95},
    "Angina": {"Síndrome coronario agudo": 0.78},
    # ...
}
```

### 3. Asignación de Severidad
```python
# Un LLM asigna severidad a cada diagnóstico único
severidades = {
    "Infarto": "S9",        # Muy grave
    "Angina": "S7",         # Grave
    "Reflujo": "S3",        # Leve
    "Ansiedad": "S2",       # Muy leve
    "Costocondritis": "S1"  # Mínima
}
```

### 4. Cálculo de Métricas
```python
# Score semántico: mejor match entre DDX y GDX
semantic_score = 0.95  # (Infarto ↔ Infarto agudo)

# Score de severidad: distancia normalizada
severity_score = 0.15  # (cercano a severidad correcta)
```

## 📁 Componentes Principales

### candidate-prompts/
Contiene los prompts e instrucciones para que los modelos generen diagnósticos:

- **candidate_prompt.txt**: Prompt cuidadosamente diseñado que simula cómo un médico abordaría un caso
- **candidate_output_schema.json**: Define el formato JSON esperado para las respuestas

### datasets/
Datasets médicos validados y estructurados:

- **Origen**: Procesados desde `data29/` mediante ETL
- **Formato**: JSON con casos clínicos y diagnósticos de referencia
- **Tamaños**: Desde 5 casos (desarrollo) hasta 1000+ (evaluación completa)

### pipeline/
El corazón del sistema de evaluación:

- **run.py**: Orquesta todo el proceso de evaluación
- **config.yaml**: Define qué modelos evaluar y cómo
- **eval-prompts/**: Prompts para el juez de severidad
- **results/**: Almacena resultados de cada experimento

## 🚀 Ejecutar un Experimento

### 1. Configurar el experimento

Editar `pipeline/config.yaml`:
```yaml
experiment_name: "Mi Experimento GPT-4"
dataset_path: "bench/datasets/ramedis-45.json"

llm_configs:
  candidate_dx_gpt:
    model: "gpt-4o"  # o "jonsnow", "medgemma", etc.
    prompt: "../candidate-prompts/candidate_prompt.txt"
    
  severity_assigner_llm:
    model: "gpt-4o"
    prompt: "eval-prompts/severity_assignment_batch_prompt.txt"
```

### 2. Ejecutar

```bash
cd bench/pipeline
python run.py
```

### 3. Resultados

Se genera una carpeta en `results/` con:
- **candidate_responses.json**: DDX generados
- **semantic_evaluation.json**: Scores de similitud
- **severity_assignments.json**: Severidades asignadas
- **severity_evaluation.json**: Evaluación final
- **summary.json**: Métricas agregadas

## 📊 Métricas de Evaluación

### Score Semántico (0-1)
- **1.0**: Match perfecto con diagnóstico correcto
- **0.8+**: Muy buena precisión diagnóstica
- **0.6+**: Precisión aceptable
- **<0.6**: Precisión pobre

### Score de Severidad (0-1)
- **0.0**: Severidad perfectamente estimada
- **<0.2**: Muy buena estimación
- **<0.4**: Estimación aceptable
- **>0.4**: Estimación pobre

## 🧪 Modelos Evaluables

El sistema puede evaluar cualquier modelo que implemente la interfaz LLM:

### Azure OpenAI
- GPT-4o
- GPT-4-turbo
- GPT-3.5-turbo

### Hugging Face
- JonSnow (médico especializado)
- MedGemma (Google)
- Sakura (multilingüe médico)
- OpenBio (biomédico)

### Añadir Nuevo Modelo

1. Configurar endpoint en `.env`:
   ```env
   MIMODELO_ENDPOINT_URL=https://mi-modelo.hf.space
   ```

2. Usar en config.yaml:
   ```yaml
   model: "mimodelo"
   ```

## 📈 Visualización de Resultados

Los resultados se pueden visualizar en el dashboard interactivo:

```bash
cd bench/pipeline/results/dashboard
python serve_dashboard.py
# Abrir http://localhost:8000
```

El dashboard permite:
- Comparar múltiples modelos
- Ver evolución temporal
- Analizar casos específicos
- Exportar gráficos

## 🔧 Desarrollo y Testing

### Dataset de desarrollo
Para desarrollo rápido, usa `ramedis-5.json`:
```yaml
dataset_path: "bench/datasets/ramedis-5.json"
```

### Modo debug
En `run.py`, activar logs detallados:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Tests unitarios
```bash
pytest tests/test_bench/
```

## 📝 Mejores Prácticas

1. **Empezar pequeño**: Usar dataset de 5 casos para validar configuración
2. **Versionar prompts**: Guardar versiones de prompts que funcionan bien
3. **Documentar experimentos**: Añadir notas en config.yaml
4. **Comparar fair**: Usar mismo dataset y prompts para todos los modelos
5. **Múltiples runs**: Ejecutar 3+ veces para manejar variabilidad

## 🎯 Casos de Uso

- **Investigación**: Comparar capacidades diagnósticas de diferentes LLMs
- **Desarrollo**: Mejorar prompts para mejor rendimiento
- **Validación**: Verificar que actualizaciones no degraden performance
- **Selección**: Elegir el mejor modelo para producción

## 🚨 Consideraciones Importantes

- Los datasets son sintéticos/anonimizados, NO contienen datos reales de pacientes
- Los resultados son para investigación, NO para diagnóstico clínico real
- La evaluación es automática, puede tener sesgos o limitaciones
- Siempre validar con expertos médicos antes de conclusiones

## 🔗 Referencias

- [Pipeline - Documentación detallada](pipeline/README.md)
- [Dashboard - Guía de visualización](pipeline/results/dashboard/README.md)
- [Datasets - Origen y estructura](../data29/README.md)