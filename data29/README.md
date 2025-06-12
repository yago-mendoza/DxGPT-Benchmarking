# Data29 - Gestión y Procesamiento de Datos Médicos 📊

Este directorio es el corazón del procesamiento de datos del proyecto. Aquí es donde los datasets médicos crudos se transforman en datasets limpios y estructurados que el pipeline de evaluación puede consumir. El nombre "data29" hace referencia a Foundation 29, organización dedicada a enfermedades raras.

## 🎯 Propósito

La carpeta `data29` cumple tres funciones fundamentales:

1. **Almacenamiento de datos crudos**: Datasets originales de diferentes fuentes médicas
2. **ETL (Extract, Transform, Load)**: Scripts para limpiar y estructurar los datos
3. **Control de calidad**: Validación de integridad y consistencia (futuro health-checker)

## 🏗️ Estructura

```
data29/
├── README.md              # Este archivo
├── health-checker/        # Validador de calidad de datos (en desarrollo)
└── data-repos/           # Repositorio principal de datos
    ├── raw/              # Datos crudos sin procesar
    │   ├── more.txt
    │   └── ukranian.csv
    └── urg_torre & ramedis_bench/
        ├── current-data/  # Datos procesados y listos para usar
        │   ├── RAMEDIS.json      # Dataset principal para bench
        │   ├── urg_torre.json    # Dataset de urgencias
        │   └── visualisations/   # Análisis visual de los datos
        └── etl/          # Scripts y procesos de transformación
            ├── 25.05/    # Versión mayo
            └── 25.06/    # Versión junio (más reciente)
```

## 📊 Datasets Principales

### RAMEDIS.json
**Ubicación**: `data-repos/urg_torre & ramedis_bench/current-data/RAMEDIS.json`

Dataset de casos médicos con diagnósticos diferenciales. Cada caso incluye:
- **id**: Identificador único
- **case**: Descripción clínica del paciente
- **diagnoses**: Lista de diagnósticos correctos (GDX) con:
  - `name`: Nombre del diagnóstico
  - `severity`: Nivel de severidad (S0-S10)
  - `code`: Código ICD-10 (cuando aplica)

Ejemplo de estructura:
```json
{
  "id": "RAMEDIS_0001",
  "case": "A 45-year-old male presents with chest pain...",
  "diagnoses": [
    {
      "name": "Myocardial infarction",
      "severity": "S9",
      "code": "I21.9"
    }
  ]
}
```

### urg_torre.json
**Ubicación**: `data-repos/urg_torre & ramedis_bench/current-data/urg_torre.json`

Dataset de casos de urgencias hospitalarias con estructura similar pero enfocado en escenarios de emergencia.

## 🔄 Proceso ETL

El proceso de transformación de datos (ETL) se documenta en las carpetas `etl/25.05/` y `etl/25.06/`. Cada versión incluye:

### 1. Datos Originales
En `original-csvs/` se encuentran los datasets crudos de diferentes fuentes:
- `test_RAMEDIS.csv`: Casos de diagnóstico diferencial
- `test_HMS.csv`: Casos del Hospital Management System
- `test_LIRICAL.csv`: Casos de enfermedades genéticas
- `test_critical.csv`, `test_severe.csv`: Casos por severidad

### 2. Scripts de Transformación
Cada subcarpeta en `etl/25.06/` representa una etapa de transformación:

- **ramedis-v2-added-origin-column**: Añade columna de origen para trazabilidad
- **ramedis-v3-formatting-diagnosis-names**: Normaliza nombres de diagnósticos
- **ramedis-v4-setting-complexity**: Añade niveles de complejidad
- **both-rv5-uv7-diagnosis-severity-assessment**: Asigna severidades usando mapping

### 3. Mapeos y Taxonomías
En `legacy-mappings/`:
- `disease2name.json`: Mapeo de códigos a nombres de enfermedades
- `hpo2name.json`: Human Phenotype Ontology
- `rare_ramedis_class.jsonl`: Clasificación de enfermedades raras

## 🏥 Health Checker (Futuro)

El directorio `health-checker/` está destinado a contener herramientas de validación de calidad:

### Funcionalidades planeadas:

1. **Validación de códigos ICD-10**:
   ```python
   from utils.icd10 import ICD10Taxonomy
   
   def validate_dataset(dataset_path):
       taxonomy = ICD10Taxonomy()
       errors = []
       
       for case in dataset:
           for diagnosis in case['diagnoses']:
               if 'code' in diagnosis:
                   if not taxonomy.get(diagnosis['code']):
                       errors.append(f"Invalid ICD-10: {diagnosis['code']}")
       
       return errors
   ```

2. **Verificación de severidades**:
   - Validar que todas las severidades estén en rango S0-S10
   - Detectar inconsistencias (ej: "Common cold" con S10)

3. **Integridad de datos**:
   - Casos sin diagnósticos
   - Diagnósticos duplicados
   - Campos faltantes o malformados

4. **Estadísticas de calidad**:
   - Distribución de severidades
   - Cobertura de códigos ICD-10
   - Balance de tipos de casos

## 📈 Visualizaciones

La carpeta `visualisations/` contiene análisis gráficos generados por `stats_new.py`:

- **severity_distribution.png**: Histograma de distribución de severidades
- **diagnostic_codes_distribution.png**: Frecuencia de códigos ICD-10
- **case_complexity.jpg**: Análisis de complejidad de casos
- **diagnosis_network.png**: Red de relaciones entre diagnósticos

## 🔧 Uso en el Pipeline

Los datos procesados en `data29` son consumidos por el pipeline de evaluación:

```yaml
# bench/pipeline/config.yaml
dataset_path: "bench/datasets/RAMEDIS.json"
```

El archivo referenciado es una copia del procesado en `data29/data-repos/urg_torre & ramedis_bench/current-data/`.

## 🚀 Cómo Añadir Nuevos Datasets

1. **Colocar datos crudos** en `data-repos/raw/`

2. **Crear script ETL**:
   ```python
   # etl/nueva_version/transform.py
   import json
   import pandas as pd
   
   # Leer datos crudos
   df = pd.read_csv('../raw/nuevo_dataset.csv')
   
   # Transformar al formato esperado
   cases = []
   for _, row in df.iterrows():
       case = {
           "id": f"NEW_{row['id']}",
           "case": row['description'],
           "diagnoses": [
               {
                   "name": row['diagnosis'],
                   "severity": assign_severity(row['diagnosis']),
                   "code": get_icd10_code(row['diagnosis'])
               }
           ]
       }
       cases.append(case)
   
   # Guardar
   with open('nuevo_dataset.json', 'w') as f:
       json.dump(cases, f, indent=2)
   ```

3. **Validar con health-checker** (cuando esté implementado)

4. **Copiar a bench/datasets/** para uso en experimentos

## 📝 Notas Importantes

- Los datos médicos son sensibles: NO subir datos reales de pacientes
- Mantener trazabilidad: documentar origen y transformaciones
- Versionar cambios: usar carpetas con fechas para ETL
- Validar siempre: ejecutar checks de calidad antes de usar

## 🔗 Referencias

- [ICD-10 Browser](https://icd.who.int/browse10/2019/en)
- [Human Phenotype Ontology](https://hpo.jax.org/)
- [Foundation 29](https://www.foundation29.org/)