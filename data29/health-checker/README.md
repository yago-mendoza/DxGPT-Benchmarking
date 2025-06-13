# Health Checker - Analizador de Diversidad ICD10

Herramienta simple para analizar la diversidad de códigos ICD10 en datasets médicos.

## Instalación

```bash
# Desde la raíz del proyecto (donde está pyproject.toml)
pip install -e .
```

## Uso

1. **Configura tu dataset** en `config.yaml`:
```yaml
dataset: "datasets/RAMEDIS.json"
output_dir: "outputs"
```

2. **Ejecuta el análisis**:
```bash
python analyze.py
```

3. **Abre el resultado**: Se genera un archivo HTML en `outputs/[dataset]_icd10_analysis.html`

## ¿Qué analiza?

- **Cobertura**: Porcentaje de casos con códigos ICD10
- **Diversidad**: Distribución por capítulos ICD10 (I-XXII)
- **Jerarquía**: Capítulos, categorías y subcategorías
- **Validación**: Identifica códigos inválidos

## Estructura del Dataset

El sistema busca códigos ICD10 en estos campos:
- `icd10`, `icd10_code`, `icd_code`, `diagnosis_code`
- `diagnoses[].icd10_code`

## Salida

Un archivo HTML interactivo con múltiples pestañas:

### Pestañas disponibles:
1. **Resumen**: Métricas principales y cobertura
2. **Gráficos**: 
   - Pie chart de distribución por capítulos
   - Gráfico de barras horizontales
   - Línea de tendencia acumulativa
3. **Jerarquía Interactiva**:
   - Sunburst chart navegable
   - Click para explorar capítulos → categorías → bloques
   - Breadcrumb de navegación
   - Panel de información contextual
4. **Tabla Detallada**: Todos los capítulos con badges de estado
5. **Análisis Profundo**:
   - Gráfico de Pareto (concentración)
   - Matriz de especialidades médicas

### Características interactivas:
- Navegación por pestañas
- Tooltips con información detallada
- Zoom interactivo en la jerarquía
- Colores y badges indicando niveles
- Gráficos responsivos

Todo en un único archivo HTML autocontenido con JavaScript embebido.