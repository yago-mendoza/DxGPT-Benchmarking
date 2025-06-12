# Dashboard de Visualización - DxGPT Benchmark 📊

Este dashboard interactivo permite visualizar, comparar y analizar los resultados de múltiples experimentos de evaluación de modelos de diagnóstico médico. Proporciona insights visuales sobre el rendimiento en las dos dimensiones clave: precisión semántica y estimación de severidad.

## 🎯 ¿Qué muestra el Dashboard?

### Vista Principal: Gráfico de Dispersión 2D

El gráfico principal posiciona cada modelo en un espacio bidimensional:

- **Eje X (Severity Score)**: Qué tan bien estima la gravedad (0 = perfecto, 1 = pésimo)
- **Eje Y (Semantic Score)**: Qué tan bien identifica diagnósticos (0 = pésimo, 1 = perfecto)

**Interpretación visual**:
```
          Semantic Score ↑
                1.0 ┌─────────────────────┐
                    │ ◆ Ideal             │
                    │   (Alta precisión,  │
                    │    baja distancia)  │
                0.5 ├─────────────────────┤
                    │         ◇           │
                    │     Moderado        │
                    │                     │
                0.0 └─────────────────────┘
                    0.0       0.5        1.0
                         Severity Score →
```

**Cuadrantes**:
- **Superior Izquierda** (ideal): Alta precisión diagnóstica + buena estimación de severidad
- **Superior Derecha**: Identifica bien pero estima mal la gravedad
- **Inferior Izquierda**: Estima bien gravedad pero falla en diagnósticos
- **Inferior Derecha** (peor): Falla en ambas dimensiones

### Visualizaciones Disponibles

1. **Comparison View**: Todos los modelos en un solo gráfico
2. **Detailed Analysis**: Gráficos individuales por modelo
3. **JSON Explorer**: Datos crudos para análisis profundo
4. **Experiment Panel**: Lista interactiva de experimentos

## 🚀 Cómo Ejecutar el Dashboard

### Opción 1: Script Python Incluido (Recomendado)

```bash
cd bench/pipeline/results/dashboard/
python serve_dashboard.py
```

El script automáticamente:
- Detecta todos los experimentos disponibles
- Inicia servidor en puerto 8000
- Abre el navegador (opcional)

### Opción 2: Servidor HTTP Simple

```bash
# Con Python
python -m http.server 8000

# Con Node.js
npx http-server -p 8000

# Con PHP
php -S localhost:8000
```

Luego navegar a: http://localhost:8000

### Opción 3: Servidor de Desarrollo

Para desarrollo con hot-reload:
```bash
# Instalar live-server globalmente
npm install -g live-server

# Ejecutar
live-server --port=8000
```

## 📁 Estructura de Archivos

```
dashboard/
├── README.md              # Este archivo
├── serve_dashboard.py     # Script servidor con auto-detección
└── scripts/              # Aplicación web
    ├── index.html        # Interfaz principal
    ├── script.js         # Lógica de Chart.js
    └── style.css         # Estilos responsivos
```

### Datos Esperados

El dashboard busca experimentos en el directorio padre:

```
../experiment_{modelo}_{timestamp}/
    ├── summary.json              # Métricas agregadas
    ├── semantic_evaluation.json  # Detalles semánticos
    └── severity_evaluation.json  # Detalles de severidad
```

## 🎨 Características del Dashboard

### 1. Panel de Experimentos (Togglable)

Lista interactiva de todos los experimentos disponibles:
- Checkbox para incluir/excluir del análisis
- Información del modelo y timestamp
- Códigos de color por tipo de modelo

### 2. Vista de Comparación

Gráfico scatter principal con:
- **Puntos de datos**: Cada modelo es un punto
- **Tooltips informativos**: Hover para ver detalles
- **Leyenda interactiva**: Click para mostrar/ocultar
- **Zoom y pan**: Navegación interactiva

### 3. Análisis Detallado

Opciones de visualización:
- **Grid 1x1**: Un gráfico grande
- **Grid 1x2**: Dos gráficos lado a lado
- **Grid 2x2**: Cuatro gráficos en cuadrícula

Tipos de gráficos disponibles:
- Scatter plot (severity vs semantic)
- Distribución de scores semánticos
- Distribución de scores de severidad
- Análisis optimista vs pesimista

### 4. Explorador JSON

Visualización estructurada de datos crudos:
- Navegación por árbol expandible
- Búsqueda dentro del JSON
- Exportación de datos

### 5. Funciones de Exportación

- **Export Chart**: Guarda gráfico como PNG
- **Copy Data**: Copia datos al portapapeles
- **Download JSON**: Descarga datos completos

## 📊 Interpretación de Visualizaciones

### Gráfico Principal: Posicionamiento de Modelos

```javascript
// Ejemplo de interpretación
{
  "GPT-4o": {
    "position": [0.15, 0.89],  // [severity, semantic]
    "interpretation": "Excelente en ambas dimensiones"
  },
  "MedGemma": {
    "position": [0.45, 0.75],
    "interpretation": "Bueno en diagnóstico, moderado en severidad"
  }
}
```

### Métricas Mostradas

1. **Mean Scores**: Promedio sobre todos los casos
2. **Standard Deviation**: Consistencia del modelo
3. **Range**: Mejor y peor caso
4. **Optimist/Pessimist Ratio**: Tendencia a sub/sobreestimar

### Colores y Símbolos

- **Azul**: Modelos GPT (OpenAI)
- **Verde**: Modelos médicos especializados
- **Naranja**: Modelos open-source
- **Rojo**: Modelos con bajo rendimiento
- **Tamaño del punto**: Proporcional al número de casos

## 🔧 Personalización

### Modificar Colores de Modelos

En `script.js`:
```javascript
const modelColors = {
    'gpt-4o': 'rgba(54, 162, 235, 0.8)',      // Azul
    'jonsnow': 'rgba(75, 192, 192, 0.8)',     // Verde
    'medgemma': 'rgba(255, 159, 64, 0.8)',    // Naranja
    // Añadir nuevos modelos aquí
};
```

### Ajustar Escalas

```javascript
scales: {
    x: {
        min: 0,
        max: 1,
        title: {
            text: 'Severity Score (lower is better)'
        }
    },
    y: {
        min: 0,
        max: 1,
        title: {
            text: 'Semantic Score (higher is better)'
        }
    }
}
```

### Añadir Nuevas Visualizaciones

1. Crear nueva función en `script.js`:
```javascript
function createCustomChart(containerId, experiments) {
    // Tu lógica de visualización
}
```

2. Añadir opción en el HTML:
```html
<button onclick="showCustomView()">Mi Vista</button>
```

## 🐛 Solución de Problemas

### No se cargan experimentos

1. **Verificar servidor HTTP**: No abrir HTML directamente
2. **Revisar consola**: F12 → Console para errores
3. **Validar JSONs**: Cada experimento debe tener los 3 JSONs requeridos
4. **Permisos**: Verificar permisos de lectura en las carpetas

### Gráficos no se muestran

1. **Cache del navegador**: Ctrl+F5 para refrescar
2. **CDN de Chart.js**: Verificar conexión a internet
3. **Datos válidos**: Revisar que los scores estén en [0,1]

### Exportación falla

1. **Navegador compatible**: Chrome/Firefox/Edge modernos
2. **Tamaño del canvas**: Reducir cantidad de experimentos
3. **Memoria**: Cerrar otras pestañas si hay muchos datos

## 🚀 Características Avanzadas

### Análisis Comparativo Automático

El dashboard calcula automáticamente:
- **Best performer**: Modelo más cercano a (0, 1)
- **Most consistent**: Menor desviación estándar
- **Best semantic**: Mayor score semántico promedio
- **Best severity**: Menor score de severidad promedio

### Filtros Dinámicos

- Filtrar por rango de fechas
- Filtrar por tipo de modelo
- Filtrar por score mínimo
- Mostrar solo top N modelos

### Integración con Pipeline

El dashboard se actualiza automáticamente cuando:
- Se ejecutan nuevos experimentos
- Se modifican resultados existentes
- Se añaden nuevos modelos

## 📈 Mejores Prácticas de Visualización

1. **Comparar similares**: Agrupar modelos de la misma familia
2. **Considerar varianza**: No solo ver promedios
3. **Múltiples ejecuciones**: Promediar resultados de varias corridas
4. **Contexto clínico**: Recordar que menor severidad error puede ser crítico
5. **Documentar insights**: Usar la función de notas para observaciones

## 🔗 Exportación y Reportes

### Generar Reporte Automático

```python
# generate_report.py
from dashboard import DashboardData

data = DashboardData("../")
report = data.generate_comparison_report(
    models=['gpt-4o', 'jonsnow', 'medgemma'],
    output_format='markdown'
)
```

### Integración con Notebooks

```python
# En Jupyter
from IPython.display import IFrame
IFrame('http://localhost:8000', width=1000, height=600)
```

## 🎯 Roadmap Futuro

- [ ] Filtros avanzados por fecha/modelo
- [ ] Exportación a Excel/CSV
- [ ] Análisis de tendencias temporales
- [ ] Comparación estadística (t-test, ANOVA)
- [ ] Modo presentación fullscreen
- [ ] Temas oscuro/claro
- [ ] API REST para integración

## 📚 Referencias

- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [Metodología de Evaluación](../../README.md)
- [Estructura de Resultados](../README.md)

