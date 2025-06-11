# Experiment Dashboard

## Descripción

El archivo `experiment_dashboard.html` es un dashboard unificado para visualizar y comparar los resultados de los experimentos de diagnóstico DxGPT.

## Cómo usar

### Opción 1: Usando el script Python incluido (recomendado)

```bash
cd bench/pipeline/results/visualisation/
python3 serve_dashboard.py
```

Luego abre http://localhost:8000/experiment_dashboard.html en tu navegador.

### Opción 2: Usando Python directamente

```bash
cd bench/pipeline/results/visualisation/
python3 -m http.server 8000
```

Luego abre http://localhost:8000/experiment_dashboard.html en tu navegador.

### Opción 3: Usando Node.js

Si tienes Node.js instalado:

```bash
cd bench/pipeline/results/visualisation/
npx http-server -p 8000
```

## Estructura de archivos esperada

El dashboard espera encontrar los experimentos en la siguiente estructura:

```
bench/pipeline/results/
├── visualisation/
│   ├── experiment_dashboard.html
│   ├── serve_dashboard.py
│   └── README.md
├── experiment_gpt_4o_20250609011432/
│   ├── summary.json
│   ├── semantic_evaluation.json
│   └── severity_evaluation.json
├── experiment_jonsnow_20250610123831/
│   └── ...
└── ...
```

El script `serve_dashboard.py` detecta automáticamente todas las carpetas que empiezan con "experiment_" en el directorio padre.

## Funcionalidades

1. **Vista de Comparación**: Muestra todos los experimentos seleccionados en un gráfico de dispersión 2D (Severity Score vs Semantic Score)

2. **Vista de Análisis Detallado**: Permite ver gráficos individuales con configuración de grid (1x1, 1x2, 2x2)

3. **Explorador JSON**: Permite ver los datos JSON crudos de cada experimento

4. **Exportación**: Permite exportar los gráficos como imágenes PNG

## Solución de problemas

Si no se cargan los experimentos:

1. Verifica que estés sirviendo los archivos desde un servidor web (no abriendo el HTML directamente)
2. Verifica que la estructura de carpetas sea correcta
3. Abre la consola del navegador (F12) para ver mensajes de error
4. Verifica que los archivos JSON existan en las carpetas de experimentos

