# AzureLLM v4 🚀

**Wrapper elegante y poderoso para Azure OpenAI API**

Una librería diseñada para desarrolladores que quieren máxima productividad con mínima complejidad. Configuración automática, esquemas JSON, plantillas reutilizables y código limpio.

## 🎯 Características Principales

- ✅ **Configuración automática** desde variables de entorno
- ✅ **Salida estructurada** con esquemas JSON
- ✅ **Sistema de plantillas** reutilizables
- ✅ **Procesamiento por lotes (batch)** eficiente
- ✅ **API limpia** con alias intuitivos
- ✅ **Manejo de errores** elegante
- ✅ **Compatibilidad hacia atrás** total
- ✅ **Zero-config** para casos simples

## 📦 Instalación

```bash
pip install openai python-dotenv pyyaml
```

> **Note:** La librería requiere `openai` para Azure OpenAI y opcionalmente `python-dotenv` para variables de entorno.

## ⚙️ Configuración

### Variables de Entorno

Crea un archivo `.env` en tu proyecto:

```bash
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_API_KEY=tu-api-key-aqui
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

> **Tip:** La librería carga automáticamente las variables de entorno. No necesitas configuración adicional.

## 🚀 Inicio Rápido

### Uso Básico

```python
from utils.llm import Azure

# ¡Configuración automática desde .env!
llm = Azure("gpt-4o")
response = llm.generate("Explica qué es la inteligencia artificial")
print(response)
```

### Una Línea de Código

```python
from utils.llm import quick_generate

# Para casos súper simples
response = quick_generate("Traduce 'Hello' al español")
```

## 📝 Ejemplos de Uso

### 1. Generación Básica con Parámetros

```python
from utils.llm import Azure

llm = Azure("gpt-4o")

# Con temperatura baja para respuestas consistentes
response = llm.generate(
    "Resume este texto en 3 puntos clave",
    temperature=0.2,
    max_tokens=150
)
```

### 2. Salida Estructurada con JSON

```python
# Definir esquema de salida
schema = {
    "type": "object",
    "properties": {
        "resumen": {"type": "string"},
        "puntos_clave": {
            "type": "array",
            "items": {"type": "string"}
        },
        "categoria": {"type": "string"}
    },
    "required": ["resumen", "puntos_clave", "categoria"]
}

# Generar respuesta estructurada
response = llm.generate(
    "Analiza este artículo sobre IA...",
    schema=schema
)

# response es un dict con la estructura definida
print(response["resumen"])
print(response["puntos_clave"])
```

### 3. Plantillas Reutilizables

```python
# Crear plantilla con variables
traductor = llm.template(
    "Traduce '{texto}' del {idioma_origen} al {idioma_destino}",
    temperature=0.3
)

# Usar la plantilla múltiples veces
resultado1 = traductor(
    texto="Hello world", 
    idioma_origen="inglés", 
    idioma_destino="español"
)

resultado2 = traductor(
    texto="Bonjour", 
    idioma_origen="francés", 
    idioma_destino="español"
)
```

### 4. Plantillas con Esquemas

```python
# Plantilla que siempre devuelve JSON estructurado
analizador = llm.template(
    "Analiza el sentimiento de: '{texto}'",
    schema={
        "type": "object",
        "properties": {
            "sentimiento": {"type": "string"},
            "confianza": {"type": "number"},
            "palabras_clave": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["sentimiento", "confianza", "palabras_clave"]
    }
)

resultado = analizador(texto="¡Me encanta este producto!")
print(f"Sentimiento: {resultado['sentimiento']}")
print(f"Confianza: {resultado['confianza']}")
```

### 5. Procesamiento por Lotes (Batch)

```python
# Procesar múltiples items en una sola llamada
items = [
    {"id": 1, "text": "Hello world"},
    {"id": 2, "text": "Good morning"},
    {"id": 3, "text": "How are you?"}
]

# Sin schema - devuelve lista de strings
responses = llm.generate(
    "Traduce cada texto al español",
    batch_items=items
)
# responses = ["Hola mundo", "Buenos días", "¿Cómo estás?"]

# Con schema - devuelve lista estructurada
schema = {
    "type": "object",
    "properties": {
        "original": {"type": "string"},
        "traduccion": {"type": "string"},
        "idioma": {"type": "string"}
    }
}

responses = llm.generate(
    "Traduce cada texto al español y detecta el idioma original",
    batch_items=items,
    schema=schema
)
# responses = [
#     {"original": "Hello world", "traduccion": "Hola mundo", "idioma": "inglés"},
#     {"original": "Good morning", "traduccion": "Buenos días", "idioma": "inglés"},
#     {"original": "How are you?", "traduccion": "¿Cómo estás?", "idioma": "inglés"}
# ]
```

## 🔧 Configuración Avanzada

### Configuración Personalizada

```python
from utils.llm import LLMConfig, AzureLLM

# Configuración explícita
config = LLMConfig(
    endpoint="https://mi-recurso.openai.azure.com/",
    api_key="mi-api-key",
    deployment_name="gpt-4o",
    temperature=0.7,
    validate_schema=True  # Validación strict de esquemas
)

llm = AzureLLM(config=config)
```

### Configuración desde Entorno con Overrides

```python
from utils.llm import Azure

# Usar variables de entorno pero override algunos valores
llm = Azure("gpt-4o", temperature=0.1, validate_schema=True)
```

## 📚 API Reference

### Azure / AzureLLM

La clase principal para interactuar con Azure OpenAI.

```python
llm = Azure(deployment_name, *, config=None, **config_overrides)
```

**Parámetros:**
- `deployment_name`: Nombre del deployment (ej: "gpt-4o")
- `config`: Objeto LLMConfig personalizado (opcional)
- `**config_overrides`: Overrides de configuración

#### Método `generate()`

```python
response = llm.generate(
    prompt,
    *,
    variables=None,
    schema=None,
    batch_items=None,
    max_tokens=None,
    temperature=None
)
```

**Parámetros:**
- `prompt`: Texto del prompt (puede contener {variables})
- `variables`: Dict con valores para sustituir en el prompt
- `schema`: Esquema JSON para salida estructurada
- `batch_items`: Lista de diccionarios para procesamiento por lotes
- `max_tokens`: Límite de tokens
- `temperature`: Nivel de creatividad (0.0-1.0)

**Retorna:**
- String para salida de texto simple
- Dict para salida estructurada con schema
- List para procesamiento por lotes

#### Método `template()`

```python
template = llm.template(template_string, *, schema=None, **fixed_params)
```

Crea una plantilla reutilizable con parámetros fijos.

### Funciones de Conveniencia

#### `quick_generate()`

```python
from utils.llm import quick_generate

response = quick_generate(prompt, deployment_name="gpt-4o", **kwargs)
```

Generación rápida en una línea para casos simples.

#### `create_llm()`

```python
from utils.llm import create_llm

llm = create_llm(deployment_name="gpt-4o", **config_overrides)
```

Factory function para crear instancias de LLM.

### Schema

Manejo de esquemas JSON optimizados para Azure OpenAI.

```python
from utils.llm import Schema

# Desde dict
schema = Schema.load({
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"]
})

# Desde archivo YAML
schema = Schema.load("mi_schema.yaml")
```

## 💡 Tips y Mejores Prácticas

### 🎯 Optimización de Costos

```python
# Usa max_tokens para controlar costos
response = llm.generate(
    "Resume en 50 palabras...", 
    max_tokens=100  # Limita la respuesta
)

# Temperatura baja para respuestas más predecibles
llm = Azure("gpt-4o", temperature=0.1)
```

### 🔒 Esquemas Robustos

```python
# Los esquemas se optimizan automáticamente para Azure
schema = {
    "type": "object",
    "properties": {
        "campo": {"type": "string"}
    }
    # Azure añade automáticamente:
    # "additionalProperties": false
    # "required": ["campo"]
}
```

> **Note:** AzureLLM optimiza automáticamente los esquemas para el modo strict de Azure OpenAI.

### 🔄 Reutilización de Instancias

```python
# ✅ Buena práctica: reutilizar instancia
llm = Azure("gpt-4o")
for texto in textos:
    response = llm.generate(f"Analiza: {texto}")

# ❌ Evitar: crear nueva instancia cada vez
for texto in textos:
    llm = Azure("gpt-4o")  # Ineficiente
    response = llm.generate(f"Analiza: {texto}")
```

### 🧩 Variables en Plantillas

```python
# ✅ Usa variables para prompts dinámicos
template = llm.template("Traduce '{texto}' al {idioma}")
result = template(texto="Hello", idioma="español")

# ❌ Evitar concatenación manual
prompt = f"Traduce '{texto}' al {idioma}"  # Menos flexible
```

### 🚀 Procesamiento Eficiente por Lotes

```python
# ✅ Procesar múltiples items en una sola llamada
items = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]
greetings = llm.generate(
    "Genera un saludo personalizado para cada persona",
    batch_items=items
)

# ❌ Evitar: múltiples llamadas individuales
for item in items:
    greeting = llm.generate(f"Saluda a {item['name']}")  # Ineficiente
```

> **Tip:** El procesamiento por lotes es ideal para transformaciones cortas y consistentes. Para outputs muy largos, usa lotes más pequeños para mantener la calidad.

### Procesador de Datos por Lotes

```python
from utils.llm import Azure

llm = Azure("gpt-4o")

# Datos a procesar
productos = [
    {"nombre": "Laptop", "precio": 999},
    {"nombre": "Mouse", "precio": 29},
    {"nombre": "Teclado", "precio": 79}
]

# Schema para la respuesta
schema = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "categoria": {"type": "string"},
        "rango_precio": {"type": "string"},
        "descripcion_marketing": {"type": "string"}
    }
}

# Procesar todos los productos en una sola llamada
resultados = llm.generate(
    """Para cada producto, determina su categoría, rango de precio 
    (económico/medio/premium) y genera una descripción de marketing atractiva""",
    batch_items=productos,
    schema=schema,
    temperature=0.7
)

# Mostrar resultados
for i, resultado in enumerate(resultados):
    print(f"\nProducto {i+1}:")
    print(f"  Nombre: {resultado['nombre']}")
    print(f"  Categoría: {resultado['categoria']}")
    print(f"  Rango: {resultado['rango_precio']}")
    print(f"  Marketing: {resultado['descripcion_marketing']}")
```

## 🔧 Troubleshooting

### Error: Missing AZURE_OPENAI_ENDPOINT

```bash
ValueError: Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY
```

**Solución:** Verifica que tienes un archivo `.env` con las variables correctas o pasa la configuración explícitamente.

### Error: KeyError en Template

```python
KeyError: Missing template variable: 'nombre'
```

**Solución:** Asegúrate de pasar todas las variables definidas en el template:

```python
template = llm.template("Hola {nombre}")
response = template(nombre="Juan")  # ✅ Variable proporcionada
```

### Respuesta No-JSON con Schema

```
UserWarning: Model returned non-JSON despite schema constraint
```

**Solución:** Esto es normal. AzureLLM maneja automáticamente casos donde el modelo no respeta el esquema y devuelve el texto raw.

### Validación de Schema

```python
# Habilitar validación strict (requiere jsonschema)
pip install jsonschema

llm = Azure("gpt-4o", validate_schema=True)
```

## 📄 Compatibilidad hacia Atrás

AzureLLM v4 mantiene compatibilidad total con versiones anteriores:

```python
# Parámetros antiguos aún funcionan
response = llm.generate(
    "Hola {nombre}",
    prompt_vars={"nombre": "Juan"},    # Ahora se llama 'variables'
    output_schema=schema               # Ahora se llama 'schema'
)
```

> **Note:** Se recomienda usar los nuevos nombres de parámetros para código nuevo.

## 🎉 Ejemplos Completos

### Analizador de Sentimientos

```python
from utils.llm import Azure

llm = Azure("gpt-4o")

analizador = llm.template(
    """Analiza el sentimiento del siguiente texto: "{texto}"
    
    Responde con el sentimiento (positivo/negativo/neutro) y una puntuación de confianza del 0 al 1.""",
    
    schema={
        "type": "object",
        "properties": {
            "sentimiento": {
                "type": "string",
                "enum": ["positivo", "negativo", "neutro"]
            },
            "confianza": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "razon": {"type": "string"}
        },
        "required": ["sentimiento", "confianza", "razon"]
    },
    temperature=0.2
)

# Usar el analizador
textos = [
    "¡Me encanta este producto!",
    "El servicio fue terrible",
    "El clima está nublado hoy"
]

for texto in textos:
    resultado = analizador(texto=texto)
    print(f"Texto: {texto}")
    print(f"Sentimiento: {resultado['sentimiento']} ({resultado['confianza']:.2f})")
    print(f"Razón: {resultado['razon']}\n")
```

---

**AzureLLM v4** - Desarrollado con 💛 para máxima productividad y mínima fricción.
