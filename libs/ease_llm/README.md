# EaseLLM: Unified LLM Client

EaseLLM offers a streamlined way to interact with various LLM providers using a single interface. It emphasizes clear YAML-based prompt configuration and secure environment variable management.

**Design Philosophy:**
*   **Simplicity:** One client, multiple providers. The `model` parameter in `EaseLLM(provider, model)` directly maps to environment variable prefixes.
*   **Configuration over Code:** Prompts and parameters defined in YAML, not hardcoded.
*   **Security & Flexibility:** API credentials and endpoints managed via `.env` files.
*   **Structured Output Focus:** Designed for LLMs supporting JSON schema responses (e.g., OpenAI's JSON mode).

## Use
```python
from ease_llm_client import EaseLLM
llm = EaseLLM(provider="azure", model="o1")  # o "huggingface"
result = llm.call("prompt_config.yaml")

# Con prompt específico
result = llm.call("doc_analyzer.yaml", 
                                                  # SUB-PROMPT SELECTION
                  prompt_id="quick_analysis",
                                                  # PROMPT PLACEHOLDERS
                  specialty="documentación API",
                  topic="REST endpoints")
```

The argument `model` determines which environment variables to look for: `{PROVIDER}_{MODEL}_*`

## YAML Config

* **Format**: See `yaml_config_guide/YAML_CONFIG_GUIDE.md`
* **Processing**: `llm_call_config_loader.py` extracts fields from the YAML and converts them into an attribute-based object ready for OpenAI calls

## 1. Supported Integrations

### Providers
*   **`azure`**: Azure OpenAI Service.
*   **`huggingface`**: Hugging Face Inference Endpoints (or any other OpenAI-compatible API that only requires an API Key and Endpoint URL).

### Models & Endpoints (`model` slug)
The `model` slug you provide to `EaseLLM(provider="...", model="<slug>")` is directly used to find the corresponding environment variables.

*   **Azure OpenAI (`provider="azure"`):**
    *   The `model` slug (e.g., `"o1"`, `"gpt-4o"`) defines the prefix for environment variables.
    *   Required Env Vars:
        *   `AZURE_<MODEL_SLUG_UPPERCASE>_API_KEY`
        *   `AZURE_<MODEL_SLUG_UPPERCASE>_ENDPOINT`
        *   `AZURE_<MODEL_SLUG_UPPERCASE>_API_VERSION`
        *   `AZURE_<MODEL_SLUG_UPPERCASE>_DEPLOYMENT` (This is the actual model deployment name on Azure).

*   **Hugging Face Endpoints & OpenAI-compatible APIs (`provider="huggingface"`):**
    *   The `model` slug (e.g., `"jsl"`, `"gpt-4o"`, `"mixtral_inference"`) defines the prefix.
    *   Required Env Vars:
        *   `HF_<MODEL_SLUG_UPPERCASE>_API_KEY`
        *   `HF_<MODEL_SLUG_UPPERCASE>_ENDPOINT`
    *   Optional Env Var:
        *   `HF_<MODEL_SLUG_UPPERCASE>_MODEL`: If the endpoint requires a specific model name to be passed in the API call (beyond what's implied by the endpoint URL itself). If not set, `EaseLLM` uses a sensible default (like "default" or the slug itself, but it's best to set it if the API needs it).
    *   **Note for OpenAI API:** To use the official OpenAI API, set `provider="huggingface"`, use a slug like `"gpt-4o"`, and configure:
        *   `HF_GPT_4O_API_KEY="sk-your_openai_api_key"`
        *   `HF_GPT_4O_ENDPOINT="https://api.openai.com/v1"`
        *   `HF_GPT_4O_MODEL="gpt-4o"` (or other OpenAI model ID like "gpt-3.5-turbo")

#### Add New Endpoint

1. **Environment Variables**: `{PROVIDER}_{MODEL}_API_KEY`, `{PROVIDER}_{MODEL}_ENDPOINT`, etc.
2. **Clause in `ease_llm_client.py`**: Implement `_init_{provider}()` method to configure the client.
3. **OpenAI Schema Compatibility**: Endpoints must follow the OpenAI API schema (otherwise, refactor `ease_llm_client.py` and `llm_call_config_loader.py`).

The OpenAIConfig class at `llm_call_config_loader.py` and the YAML structure it expects are designed with OpenAI's API capabilities in mind, particularly the response_format for JSON schema enforcement. If a provider has a different way of handling structured output or its API call parameters differ significantly, adjustments in the EaseLLM.call() method will be necessary for that provider.

## 2. Getting Started

### Installation
1.  Create and activate a virtual environment.
2.  Install:
    ```bash
    pip install openai pyyaml python-dotenv
    ```

### `.env` Configuration
Create a `.env` file in your project root. `EaseLLM` loads this automatically.

**Example `.env` file:**
```env
# --- Azure OpenAI Deployments (provider="azure") ---
# Model slug: "o1"
AZURE_O1_API_KEY="your_azure_api_key_for_o1_deployment"
AZURE_O1_ENDPOINT="https://your-o1-instance.openai.azure.com/"
AZURE_O1_API_VERSION="2024-02-01" # Or your API version
AZURE_O1_DEPLOYMENT="o1-deployment-name" # Actual deployment name for "o1"

# Model slug: "gpt-4o" (for an Azure deployment of GPT-4o)
AZURE_GPT_4O_API_KEY="your_azure_api_key_for_gpt4o_deployment"
AZURE_GPT_4O_ENDPOINT="https://your-gpt4o-instance.openai.azure.com/"
AZURE_GPT_4O_API_VERSION="2024-02-01"
AZURE_GPT_4O_DEPLOYMENT="gpt-4o" # Actual deployment name for "gpt-4o"

# --- Hugging Face Endpoints & OpenAI-compatible APIs (provider="huggingface") ---
# Model slug: "jsl" (for a Hugging Face Inference Endpoint)
HF_JSL_API_KEY="hf_your_hugging_face_token_for_jsl_endpoint"
HF_JSL_ENDPOINT="https://jsl-endpoint.huggingface.cloud"
# HF_JSL_MODEL: Optional, e.g., "meta-llama/Llama-2-7b-chat-hf" if needed by the endpoint

# Model slug: "gpt-4o" (for official OpenAI API)
HF_GPT_4O_API_KEY="sk-your_openai_api_key"
HF_GPT_4O_ENDPOINT="https://api.openai.com/v1"
HF_GPT_4O_MODEL="gpt-4o" # OpenAI model ID

# Model slug: "claude3opus" (example for an Anthropic API via compatible gateway)
# HF_CLAUDE3OPUS_API_KEY="your_anthropic_or_gateway_key"
# HF_CLAUDE3OPUS_ENDPOINT="https://your-anthropic-compatible-gateway.com/v1"
# HF_CLAUDE3OPUS_MODEL="claude-3-opus-20240229"
```

## 3. How to write the YAML?

Define prompts in YAML files.

The `OpenAIConfig` class (used internally by `EaseLLM`) parses these. The YAML structure leverages OpenAI's `response_format` with `json_schema` for structured output. This format is compatible with Azure OpenAI, official OpenAI, and many Hugging Face endpoints.

**Key YAML Sections:**
*   `schema_info`: Metadata for the JSON schema (`name`, `description`).
*   `json_schema`: The JSON Schema definition for the LLM's output (see [json-schema.org](https://json-schema.org/)).
*   `prompts`: A list of prompt templates. Each entry has:
    *   `id`: Unique identifier for the prompt.
    *   `system`: System message to guide the LLM.
    *   `user`: User message template (can include `{placeholders}`).
*   `generation_params`: Default LLM generation settings (e.g., `temperature`, `max_tokens`, `top_p`).


```yaml
# ========================================
# SISTEMA DE ANÁLISIS DE DOCUMENTOS TÉCNICOS
# ========================================
# Este YAML muestra TODAS las características posibles con comentarios

# ----------------------------------------
# INFORMACIÓN DEL SCHEMA (OBLIGATORIO: name)
# ----------------------------------------
schema_info:
  name: "doc_analyzer_pro"           # OBLIGATORIO: identificador único
  description: "Analiza documentos"  # OPCIONAL: descripción del schema
  version: "4.2.1"                   # OPCIONAL: versionado semántico
  author: "Tech Team"                # OPCIONAL: metadata custom
  created: "2024-01-15"              # OPCIONAL: cualquier campo extra
  tags: ["nlp", "analysis"]          # OPCIONAL: arrays también permitidos

# ----------------------------------------
# JSON SCHEMA (TODO OBLIGATORIO)
# ----------------------------------------
json_schema:                         # OBLIGATORIO: bloque completo
  type: "object"                     # OBLIGATORIO: siempre "object" para structured output
  properties:                        # OBLIGATORIO: define campos de salida
    
    # === STRINGS CON TODAS LAS RESTRICCIONES ===
    document_id:
      type: "string"                 # OBLIGATORIO: tipo de dato
      description: "ID único"        # OPCIONAL pero RECOMENDADO: ayuda al modelo
      pattern: "^DOC-[0-9]{8}$"      # OPCIONAL: regex (DOC-12345678)
      examples: ["DOC-12345678"]     # OPCIONAL: ejemplos para el modelo
    
    title:
      type: "string"
      minLength: 10                  # OPCIONAL: mínimo caracteres
      maxLength: 200                 # OPCIONAL: máximo caracteres
      pattern: "^[A-Z].*[.]$"        # OPCIONAL: empieza mayúscula, termina punto
      
    category:
      type: "string"
      enum: ["tech", "legal", "medical", "financial"]  # OPCIONAL: valores forzados
      description: "Categoría principal"
      
    language:
      type: "string"
      enum: ["es", "en", "fr", "pt"]
      default: "es"                  # OPCIONAL: valor por defecto
      
    # === STRINGS CON FORMATOS ESPECIALES ===
    created_date:
      type: "string"
      format: "date"                 # OPCIONAL: YYYY-MM-DD
      
    author_email:
      type: "string"
      format: "email"                # OPCIONAL: valida email
      pattern: "^[^@]+@company\\.com$"  # OPCIONAL: solo emails corporativos
      
    source_url:
      type: ["string", "null"]       # OPCIONAL: puede ser string O null
      format: "uri"                  # OPCIONAL: valida URL
      
    # === NÚMEROS CON VALIDACIONES ===
    word_count:
      type: "integer"
      minimum: 100                   # OPCIONAL: valor mínimo (incluido)
      maximum: 50000                 # OPCIONAL: valor máximo (incluido)
      
    readability_score:
      type: "number"                 # number = acepta decimales
      minimum: 0.0
      maximum: 100.0
      multipleOf: 0.1                # OPCIONAL: precisión decimal (0.1, 0.2, etc)
      
    version_number:
      type: "integer"
      minimum: 1
      exclusiveMinimum: false        # OPCIONAL: si incluye el límite (default: false)
      
    sections_count:
      type: "integer"
      multipleOf: 2                  # OPCIONAL: debe ser par (múltiplo de 2)
      
    # === BOOLEANOS ===
    is_published:
      type: "boolean"
      default: false                 # OPCIONAL: valor por defecto
      
    requires_review:
      type: "boolean"
      description: "Necesita revisión humana"
      
    # === ARRAYS SIMPLES ===
    keywords:
      type: "array"
      description: "Palabras clave"
      items:                         # OBLIGATORIO si type="array"
        type: "string"
        pattern: "^[a-z]+$"          # OPCIONAL: solo minúsculas
        minLength: 3                 # OPCIONAL: min caracteres por item
        maxLength: 20                # OPCIONAL: max caracteres por item
      minItems: 3                    # OPCIONAL: mínimo elementos
      maxItems: 10                   # OPCIONAL: máximo elementos
      uniqueItems: true              # OPCIONAL: no duplicados
      
    priority_levels:
      type: "array"
      items:
        type: "string"
        enum: ["low", "medium", "high", "critical"]  # items con enum
      minItems: 1
      
    # === ARRAYS DE OBJETOS ===
    sections:
      type: "array"
      description: "Secciones del documento"
      items:
        type: "object"               # array de objetos
        properties:
          section_id:
            type: "string"
            pattern: "^SEC-[0-9]{3}$"
          title:
            type: "string"
            minLength: 5
          content:
            type: "string"
            minLength: 100
            maxLength: 5000
          subsections:               # array dentro de objeto
            type: "array"
            items:
              type: "object"
              properties:
                subtitle:
                  type: "string"
                body:
                  type: "string"
                  minLength: 50
              required: ["subtitle", "body"]
            maxItems: 5
          metadata:                  # objeto dentro de objeto en array
            type: "object"
            properties:
              author:
                type: "string"
              last_modified:
                type: "string"
                format: "date-time"
            additionalProperties: true  # permite campos extra en metadata
        required: ["section_id", "title", "content"]  # requeridos del item
        additionalProperties: false  # no permite campos extra en section
      minItems: 1
      maxItems: 20
      
    # === OBJETOS ANIDADOS COMPLEJOS ===
    analysis_results:
      type: "object"
      description: "Resultados del análisis"
      properties:
        
        # Objeto con propiedades mixtas
        summary_stats:
          type: "object"
          properties:
            total_words:
              type: "integer"
              minimum: 0
            unique_words:
              type: "integer"
              minimum: 0
            avg_sentence_length:
              type: "number"
              minimum: 0
              multipleOf: 0.01
            complexity_metrics:      # sub-objeto
              type: "object"
              properties:
                flesch_score:
                  type: "number"
                  minimum: 0
                  maximum: 100
                grade_level:
                  type: "integer"
                  minimum: 1
                  maximum: 20
                fog_index:
                  type: "number"
                  minimum: 0
              required: ["flesch_score"]
          required: ["total_words", "unique_words"]
          
        # Array de objetos con enum
        detected_topics:
          type: "array"
          items:
            type: "object"
            properties:
              topic:
                type: "string"
                enum: ["AI", "blockchain", "cloud", "security", "data", "other"]
              confidence:
                type: "number"
                minimum: 0
                maximum: 1
                multipleOf: 0.01
              subtopics:
                type: "array"
                items:
                  type: "string"
                maxItems: 5
            required: ["topic", "confidence"]
          minItems: 1
          maxItems: 10
          
      required: ["summary_stats", "detected_topics"]
      additionalProperties: false    # estricto: no campos extra
      
    # === TIPOS MIXTOS Y CASOS ESPECIALES ===
    revision_info:
      oneOf:                         # OPCIONAL: uno de estos tipos
        - type: "null"               # puede ser null
        - type: "object"             # O un objeto
          properties:
            revision_number:
              type: "integer"
              minimum: 1
            revised_by:
              type: "string"
            revision_date:
              type: "string"
              format: "date"
          required: ["revision_number", "revised_by"]
          
    external_references:
      type: ["array", "null"]        # puede ser array O null
      items:
        type: "string"
        format: "uri"
      
    custom_metadata:
      type: "object"
      additionalProperties:          # permite cualquier propiedad
        type: ["string", "number", "boolean"]  # pero solo estos tipos
      
    # === PROPIEDADES CON DEPENDENCIAS ===
    publication_details:
      type: "object"
      properties:
        published:
          type: "boolean"
        publish_date:
          type: "string"
          format: "date"
        publisher:
          type: "string"
        isbn:
          type: "string"
          pattern: "^978-[0-9]{10}$"
      # Si published=true, entonces publish_date y publisher son requeridos
      dependencies:
        published: ["publish_date", "publisher"]
      
  required:                          # OBLIGATORIO: campos que SIEMPRE deben estar
    - "document_id"
    - "title"
    - "category"
    - "word_count"
    - "is_published"
    - "keywords"
    - "sections"
    - "analysis_results"
    
  additionalProperties: false        # OPCIONAL: no permite campos no definidos

# ----------------------------------------
# PROMPTS (OBLIGATORIO: mínimo uno con id="default")
# ----------------------------------------
prompts:                             # OBLIGATORIO: bloque completo
  
  # Prompt simple sin variables (OBLIGATORIO tener uno "default")
  - id: "default"                    # OBLIGATORIO: id="default"
    system: "Eres un analizador de documentos técnicos experto."  # OBLIGATORIO
    user: "Analiza el siguiente documento técnico en detalle."     # OBLIGATORIO
    
  # Prompt con una variable
  - id: "quick_analysis"
    system: "Eres un analizador rápido especializado en {specialty}."
    user: "Analiza rápidamente este documento sobre {topic}."
    
  # Prompt con múltiples variables
  - id: "custom_analysis"
    system: |
      Eres un analizador de documentos {doc_type} con experiencia en {industry}.
      Tu nivel de detalle debe ser {detail_level}.
      Considera el contexto de {context}.
    user: |
      Analiza el siguiente documento:
      
      Título: {title}
      Contenido: {content}
      
      Requisitos específicos:
      - Idioma de salida: {output_language}
      - Enfoque en: {focus_areas}
      - Excluir: {exclude_topics}
      
      Consideraciones adicionales: {additional_notes}
      
  # Prompt multilinea complejo
  - id: "comparative_analysis"
    system: |
      Eres un experto en análisis comparativo de documentos técnicos.
      
      Especialidades:
      - Identificación de patrones
      - Detección de inconsistencias
      - Evaluación de calidad
      
      Debes ser objetivo y detallado en tu análisis.
    user: |
      Compara y analiza estos documentos:
      
      Documento Principal:
      """
      {main_document}
      """
      
      Documentos de Referencia:
      {reference_docs}
      
      Criterios de comparación: {criteria}

# ----------------------------------------
# PARÁMETROS DE GENERACIÓN (TODO OPCIONAL)
# ----------------------------------------
generation_params:                   # OPCIONAL: todo el bloque
  temperature: 0.3                   # OPCIONAL: 0=determinista, 1=creativo
  max_tokens: 3000                   # OPCIONAL: null=sin límite
  top_p: 0.95                        # OPCIONAL: nucleus sampling
  frequency_penalty: 0.2             # OPCIONAL: penaliza repetición
  presence_penalty: 0.1              # OPCIONAL: fomenta variedad
  stop: ["\\n\\n\\n", "---"]         # OPCIONAL: secuencias de parada
  n: 1                               # OPCIONAL: número de respuestas
  logprobs: false                    # OPCIONAL: log probabilities
  echo: false                        # OPCIONAL: eco del prompt

# REGLAS CLAVE (knowledge as a context for LLMs):
# - required fields no pueden tener "default"
# - si type="array" entonces "items" es OBLIGATORIO
# - enum nunca puede estar vacío
# - pattern + enum = validación robusta
# - additionalProperties: false = respuesta estricta
# - minX debe ser <= maxX siempre
# - multipleOf solo aplica a number/integer
# - format solo aplica a strings
```