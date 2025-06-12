# Tests - Suite de Pruebas 🧪

Este directorio contiene las pruebas unitarias y de integración para validar el correcto funcionamiento de los módulos en `utils/`. La filosofía es simple: cada módulo de utilidad tiene su correspondiente suite de tests que garantiza su funcionamiento aislado y en conjunto.

## 🎯 ¿Por qué son importantes los tests?

Los tests en este proyecto sirven para:

1. **Validar funcionalidad**: Asegurar que cada módulo hace lo que promete
2. **Detectar regresiones**: Alertar cuando un cambio rompe funcionalidad existente
3. **Documentar comportamiento**: Los tests son ejemplos ejecutables de cómo usar cada módulo
4. **Facilitar refactoring**: Cambiar implementación con confianza si los tests pasan

## 🏗️ Estructura de Tests

```
tests/
├── README.md           # Este archivo
├── test_src/          # Tests para código fuente principal (si existe)
└── test_utils/        # Tests para módulos de utilidades
    ├── test_bert/
    │   └── test_bert_similarity.py    # Tests del juez semántico
    ├── test_icd10/
    │   └── test_taxonomy.py           # Tests de navegación ICD-10
    └── test_llm/
        └── stress_llm_test.py         # Tests de estrés para LLMs
```

## 🚀 Ejecutar Tests

### Prerequisitos

```bash
# Asegurarse de tener el proyecto instalado
pip install -e .

# Instalar dependencias de desarrollo
pip install -e .[dev]
```

### Comandos básicos

```bash
# Ejecutar TODOS los tests
pytest

# Tests con output detallado
pytest -v

# Tests de un módulo específico
pytest tests/test_utils/test_bert/

# Test específico
pytest tests/test_utils/test_bert/test_bert_similarity.py::test_single_comparison

# Con cobertura de código
pytest --cov=utils --cov-report=html
# Luego abrir htmlcov/index.html en el navegador
```

## 📝 Ejemplos de Tests por Módulo

### Tests de BERT (Juez Semántico)

Los tests en `test_bert_similarity.py` validan:
- Comparación simple entre dos términos
- Comparación múltiple (muchos a muchos)
- Manejo de errores de API
- Warm-up del endpoint
- Scores en rango [0, 1]

```python
def test_medical_terms_similarity():
    """Test que términos médicos relacionados tengan alta similitud."""
    result = calculate_semantic_similarity("diabetes", "hyperglycemia")
    assert result["diabetes"]["hyperglycemia"] > 0.7
```

### Tests de ICD-10

Los tests en `test_taxonomy.py` verifican:
- Búsqueda por código y nombre
- Navegación de jerarquía (padres, hijos, hermanos)
- Conversión código ↔ nombre
- Manejo de casos edge (códigos inválidos)

```python
def test_covid_lookup():
    """Test que podemos encontrar COVID-19 por código o nombre."""
    taxonomy = ICD10Taxonomy()
    by_code = taxonomy.get("U07.1")
    by_name = taxonomy.get("COVID-19")
    assert by_code['name'] == "COVID-19"
    assert by_name['code'] == "U07.1"
```

### Tests de LLM (Stress Testing)

Los tests en `stress_llm_test.py` estresan:
- Generación con diferentes modelos
- Procesamiento en batch
- Manejo de esquemas JSON
- Timeouts y reintentos
- Fallback entre proveedores

```python
def test_batch_processing():
    """Test procesamiento de múltiples items en paralelo."""
    llm = get_llm("gpt-4o")
    items = [{"text": f"Item {i}"} for i in range(10)]
    results = llm.generate("Procesa cada item", batch_items=items)
    assert len(results) == 10
```

## 🧪 Escribir Nuevos Tests

### Estructura básica de un test

```python
import pytest
from utils.mi_modulo import mi_funcion

def test_funcionamiento_basico():
    """Descripción clara de qué se está testeando."""
    # Arrange - Preparar datos
    input_data = "ejemplo"
    
    # Act - Ejecutar función
    result = mi_funcion(input_data)
    
    # Assert - Verificar resultado
    assert result == "resultado esperado"

def test_manejo_de_errores():
    """Test que la función maneja errores correctamente."""
    with pytest.raises(ValueError):
        mi_funcion(None)
```

### Fixtures para datos compartidos

```python
@pytest.fixture
def sample_medical_terms():
    """Términos médicos de ejemplo para tests."""
    return {
        "symptoms": ["fever", "cough", "fatigue"],
        "diseases": ["influenza", "covid-19", "common cold"]
    }

def test_with_fixture(sample_medical_terms):
    """Test usando fixture."""
    symptoms = sample_medical_terms["symptoms"]
    # ... usar los datos en el test
```

## 🎯 Mejores Prácticas

1. **Un test, una cosa**: Cada test debe verificar un solo comportamiento
2. **Nombres descriptivos**: `test_calculate_similarity_with_invalid_input_raises_error`
3. **Independencia**: Los tests no deben depender unos de otros
4. **Rapidez**: Usa mocks para servicios externos cuando sea posible
5. **Cobertura**: Apunta a >80% de cobertura, pero prioriza casos críticos

## 🔧 Configuración de Tests

Los tests respetan las mismas variables de entorno que el código principal:

```env
# .env.test (opcional, para configuración específica de tests)
SAPBERT_API_URL=https://test-endpoint.com
HF_TOKEN=test_token
# ... etc
```

## 📊 Reporte de Cobertura

Para generar un reporte detallado de cobertura:

```bash
# Generar reporte HTML
pytest --cov=utils --cov-report=html

# Generar reporte en terminal
pytest --cov=utils --cov-report=term-missing

# Solo mostrar archivos con <100% cobertura
pytest --cov=utils --cov-report=term-missing --cov-fail-under=80
```

## 🚨 Tests de Integración vs Unitarios

- **Tests Unitarios**: Prueban funciones individuales aisladas
- **Tests de Integración**: Prueban módulos trabajando juntos
- **Tests de Estrés**: Prueban límites y casos extremos

En este proyecto, la mayoría son tests de integración porque los módulos de `utils` interactúan con servicios externos (APIs de HuggingFace, Azure, etc).

## 💡 Tips para Testing

1. **Usa el modo verbose** (`pytest -v`) para debugging
2. **Ejecuta tests frecuentemente** durante el desarrollo
3. **Escribe el test primero** cuando encuentres un bug
4. **Mockea servicios externos** para tests más rápidos y confiables
5. **Revisa la cobertura** pero no te obsesiones con el 100%

## 🔗 Recursos

- [Documentación oficial de pytest](https://docs.pytest.org/)
- [pytest-cov para cobertura](https://pytest-cov.readthedocs.io/)
- [Guía de testing en Python](https://realpython.com/pytest-python-testing/)