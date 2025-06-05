"""
AzureLLM v4 - Wrapper de alto nivel para Azure OpenAI API

Este módulo proporciona una interfaz unificada para trabajar con modelos de Azure OpenAI,
incluyendo soporte para plantillas, esquemas de salida estructurada y simplicidad directa.

Uso típico:
    from utils.llm import Azure
    
    # Configuración automática desde .env
    llm = Azure("gpt-4o")
    response = llm.generate("Hola, ¿cómo estás?")
"""

from .azure import (
    # Clase principal y alias limpio
    AzureLLM,
    Azure,
    
    # Configuración y esquemas
    LLMConfig,
    Schema,
    
    # Sistema de plantillas
    Template,
    
    # Procesamiento por lotes
    BatchProcessor,
    
    # Funciones de conveniencia
    create_llm,
    quick_generate
)

__version__ = "4.0.0"
__author__ = "AzureLLM Team"

__all__ = [
    # Interfaz principal (lo más usado)
    "Azure",
    "AzureLLM",
    
    # Configuración avanzada
    "LLMConfig",
    "Schema",
    
    # Plantillas
    "Template",
    
    # Batch processing
    "BatchProcessor",
    
    # Conveniencia
    "create_llm",
    "quick_generate"
]
