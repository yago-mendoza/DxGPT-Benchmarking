"""
__init__

AzureLLM v4 - Wrapper de alto nivel para Azure OpenAI API

Este módulo proporciona una interfaz unificada para trabajar con modelos de Azure OpenAI,
incluyendo soporte para plantillas, esquemas de salida estructurada y simplicidad directa.

Uso típico:
    from utils.llm import Azure
    
    # Configuración automática desde .env
    llm = Azure("gpt-4o")
    response = llm.generate("Hola, ¿cómo estás?")
"""

# Import all implementations
from .azure import (
    Azure, 
    AzureLLM,
    LLMConfig,
    Schema,
    create_llm as create_azure_llm,
    quick_generate as quick_generate_azure
)

# Export all
__all__ = [
    # Azure exports
    'Azure',
    'AzureLLM',
    'LLMConfig',
    'Schema',
    
    # Factory functions
    'create_llm',
    'quick_generate',
    
    # Base class if available
    'BaseLLM',
]