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

# Import base class
from .base import BaseLLM

# Import all implementations
from .azure import (
    Azure, 
    AzureLLM,
    LLMConfig,
    Schema,
    create_llm as create_azure_llm,
    quick_generate as quick_generate_azure
)

from .hugging import (
    Hugging,
    HuggingLLM,
    LLMConfig as HuggingLLMConfig,
    Schema as HuggingSchema,
    create_llm as create_hugging_llm,
    quick_generate as quick_generate_hugging
)

def get_llm(model_name: str, **kwargs) -> BaseLLM:
    """Try each LLM provider until one works."""
    providers = [HuggingLLM, AzureLLM]
    
    for provider in providers:
        try:
            return provider(model_name, **kwargs)
        except Exception as e:
            continue
            
    raise RuntimeError(f"No working LLM provider found for model {model_name}")

# Default implementations
def create_llm(deployment_name: str = None, **config_overrides) -> BaseLLM:
    """Create LLM instance trying different providers."""
    return get_llm(deployment_name or "default", **config_overrides)

def quick_generate(prompt: str, deployment_name: str = "default", **kwargs):
    """Quick one-shot generation."""
    llm = create_llm(deployment_name)
    return llm.generate(prompt, **kwargs)

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

    # Hugging exports
    'Hugging',
    'HuggingLLM',
    
    # Base class if available
    'BaseLLM',

    # Get LLM
    'get_llm',
]