"""
Paquete ease_llm

[!] Permite importar "from ease_llm import EaseLLM, OpenAIConfig"

~~~~~~~~~~~~~~~~
Contiene:

- EaseLLM … cliente unificado (Azure / HuggingFace)
- OpenAIConfig … loader de archivos YAML de prompt
"""

from .ease_llm_client import EaseLLM          # noqa: F401
from .llm_call_config_loader import OpenAIConfig   # noqa: F401

__all__: list[str] = ["EaseLLM", "OpenAIConfig"]