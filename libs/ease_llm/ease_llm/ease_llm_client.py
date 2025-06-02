# file: llm_call_config_loader.py
# Defines the methods to extract fields from a YAML LLM-call configuration
# Agnostic to LLM provider or model (this is instantiated by the EaseLLM class)

# file: ease_llm_client.py
# Imports the OpenAIConfig class from llm_call_config_loader.py
# Uses its methods to extract fields from a YAML LLM-call configuration (needs the path)
# Builds the configuration to call the LLM provider
# Everything is triggered by the "call" method

"""
ease_llm_client.py
─────────────────
Uso:

    from ease_llm_client import EaseLLM

    llm = EaseLLM(provider="azure", model="o1") # o "gpt-4o"
    result = llm.call("config_poema.yaml")
"""
from __future__ import annotations
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from typing import Dict, Any, Optional
import os, json, pathlib

from llm_call_config_loader import OpenAIConfig   #  👈 tu módulo intacto

load_dotenv()   # lee .env una sola vez al importar

# --------------------------------------------------------------------------- #
# ••• utilidades pequeñas •••
# --------------------------------------------------------------------------- #
def _required(name: str, value: Optional[str]) -> str:
    """Lanza un error claro si falta la variable de entorno `name`."""
    if not value:
        raise RuntimeError(f"Variable de entorno {name} no definida")
    return value


def _env(prefix: str, field: str) -> str:
    """Construye AZURE_4O_API_KEY a partir de prefix='AZURE_4O' y field='API_KEY'."""
    return f"{prefix}_{field}"


# --------------------------------------------------------------------------- #
# ••• clase principal •••
# --------------------------------------------------------------------------- #
class EaseLLM:
    """
    Cliente unificado:
        provider = "azure" | "huggingface"
        model    = slug del deployment (ej. "o1", "gpt-4o") o nombre del modelo (HF)
    """

    # ...................... #
    #  constructor
    # ...................... #
    def __init__(self, provider: str, model: str | None = None):
        provider = provider.lower().strip()

        if provider not in ("azure", "huggingface"):
            raise ValueError(f"Proveedor desconocido: {provider}")

        self.provider = provider
        self.model    = (model or "").upper().replace("-", "_")      # "gpt-4o" -> "GPT_4O" (in order to find the env variables)

        # Construimos cliente según proveedor
        if provider == "azure":
            self._init_azure()
        else:                                          # huggingface
            self._init_huggingface()

    # ...................... #
    #  privados de init
    # ...................... #
    def _init_azure(self):
        if not self.model:
            raise ValueError("Para Azure debes indicar model (p. ej. 'o1')")

        prefix = f"AZURE_{self.model}"

        self._client = AzureOpenAI(
            api_key        = _required(_env(prefix, "API_KEY"),
                                        os.getenv(_env(prefix, "API_KEY"))),
            azure_endpoint = _required(_env(prefix, "ENDPOINT"),
                                        os.getenv(_env(prefix, "ENDPOINT"))),
            api_version    = _required(_env(prefix, "API_VERSION"),
                                        os.getenv(_env(prefix, "API_VERSION")))
        )
        # en Azure el "modelo" se llama "deployment"
        self._model_name = _required(_env(prefix, "DEPLOYMENT"),
                                     os.getenv(_env(prefix, "DEPLOYMENT")))

    def _init_huggingface(self):
        if not self.model:
            raise ValueError("Para HuggingFace debes indicar model (p. ej. 'jsl')")

        prefix = f"HF_{self.model}"

        self._client = OpenAI(
            api_key  = _required(_env(prefix, "API_KEY"),
                                os.getenv(_env(prefix, "API_KEY"))),
            base_url = _required(_env(prefix, "ENDPOINT"),
                                os.getenv(_env(prefix, "ENDPOINT")))
        )
        self._model_name = (os.getenv(_env(prefix, "MODEL"))
                        or "default") 

    # ...................... #
    #  método público
    # ...................... #
    def call(self,
             yaml_path: str | os.PathLike,
             prompt_id: str | None = None,
             **prompt_vars) -> Dict[str, Any]:
        """
        Ejecuta la llamada usando tu mismo YAML.

        Args:
            yaml_path    – ruta de config YAML.
            prompt_id    – id del prompt (o None = primero).
            **prompt_vars– variables para .format del user-prompt.
        """
        cfg = OpenAIConfig(pathlib.Path(yaml_path))

        # El modelo ya está configurado en el constructor (Azure o HF)
        model_name = self._model_name
        if model_name is None:
            raise RuntimeError(
                "No se encontró modelo configurado. Para Azure usa model, "
                "para HuggingFace indica model o define HF_MODEL en el entorno")

        response = self._client.chat.completions.create(
            model           = model_name,
            messages        = cfg.get_messages(prompt_id, **prompt_vars),
            response_format = cfg.get_response_format(),
            temperature     = cfg.temperature,
            max_tokens      = cfg.max_tokens,
            **{k: v for k, v in cfg.generation_params.items()
               if k not in ("temperature", "max_tokens") and v is not None}
        )

        # structured-output: choices[0].message.content ya es JSON string
        return json.loads(response.choices[0].message.content)
