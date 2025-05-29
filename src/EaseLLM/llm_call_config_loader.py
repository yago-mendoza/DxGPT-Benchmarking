# file: llm_call_config_loader.py
# Defines the methods to extract fields from a YAML LLM-call configuration
# Agnostic to LLM provider or model (this is instantiated by the EaseLLM class)

# file: ease_llm_client.py
# Imports the OpenAIConfig class from llm_call_config_loader.py
# Uses its methods to extract fields from a YAML LLM-call configuration (needs the path)
# Builds the configuration to call the LLM provider
# Everything is triggered by the "call" method

import yaml
from typing import Dict, Any, Optional, List

class OpenAIConfig:
    """Carga y maneja configuración YAML para llamadas a OpenAI"""
    
    def __init__(self, yaml_path: str):
        """Inicializa con la ruta del archivo YAML"""
        self.yaml_path = yaml_path
        self._config = self._load_yaml()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Carga el archivo YAML"""
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @property
    def json_schema_name(self) -> str:
        """Obtiene el nombre del schema JSON"""
        return self._config['schema_info']['name']
    
    @property
    def json_schema(self) -> Dict[str, Any]:
        """Obtiene el schema JSON completo"""
        return self._config['json_schema']
    
    def get_prompt(self, prompt_id: Optional[str] = None) -> Dict[str, str]:
        """
        Obtiene un prompt específico o el primero por defecto
        
        Args:
            prompt_id: ID del prompt a usar. Si es None, usa el primero.
            
        Returns:
            Dict con 'system' y 'user' prompts
        """
        prompts = self._config['prompts']
        
        if prompt_id is None:
            # Usar el primer prompt
            prompt = prompts[0]
        else:
            # Buscar prompt por ID
            prompt = next((p for p in prompts if p['id'] == prompt_id), None)
            if prompt is None:
                raise ValueError(f"Prompt con ID '{prompt_id}' no encontrado")
        
        return {
            'system': prompt['system'],
            'user': prompt['user']
        }
    
    def get_all_prompt_ids(self) -> List[str]:
        """Lista todos los IDs de prompts disponibles"""
        return [p['id'] for p in self._config['prompts']]
    
    @property
    def generation_params(self) -> Dict[str, Any]:
        """Obtiene parámetros de generación"""
        return self._config.get('generation_params', {})
    
    @property
    def temperature(self) -> float:
        """Obtiene temperatura"""
        return self.generation_params.get('temperature', 0.7)
    
    @property
    def max_tokens(self) -> Optional[int]:
        """Obtiene max_tokens"""
        return self.generation_params.get('max_tokens')
    
    @property
    def model_config(self) -> Dict[str, str]:
        """Obtiene configuración del modelo"""
        return self._config.get('model_config', {})
    
    def get_response_format(self) -> Dict[str, Any]:
        """Construye el response_format para OpenAI"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.json_schema_name,
                "schema": self.json_schema
            }
        }
    
    def get_messages(self, prompt_id: Optional[str] = None, **kwargs) -> List[Dict[str, str]]:
        """
        Construye los mensajes para la API de OpenAI
        
        Args:
            prompt_id: ID del prompt a usar
            **kwargs: Variables para reemplazar en el user prompt
            
        Returns:
            Lista de mensajes formateados para OpenAI
        """
        prompt = self.get_prompt(prompt_id)
        
        # Reemplazar variables en el user prompt si las hay
        user_content = prompt['user']
        if kwargs:
            user_content = user_content.format(**kwargs)
        
        return [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": user_content}
        ]
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        return f"OpenAIConfig(schema='{self.json_schema_name}', prompts={len(self._config['prompts'])})"


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar configuración
    cfg = OpenAIConfig("config_poema.yaml")
    
    # Acceder a las propiedades como pediste
    print(f"Schema name: {cfg.json_schema_name}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Prompts disponibles: {cfg.get_all_prompt_ids()}")
    
    # Obtener prompt por defecto
    prompt = cfg.get_prompt()
    print(f"\nPrompt por defecto:")
    print(f"System: {prompt['system']}")
    print(f"User: {prompt['user']}")
    
    # Obtener prompt específico
    prompt_creativo = cfg.get_prompt("poeta_creativo")
    print(f"\nPrompt 'poeta_creativo':")
    print(f"System: {prompt_creativo['system']}")
    
    # Obtener response_format listo para usar
    response_format = cfg.get_response_format()
    print(f"\nResponse format: {response_format}") 