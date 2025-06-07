from typing import Dict, Any
from utils.llm import AzureLLM
import yaml
import os
import importlib.util
from pathlib import Path


def load_schema_from_path(schema_path: str):
    """
    Load a schema class from a Python file path.
    Format: "path/to/file.py:ClassName"
    """
    if ':' in schema_path:
        file_path, class_name = schema_path.split(':')
        
        # Get absolute path relative to experiment directory
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Load the module
        spec = importlib.util.spec_from_file_location("schema_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class
        schema_class = getattr(module, class_name)
        
        # Check if there's a dict version available (e.g., SEVERITY_SCHEMA_DICT)
        dict_name = f"{class_name.replace('Schema', '').upper()}_SCHEMA_DICT"
        if hasattr(module, dict_name):
            return getattr(module, dict_name)
        
        # Convert to dict format if it's a dataclass
        if hasattr(schema_class, '__dataclass_fields__'):
            # Create schema dict from dataclass
            schema_dict = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
            
            for field_name, field_info in schema_class.__dataclass_fields__.items():
                field_type = field_info.type
                
                # Simple type mapping
                if field_type == str:
                    schema_dict["properties"][field_name] = {"type": "string"}
                elif field_type == int:
                    schema_dict["properties"][field_name] = {"type": "integer"}
                elif field_type == float:
                    schema_dict["properties"][field_name] = {"type": "number"}
                elif field_type == bool:
                    schema_dict["properties"][field_name] = {"type": "boolean"}
                elif hasattr(field_type, '__origin__'):
                    # Handle List[str] etc
                    if field_type.__origin__ == list:
                        item_type = field_type.__args__[0]
                        if item_type == str:
                            schema_dict["properties"][field_name] = {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                    elif field_type.__origin__ == dict:
                        # Handle Dict[str, str] for severities
                        key_type, value_type = field_type.__args__
                        if key_type == str and value_type == str:
                            schema_dict["properties"][field_name] = {
                                "type": "object",
                                "additionalProperties": {"type": "string"}
                            }
                        else:
                            schema_dict["properties"][field_name] = {"type": "object"}
                
                # All fields are required by default
                schema_dict["required"].append(field_name)
            
            return schema_dict
        else:
            # If it's already a dict, return it
            return schema_class
    else:
        raise ValueError(f"Invalid schema path format: {schema_path}. Expected 'file.py:ClassName'")


def create_llms_from_config(config_path: str) -> Dict[str, AzureLLM]:
    """
    Create LLM instances based on config.yaml
    Returns a dictionary of named LLM instances.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    llms = {}
    
    for name, llm_config in config['llm_configs'].items():
        model = llm_config['model']
        params = llm_config.get('params', {})
        
        # Separate config params from runtime params
        config_params = {}
        runtime_params = {}
        
        # These are valid LLMConfig parameters
        valid_config_params = {'temperature', 'validate_schema', 'endpoint', 'api_key', 'api_version'}
        
        for key, value in params.items():
            if key in valid_config_params:
                config_params[key] = value
            else:
                # Store runtime params like max_tokens separately
                runtime_params[key] = value
        
        # Create LLM instance with only valid config params
        llm = AzureLLM(
            deployment_name=model,
            **config_params
        )
        
        # Store additional metadata on the LLM instance
        llm._experiment_config = {
            'prompt_path': llm_config.get('prompt'),
            'output_schema': llm_config.get('output_schema'),
            'runtime_params': runtime_params  # Store runtime params for later use
        }
        
        llms[name] = llm
    
    return llms


def get_llm_prompt(llm: AzureLLM) -> str:
    """
    Load the prompt template for an LLM from its configured path.
    """
    prompt_path = llm._experiment_config.get('prompt_path')
    if not prompt_path:
        raise ValueError(f"No prompt path configured for this LLM")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_llm_schema(llm: AzureLLM) -> Dict[str, Any]:
    """
    Get the output schema for an LLM.
    """
    schema_path = llm._experiment_config.get('output_schema')
    if not schema_path:
        return None
    
    return load_schema_from_path(schema_path)