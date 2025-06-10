"""
FILE : hugging.py
HuggingLLM v1 - Clean, powerful wrapper for Hugging Face TGI endpoints
Drop-in replacement for AzureLLM with identical interface
"""

import os
import warnings
import json
import yaml
import requests
from typing import Dict, Any, Optional, Union, Callable, List
from dataclasses import dataclass, field
from functools import cached_property
from .base import BaseLLM


# === Core Data Structures (identical to azure.py) ===

@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration for Hugging Face client."""
    endpoint: str
    api_key: str
    api_version: str = "v1"  # Not used by HF, kept for compatibility
    deployment_name: Optional[str] = None
    temperature: Optional[float] = None
    validate_schema: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, **overrides) -> 'LLMConfig':
        """Create config from environment variables with optional overrides."""
        # Try importing dotenv for convenience
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Get deployment name to construct endpoint
        deployment_name = overrides.get('deployment_name', 'default')
        
        # Construct endpoint env var name based on model
        endpoint_var = f"{deployment_name.upper()}_ENDPOINT_URL"
        endpoint = overrides.get('endpoint') or os.getenv(endpoint_var)
        
        # Get HF token
        api_key = overrides.get('api_key') or os.getenv("HF_TOKEN")
        
        if not endpoint or not api_key:
            raise ValueError(
                f"Missing {endpoint_var} or HF_TOKEN. "
                "Set them in environment or pass explicitly."
            )
        
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            **{k: v for k, v in overrides.items() if k not in ('endpoint', 'api_key', 'deployment_name')}
        )


@dataclass(frozen=True)
class Schema:
    """Immutable schema wrapper with TGI grammar support."""
    data: Dict[str, Any]
    
    @classmethod
    def load(cls, source: Union[Dict[str, Any], str]) -> 'Schema':
        """Load schema from dict or YAML file."""
        if isinstance(source, dict):
            data = source.copy()
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Schema file not found: {source}")
            
            with open(source, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        
        return cls(cls._optimize_for_tgi(data))
    
    @staticmethod
    def _optimize_for_tgi(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize schema for TGI (no changes needed, TGI handles standard JSON Schema)."""
        return schema
    
    def validate_if_enabled(self, enabled: bool) -> None:
        """Validate schema structure if validation is enabled."""
        if not enabled:
            return
            
        try:
            import jsonschema
            jsonschema.Draft7Validator.check_schema(self.data)
        except ImportError:
            warnings.warn("jsonschema not installed. Install for schema validation.", UserWarning)
        except Exception as e:
            raise ValueError(f"Invalid JSON schema: {e}")
    
    @property
    def tgi_format(self) -> Dict[str, Any]:
        """Format for TGI grammar constraint."""
        return {
            "type": "json",
            "value": self.data
        }


# === Batch Processing Support ===

class BatchProcessor:
    """Handles batch item formatting for prompts."""
    
    @staticmethod
    def format_batch_items(batch_items: List[Dict[str, Any]]) -> str:
        """Convert batch items to clean JSON format for prompt inclusion."""
        return json.dumps(batch_items, indent=2, ensure_ascii=False)
    
    @staticmethod
    def wrap_schema_for_batch(original_schema: Optional[Schema]) -> Schema:
        """Wrap a schema to handle batch responses."""
        if original_schema is None:
            # If no schema, create a simple array schema wrapped in object
            batch_schema = {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["results"],
                "additionalProperties": False
            }
        else:
            # Wrap the original schema in an array inside an object
            batch_schema = {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": original_schema.data
                    }
                },
                "required": ["results"],
                "additionalProperties": False
            }
        
        return Schema.load(batch_schema)


# === Request Building ===

class RequestBuilder:
    """Builds Hugging Face TGI requests."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.batch_processor = BatchProcessor()
    
    def build(
        self,
        prompt: str,
        schema: Optional[Schema] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        batch_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Build complete request parameters for TGI."""
        # Handle batch processing
        if batch_items is not None:
            batch_json = self.batch_processor.format_batch_items(batch_items)
            prompt = f"{prompt}\n\nProcess the following items:\n{batch_json}\n\nReturn the results in a JSON object with a 'results' array containing the processed items."
            
            # Wrap schema for batch response
            if schema is not None:
                schema = self.batch_processor.wrap_schema_for_batch(schema)
            else:
                # Even without schema, we need structured output for batch
                schema = self.batch_processor.wrap_schema_for_batch(None)
        
        # Build TGI request
        request = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or 1000,
                "temperature": temperature if temperature is not None else self.config.temperature or 0.7,
            }
        }
        
        # Add grammar constraint if schema provided
        if schema is not None:
            request["parameters"]["grammar"] = schema.tgi_format
            # Lower temperature for structured output
            request["parameters"]["temperature"] = min(request["parameters"]["temperature"], 0.3)
        
        return request


# === Response Processing ===

class ResponseProcessor:
    """Handles response parsing from TGI."""
    
    @staticmethod
    def process(response_json: Dict[str, Any], expect_json: bool = False, is_batch: bool = False) -> Union[str, Dict[str, Any], List[Any]]:
        """Process TGI response with graceful JSON parsing."""
        # TGI response format: {"generated_text": "..."}
        if isinstance(response_json, dict) and "generated_text" in response_json:
            content = response_json["generated_text"]
        else:
            # Some TGI configurations might return the text directly
            content = response_json
        
        if not expect_json:
            return content
            
        try:
            parsed = json.loads(content)
            
            # If it's a batch response, extract the results array
            if is_batch and isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
            
            return parsed
        except json.JSONDecodeError:
            warnings.warn(
                "Model returned non-JSON despite schema constraint. Returning raw text.",
                UserWarning
            )
            return content


# === Template System ===

class Template:
    """Callable template with frozen parameters."""
    
    def __init__(
        self,
        template: str,
        llm: 'HuggingLLM',
        schema: Optional[Schema] = None,
        **fixed_params
    ):
        self.template = template
        self.llm = llm
        self.schema = schema
        self.fixed_params = fixed_params
    
    def __call__(self, **variables) -> Union[str, Dict[str, Any], List[Any]]:
        """Execute template with provided variables."""
        return self.llm.generate(
            self.template,
            variables=variables,
            schema=self.schema,
            **self.fixed_params
        )


# === Hugging Face Client Wrapper ===

class HuggingFaceClient:
    """Simple wrapper for HF API calls."""
    
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to HF endpoint."""
        response = requests.post(
            f"{self.endpoint}/generate",
            headers=self.headers,
            json=request,
            timeout=120  # 2 minutes timeout for longer generations
        )
        
        response.raise_for_status()
        return response.json()


# === Main LLM Interface ===

class HuggingLLM(BaseLLM):
    """
    Drop-in replacement for AzureLLM using Hugging Face TGI endpoints.
    
    Maintains identical interface to AzureLLM for seamless compatibility.
    
    Usage:
        # Simple usage with environment variables
        llm = HuggingLLM("jonsnow")  # Uses JONSNOW_ENDPOINT_URL from .env
        
        # Custom config
        llm = HuggingLLM("jonsnow", temperature=0.7)
        
        # Full control
        config = LLMConfig.from_env(deployment_name="jonsnow")
        llm = HuggingLLM(config=config)
        
        # Batch processing (identical to AzureLLM)
        items = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]
        results = llm.generate("Translate to Spanish", batch_items=items)
    """
    
    def __init__(
        self, 
        deployment_name: Optional[str] = None,
        *,
        config: Optional[LLMConfig] = None, 
        **config_overrides
    ):
        """
        Initialize with automatic environment config or custom settings.
        
        Args:
            deployment_name: Model name (e.g., "jonsnow") - used to find {MODEL}_ENDPOINT_URL
            config: Custom LLMConfig object (optional)
            **config_overrides: Override specific config values
        """
        if config is not None:
            # Use provided config
            self.config = config
        else:
            # Auto-configure from environment with overrides
            overrides = config_overrides.copy()
            if deployment_name is not None:
                overrides['deployment_name'] = deployment_name
                
            self.config = LLMConfig.from_env(**overrides)
        
        self._request_builder = RequestBuilder(self.config)
        self._processor = ResponseProcessor()
    
    @cached_property
    def client(self) -> HuggingFaceClient:
        """Lazy-initialized Hugging Face client."""
        return HuggingFaceClient(
            endpoint=self.config.endpoint,
            api_key=self.config.api_key
        )
    
    def generate(
        self,
        prompt: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], str, Schema]] = None,
        batch_items: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        # Backward compatibility parameters
        prompt_vars: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Union[Dict[str, Any], str, Schema]] = None
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Generate response with optional structured output and batch processing.
        
        Identical interface to AzureLLM.generate() for drop-in compatibility.
        
        Args:
            prompt: Text with optional {placeholder} variables
            variables: Values to substitute in prompt (or use prompt_vars for backward compatibility)
            schema: Output schema (dict, file path, or Schema object) (or use output_schema for backward compatibility)
            batch_items: List of dicts to process in batch
            max_tokens: Token limit
            temperature: Response randomness
            
        Returns:
            String for text output, dict for structured output, list for batch output
        """
        # Handle backward compatibility (identical to AzureLLM)
        if prompt_vars is not None and variables is None:
            variables = prompt_vars
        elif prompt_vars is not None and variables is not None:
            raise ValueError("Cannot specify both 'variables' and 'prompt_vars'. Use 'variables'.")
        
        if output_schema is not None and schema is None:
            schema = output_schema
        elif output_schema is not None and schema is not None:
            raise ValueError("Cannot specify both 'schema' and 'output_schema'. Use 'schema'.")
        
        # Substitute variables in prompt
        if variables:
            try:
                prompt = prompt.format_map(variables)
            except KeyError as e:
                raise KeyError(f"Missing template variable: {e}")
        
        # Handle schema
        schema_obj = None
        if schema is not None:
            schema_obj = schema if isinstance(schema, Schema) else Schema.load(schema)
            schema_obj.validate_if_enabled(self.config.validate_schema)
        
        # Build and execute request
        request = self._request_builder.build(
            prompt, 
            schema_obj, 
            max_tokens, 
            temperature,
            batch_items
        )
        
        try:
            response = self.client.generate(request)
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Hugging Face API error: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
        
        # Process response - expect JSON if schema or batch_items provided
        expect_json = schema_obj is not None or batch_items is not None
        is_batch = batch_items is not None
        return self._processor.process(response, expect_json=expect_json, is_batch=is_batch)
    
    def template(
        self,
        template_string: str,
        *,
        schema: Optional[Union[Dict[str, Any], str]] = None,
        **fixed_params
    ) -> Template:
        """
        Create reusable template with frozen parameters.
        
        Identical interface to AzureLLM.template() for drop-in compatibility.
        
        Args:
            template_string: Prompt template with {variable} placeholders
            schema: Output schema for structured responses
            **fixed_params: Fixed parameters (max_tokens, temperature, etc.)
            
        Returns:
            Callable template object
        """
        schema_obj = Schema.load(schema) if schema is not None else None
        return Template(template_string, self, schema_obj, **fixed_params)


# === Convenience Functions (identical to azure.py) ===

def create_llm(deployment_name: Optional[str] = None, **config_overrides) -> HuggingLLM:
    """Create LLM instance with environment config and overrides."""
    return HuggingLLM(deployment_name, **config_overrides)


def quick_generate(prompt: str, deployment_name: str = "default", **kwargs) -> Union[str, Dict[str, Any], List[Any]]:
    """One-shot generation with automatic config."""
    llm = create_llm(deployment_name)
    return llm.generate(prompt, **kwargs)


# === Clean Aliases (identical to azure.py) ===

# Clean alias for even more intuitive usage
Hugging = HuggingLLM