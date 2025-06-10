"""
FILE : base.py
Base class for all LLM implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], str]] = None,
        batch_items: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def template(
        self,
        template_string: str,
        *,
        schema: Optional[Union[Dict[str, Any], str]] = None,
        **fixed_params
    ):
        """Create a reusable template."""
        pass 