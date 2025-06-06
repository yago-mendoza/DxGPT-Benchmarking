"""
Evaluator Registry

Manages available evaluators and their configurations.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    # Expected CSV schema - subclasses should override
    EXPECTED_COLUMNS = []
    COLUMN_DESCRIPTIONS = {}
    
    @abstractmethod
    def evaluate(self, dataset_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation on dataset."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get evaluator name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get evaluator description."""
        pass
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """Validate dataset has expected columns."""
        import csv
        
        try:
            with open(dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check required columns exist
                for col in self.EXPECTED_COLUMNS:
                    if col not in headers:
                        raise ValueError(f"Missing required column: {col}")
                
            return True
        except Exception as e:
            print(f"Dataset validation failed: {e}")
            return False


class EvaluatorRegistry:
    """Registry for available evaluators."""
    
    def __init__(self):
        self._evaluators: Dict[str, Type[BaseEvaluator]] = {}
        self._discover_evaluators()
    
    def _discover_evaluators(self):
        """Discover evaluators in the evaluators directory."""
        evaluators_dir = Path(__file__).parent.parent / "evaluators"
        
        if not evaluators_dir.exists():
            evaluators_dir.mkdir(exist_ok=True)
            return
        
        # Import all Python files in evaluators directory
        for py_file in evaluators_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            module_name = f"evaluators.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
                
                # Find BaseEvaluator subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseEvaluator) and 
                        obj != BaseEvaluator):
                        evaluator_name = obj().get_name()
                        self._evaluators[evaluator_name] = obj
                        
            except Exception as e:
                print(f"Failed to load evaluator from {py_file}: {e}")
    
    def register(self, evaluator_class: Type[BaseEvaluator]):
        """Manually register an evaluator."""
        instance = evaluator_class()
        self._evaluators[instance.get_name()] = evaluator_class
    
    def get_evaluator(self, name: str) -> Optional[BaseEvaluator]:
        """Get evaluator instance by name."""
        if name in self._evaluators:
            return self._evaluators[name]()
        return None
    
    def list_evaluators(self) -> List[Dict[str, str]]:
        """List all available evaluators."""
        evaluators = []
        for name, evaluator_class in self._evaluators.items():
            instance = evaluator_class()
            evaluators.append({
                'name': name,
                'description': instance.get_description(),
                'expected_columns': instance.EXPECTED_COLUMNS,
                'column_descriptions': instance.COLUMN_DESCRIPTIONS
            })
        return evaluators