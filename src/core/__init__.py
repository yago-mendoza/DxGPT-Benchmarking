"""Core components for the evaluation framework."""

from .session import EvaluationSession
from .registry import EvaluatorRegistry

__all__ = ['EvaluationSession', 'EvaluatorRegistry']