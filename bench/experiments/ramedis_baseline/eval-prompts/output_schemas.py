from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DDXSchema:
    """Schema for differential diagnosis response"""
    diagnoses: List[str]  # List of exactly 5 diagnoses


@dataclass 
class SeveritySchema:
    """Schema for severity assignment response"""
    severities: Dict[str, str]  # {"diagnosis_name": "S0-S10"}


# Alternative dict-based schemas for direct use
DDX_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 5,
            "description": "List of exactly 5 differential diagnoses"
        }
    },
    "required": ["diagnoses"],
    "additionalProperties": False
}

SEVERITY_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "severities": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "pattern": "^S([0-9]|10)$",
                "description": "Severity level from S0 to S10"
            },
            "description": "Mapping of diagnosis names to severity levels"
        }
    },
    "required": ["severities"],
    "additionalProperties": False
}